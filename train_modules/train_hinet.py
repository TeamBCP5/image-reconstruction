import os
import argparse
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import wandb
from networks.HINet import HINet
from utils.criterions import HINetLoss
from utils.utils import get_model, seed_everything
from utils.metrics import psnr_score
from data.dataset import HINetDataset, train_valid_split
from data.augmentations import get_train_transform, get_valid_transform
from inference_hinet import inference

# temporary
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transform():
    return A.Compose(
        [
            A.Resize(1224, 1632),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ],
        additional_targets={"label": "image"},
    )

def get_valid_transform():
    return A.Compose(
        [A.Resize(1224, 1632), ToTensorV2(p=1.0)], additional_targets={"label": "image"}
    )


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # compose dataset
    (
        train_input_paths,
        train_label_paths,
        valid_input_paths,
        valid_label_paths,
    ) = train_valid_split(data_dir=args.data_dir, full_train=args.full_train)
    
    train_transform = get_train_transform()
    valid_transform = get_valid_transform()
    train_dataset = HINetDataset(
        train_input_paths, train_label_paths, transforms=train_transform, mode="Train"
    )
    valid_dataset = HINetDataset(
        valid_input_paths, valid_label_paths, transforms=valid_transform, mode="Valid"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # for inference
    test_input_paths = sorted(glob('/content/data/hinet_dataset/test_input_img/*'))
    test_dataset = HINetDataset(test_input_paths, transforms=valid_transform, mode='Test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # define model
    model = get_model(model_type="hinet")
    model.to(device)
    wandb.watch(model)

    # define optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold_mode="abs",
        min_lr=1e-8,
        verbose=True,
    )

    # define criterion
    criterion = HINetLoss().to(device)

    start_epoch = 0
    if args.ckpt_load_path is not None:
        ckpt = torch.load(args.ckpt_load_path)
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Checkpoint loaded. Epoch: {start_epoch}")

    best_score = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_loss_list = []
        for img, label in tqdm(train_loader, desc="[Train]"):
            img, label = img.float().to(device), label.float().to(device)

            with autocast():
                pred = model(img)
                loss = criterion(pred, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.step(optimizer)
            scaler.update()
            train_loss_list.append(loss.item())

        valid_psnr, valid_psnr_adjusted = validation(model, valid_loader, device=device)
        train_loss = sum(train_loss_list) / len(train_loss_list)

        print(
            f"Epoch [{epoch}/{args.epochs}], Train loss: [{train_loss:.5f}] Valid PSNR: [{valid_psnr:.5f}] Valid PSNR(Adjusted): [{valid_psnr_adjusted:.5f}]\n"
        )
        scheduler.step(valid_psnr)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "Val_PSNR": valid_psnr,
                "Val_PSNR_adjusted": valid_psnr_adjusted,
                
            }
        )

        if best_score < valid_psnr:
            print(f"Best Score! Valid PSNR {valid_psnr:.5f}")
            best_score = valid_psnr
            ckpt = dict(
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            ckpt_save_path = os.path.join(
                args.ckpt_save_dir, f"best_hinet.pth"
            )
            torch.save(ckpt, ckpt_save_path)

        if valid_psnr >= 36.0:
            ckpt = dict(
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            ckpt_save_path = os.path.join(
                args.ckpt_save_dir, f"hinet_ep{epoch}_val({valid_psnr:.2f}).pth"
            )
            torch.save(ckpt, ckpt_save_path)
            print(f"Epoch {epoch} Model saved. Valid PSNR: {valid_psnr:.5f}")
            inference(model, epoch, test_loader, device, args.inference_save_dir)


def validation(model, valid_loader, device):
    model.eval()
    psnr_basic_list, psnr_adjusted_list = [], []

    with torch.no_grad():
        for img, label in tqdm(valid_loader, desc="[Validation]"):
            label = label.clone().detach().numpy().squeeze(0)
            h, w, _ = label.shape

            pred = model(img.float().to(device))[0]
            pred = torch.clamp(pred[-1], 0, 1)
            pred = pred.cpu().clone().detach().numpy()
            pred = pred * 255.0

            # basic psnr
            pred_basic = pred.copy().transpose(1, 2, 0)
            pred_basic = cv2.resize(
                pred_basic, dsize=(w, h), interpolation=cv2.INTER_CUBIC
            )
            pred_basic = np.clip(pred_basic, 0, 255)
            pred_basic = pred_basic.astype(np.uint8)
            psnr_basic = psnr_score(pred_basic.astype(float), label.astype(float), 255)

            # adjusted psnr - regularize G channel
            pred_adjusted = pred.copy().transpose(1, 2, 0)
            pred_adjusted[:, :, 1] = np.where(
                pred_adjusted[:, :, 1] > 239, 239, pred_adjusted[:, :, 1]
            )
            pred_adjusted = cv2.resize(
                pred_adjusted, dsize=(w, h), interpolation=cv2.INTER_CUBIC
            )
            pred_adjusted = np.clip(pred_adjusted, 0, 255)
            pred_adjusted = pred_adjusted.astype(np.uint8)
            psnr_adjusted = psnr_score(
                pred_adjusted.astype(float), label.astype(float), 255
            )

            psnr_basic_list.append(psnr_basic)
            psnr_adjusted_list.append(psnr_adjusted)

    valid_psnr = sum(psnr_basic_list) / len(psnr_basic_list)  # average
    valid_psnr_adjusted = sum(psnr_adjusted_list) / len(psnr_adjusted_list)  # average

    return valid_psnr, valid_psnr_adjusted


if __name__ == "__main__":
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="postprocessor")
    parser.add_argument("--exp-name", default="HINet & Immature Pix2Pix Dataset")
    parser.add_argument("--network", default="hinet")
    parser.add_argument("--data-dir", default="/content/data/hinet_dataset")
    parser.add_argument("--test-data-dir", default="/content/data/hinet_dataset")
    parser.add_argument("--inference-save-dir", default="./inference/")
    parser.add_argument("--full-train", default=True)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch-size", default=1)
    parser.add_argument("--lr", default=25e-6)
    parser.add_argument("--ckpt-load-path", default="./checkpoints/hinet_ep3_val(34.22).pth")
    parser.add_argument("--ckpt-save-dir", default="./checkpoints/")
    parser.add_argument("--seed", default=41)
    args = parser.parse_args()

    print("=" * 50)
    print(args)
    print("=" * 50)

    run = wandb.init(project=args.project, name=args.exp_name)
    wandb.config.update(args)
    train(args)
    run.finish()
    
