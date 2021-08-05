import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from networks import HINet
from utils import (
    Flags,
    psnr_score,
    HINetLoss,
    get_model,
    get_optimizer,
    get_scheduler,
    set_seed,
    print_system_envs,
)
from data import (
    HINetDataset,
    train_valid_split,
    compose_postprocessing_dataset,
    get_train_transform,
    get_valid_transform,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_envs()
    set_seed(args.seed)
    os.makedirs(args.checkpoint.save_dir, exist_ok=True)
    os.makedirs(args.data.dir, exist_ok=True)

    compose_postprocessing_dataset(args, device)

    # compose dataset
    (
        train_input_paths,
        train_label_paths,
        valid_input_paths,
        valid_label_paths,
    ) = train_valid_split(
        args.data.dir, valid_type=args.data.valid_type, full_train=args.data.full_train
    )

    train_transform = get_train_transform(args.network.name)
    valid_transform = get_valid_transform(args.network.name)
    train_dataset = HINetDataset(
        train_input_paths, train_label_paths, train_transform, mode="train"
    )
    valid_dataset = HINetDataset(
        valid_input_paths, valid_label_paths, valid_transform, mode="valid"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # compose train components
    model = get_model(args)
    model.to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    criterion = HINetLoss().to(device)

    start_epoch = 0
    if args.checkpoint.load_path is not None:
        ckpt = torch.load(args.checkpoint.load_path)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(
            f"[+] Checkpoint\n",
            f"'{args.checkpoint.load_path}' loaded\n",
            f"Resume from epoch {start_epoch}\n",
        )

    scaler = GradScaler()
    best_score = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_loss_list = []
        for img, label in tqdm(train_loader, desc="[Train]"):
            img, label = img.float().to(device), label.float().to(device)
            optimizer.zero_grad()

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
            f"[+] Epoch: {epoch}/{args.epochs}\n",
            f"* Valid PSNR: {valid_psnr:.4f}\n",
            f"* Valid PSNR(adjusted): {valid_psnr_adjusted:.4f}\n",  # clipping green channel
            f"* Train Loss: {train_loss:.4f}\n",
        )

        scheduler.step(valid_psnr)

        if best_score < valid_psnr:
            best_score = valid_psnr
            ckpt = dict(
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            ckpt_save_path = os.path.join(
                args.checkpoint.save_dir, f"ckpt_best_hinet.pth"
            )
            torch.save(ckpt, ckpt_save_path)
            print(
                f"[+] Best Score Updated!\n",
                f"Best PSNR: {best_score: .4f}\n",
                f"Checkpoint saved: '{ckpt_save_path}'\n",
            )


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
