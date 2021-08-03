import os
import zipfile
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import wandb

from utils import seed_everything, remove_all_files_in_dir, save_samples
from metrics import psnr_score
from dataset import train_valid_split, train_valid_unseen_split, ImageDataset
from preprocessing import cut_img
from model_utils import *


def train(args):
    seed_everyting(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_save_dir, exist_ok=True)
    test_samples = glob()

    if args.contain_unseen:
        (
            train_input_paths,
            train_label_paths,
            valid_input_paths,
            valid_label_paths,
            unseen_input_paths,
            unseen_label_paths,
        ) = train_valid_unseen_split(args.data_dir, full_train=args.full_train)
    else:
        (
            train_input_paths,
            train_label_paths,
            valid_input_paths,
            valid_label_paths,
        ) = train_valid_split(args.data_dir, full_train=args.full_train)

    cut_img(
        train_input_paths,
        args.stride,
        args.split_size,
        args.train_multiscale_mode,
        args.denoise_method,
        save_path=os.path.join(args.save_dir, "train_input_img"),
    )
    cut_img(
        train_label_paths,
        args.stride,
        args.split_size,
        args.train_multiscale_mode,
        args.denoise_method,
        save_path=os.path.join(args.save_dir, "train_label_img"),
    )

    train_transform = get_train_transform()
    valid_transform = get_valid_transform()
    train_dataset = Pix2PixDataset(
        args.data_dir, "Train", args.split_size, transforms=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    # define model
    G_model, D_model = get_model("pix2pix")
    G_model.to(device)
    D_model.to(device)
    wandb.watch(G_model)
    wandb.watch(D_model)

    # define optimizers & scheduler
    G_params = [p for p in G_model.parameters() if p.requires_grad]
    D_params = [p for p in D_model.parameters() if p.requires_grad]
    G_optimizer = optim.Adam(G_params, lr=args.lr_G)
    D_optimizer = optim.Adam(D_params, lr=args.lr_D)
    G_scheduler = ReduceLROnPlateau(
        G_optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold_mode="abs",
        min_lr=1e-8,
        verbose=True,
    )
    D_scheduler = ReduceLROnPlateau(
        D_optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold_mode="abs",
        min_lr=1e-8,
        verbose=True,
    )

    start_epoch = 0
    if args.ckpt_load_path is not None:
        ckpt = torch.load(args.ckpt_load_path)
        start_epoch = ckpt["epoch"]
        G_model.load_state_dict(ckpt["G_model"])
        D_model.load_state_dict(ckpt["D_model"])
        G_optimizer.load_state_dict(ckpt["G_optimizer"])
        D_optimizer.load_state_dict(ckpt["D_optimizer"])
        G_scheduler.load_state_dict(ckpt["G_scheduler"])
        D_scheduler.load_state_dict(ckpt["D_scheduler"])
        print(
            f"Checkpoint {args.ckpt_load_path} loaded! Resume train from epoch {start_epoch+1}"
        )

    # define criterions
    gan_loss_fn = GANLoss().to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    best_score = 0
    for epoch in range(start_epoch, epochs):
        G_model.train()
        D_model.train()
        train_loss_D = []
        train_loss_G = []

        train_l1 = []
        train_gan = []
        for step, (img, label) in tqdm(enumerate(iter(train_loader)), desc="[Train]"):
            img, label = img.float().to(device), label.float().to(device)

            fake_img = G_model(img)

            ## Step 1 : train discriminator
            set_requires_grad(D_model, True)
            D_optimizer.zero_grad()

            # Train with fake
            fake_set = torch.cat([img, fake_img], dim=1)
            pred_fake = D_model.forward(fake_set.detach())
            loss_D_fake = GAN_loss(pred_fake, False)

            # Train with real
            real_set = torch.cat([img, label], dim=1)
            pred_real = D_model.forward(real_set)
            loss_D_real = GAN_loss(pred_real, True)

            # Combine D Loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            D_optimizer.step()

            ## Step 2 : G_model train
            set_requires_grad(D_model, False)
            G_optimizer.zero_grad()

            # First, G(A) should fake the discriminator
            fake_set = torch.cat([img, fake_img], dim=1)
            pred_fake = D_model.forward(fake_set)
            loss_G_gan = GAN_loss(pred_fake, True)

            # Second, G(A) = B
            loss_G_L1 = L1_loss(fake_img, label)
            loss_G = loss_G_gan + loss_G_L1 * 100.0
            loss_G.backward()
            G_optimizer.step()

            train_loss_D.append(loss_D.item())
            train_loss_G.append(loss_G.item())

            train_l1.append(loss_G_L1.item())
            train_gan.append(loss_G_gan.item())

        t_loss_D = np.mean(train_loss_D)
        t_loss_G = np.mean(train_loss_G)
        t_loss_l1 = np.mean(train_l1)
        t_loss_gan = np.mean(train_gan)

        val_score = validation(
            G_model,
            valid_input_paths,
            valid_label_paths,
            split_size=args.split_size,
            stride=args.stride,
            transforms=valid_transform,
            device=device,
        )
        if args.contain_unseen:
            unseen_score = validation(
                G_model,
                unseen_input_paths,
                unseen_label_paths,
                split_size=args.split_size,
                stride=args.stride,
                transforms=valid_transform,
                device=device,
            )

        # verbose train result
        logs = dict(
            epoch=epoch,
            G_loss=t_loss_G,
            D_loss=t_loss_D,
            L1_loss=t_loss_l1 * 100,
            GAN_loss=t_loss_gan,
            Val_PSNR=val_score,
        )
        verbose = [
            f"[Epoch {epoch}/{args.epochs}]",
            f"Train G_loss: {t_loss_G:.5f} Train D_loss: {t_loss_D:.5f} Valid PSNR: {val_score:.5f}",
            f"Generator Loss - L1 loss: [{t_loss_l1*100:.5f}], GAN loss: [{t_loss_gan:.5f}]\n",
        ]
        if args.contain_unseen:
            logging_dict["Unseen_psnr"] = unseen_score
            verbose[1] += f" Unseen PSNR: {unseen_score:.5f}"

        wandb.log(logs)
        print("\n".join(verbose))

        # update scheduler
        G_scheduler.step(val_score)
        D_scheduler.step(val_score)

        # save best generator
        if best_score < val_score:
            best_score = val_score
            print(f"Best Score! {best_score: .2f}")
            best_save_path = os.path.join(
                args.ckpt_save_dir, "best_generator_pix2pix.pth"
            )
            torch.save(
                G_model.state_dict(),
                best_save_path,
                _use_new_zipfile_serialization=False,
            )

        if val_score >= 30.0:
            ckpt = dict(
                epoch=epoch,
                G_model=G_model.state_dict(),
                G_optimizer=G_optimizer.state_dict(),
                G_scheduler=G_scheduler.state_dict(),
                D_model=D_model.state_dict(),
                D_optimizer=D_optimizer.state_dict(),
                D_scheduler=D_scheduler.state_dict(),
            )
            ckpt_save_path = os.path.join(
                args.ckpt_save_dir, f"ckpt_ep{epoch:0>2d}(val{val_score:.2f}).pth"
            )
            torch.save(ckpt, ckpt_save_path)

        test_samples = predict_test_sample(
            G_model, stride=args.stride, transforms=valid_transform, device=device
        )
        save_samples(test_samples, epoch, save_dir=args.test_sample_save_dir)


def validation(
    model,
    img_paths,
    label_paths,
    split_size=512,
    stride=256,
    transforms=None,
    device=None,
):
    model.eval()

    results = []
    score = []
    input_score = []
    with torch.no_grad():
        for img_path, label_path in tqdm(
            zip(img_paths, label_paths), desc="[Validation]"
        ):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

            input_score.append(psnr_score(img.astype(float), label.astype(float), 255))
            img = transforms(image=img)["image"]

            crop = []
            position = []

            result_img = np.zeros_like(img.numpy().transpose(1, 2, 0))
            voting_mask = np.zeros_like(img.numpy().transpose(1, 2, 0))

            img = img.unsqueeze(0).float()
            for top in range(0, img.shape[2], stride):
                for left in range(0, img.shape[3], stride):
                    if (
                        top + split_size > img.shape[2]
                        or left + split_size > img.shape[3]
                    ):
                        continue
                    piece = torch.zeros([1, 3, split_size, split_size])
                    temp = img[:, :, top : top + split_size, left : left + split_size]
                    piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                    pred = model(piece.to(device))
                    pred = pred[0].cpu().clone().detach().numpy()
                    pred = pred.transpose(1, 2, 0)
                    pred = pred * 127.5 + 127.5

                    crop.append(pred)
                    position.append([top, left])

            # 가장 자리 1
            for left in range(0, img.shape[3], stride):
                if left + split_size > img.shape[3]:
                    continue
                piece = torch.zeros([1, 3, split_size, split_size])
                temp = img[
                    :,
                    :,
                    img.shape[2] - split_size : img.shape[2],
                    left : left + split_size,
                ]
                piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                with torch.no_grad():
                    pred = model(piece.to(device))

                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = pred * 127.5 + 127.5

                crop.append(pred)
                position.append([img.shape[2] - split_size, left])

            # 가장 자리 2
            for top in range(0, img.shape[2], stride):
                if top + split_size > img.shape[2]:
                    continue
                piece = torch.zeros([1, 3, split_size, split_size])

                temp = img[
                    :,
                    :,
                    top : top + split_size,
                    img.shape[3] - split_size : img.shape[3],
                ]
                piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                with torch.no_grad():
                    pred = model(piece.to(device))

                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = pred * 127.5 + 127.5

                crop.append(pred)
                position.append([top, img.shape[3] - split_size])

            # 오른쪽 아래
            piece = torch.zeros([1, 3, split_size, split_size])
            temp = img[
                :,
                :,
                img.shape[2] - split_size : img.shape[2],
                img.shape[3] - split_size : img.shape[3],
            ]
            piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

            with torch.no_grad():
                pred = model(piece.to(device))

            pred = pred[0].cpu().clone().detach().numpy()
            pred = pred.transpose(1, 2, 0)
            pred = pred * 127.5 + 127.5

            crop.append(pred)
            position.append([img.shape[2] - split_size, img.shape[3] - split_size])

            # 취합
            crop = np.array(crop).astype(np.float32)
            for num, (t, l) in enumerate(position):
                piece = crop[num]
                h, w, c = result_img[t : t + split_size, l : l + split_size, :].shape
                result_img[t : t + split_size, l : l + split_size, :] += piece[
                    :h, :w, :
                ]
                voting_mask[t : t + split_size, l : l + split_size, :] += 1

            result_img = result_img / voting_mask
            result_img = result_img.astype(np.uint8)
            results.append(result_img)
            score.append(psnr_score(result_img.astype(float), label.astype(float), 255))

    return np.mean(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="[YOUR_PROJECT_NAME]")
    parser.add_argument("--exp-name", default="[YOUR_EXPERIMENT_NAME]")
    parser.add_argument("--network", default="pix2pix")
    parser.add_argument("--data-dir", default="/content/data/")
    parser.add_argument("--full-train", default=True)
    parser.add_argument("--contain-unseen", default=True)
    parser.add_argument("--stride", default=256)
    parser.add_argument("--split-size", default=512)
    parser.add_argument("--train-multiscale-mode", default=True)
    parser.add_argument(
        "--denoise-method", default=None, help="None, 'naive', 'selective'"
    )
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch-size", default=10)
    parser.add_argument("--lr-G", default=2e-4)
    parser.add_argument("--lr-D", default=2e-4)
    parser.add_argument("--ckpt-load-path", default=None)
    parser.add_argument("--ckpt-save-dir", default="./checkpoints/")
    parser.add_argument("--test-image-dir", default="./inference_sample/")
    parser.add_argument("--test-sample-save-dir", default="./inference_sample/")
    parser.add_argument("--seed", default=41)
    args = parser.parse_args()

    print("=" * 50)
    print(args)
    print("=" * 50)

    train(args)
