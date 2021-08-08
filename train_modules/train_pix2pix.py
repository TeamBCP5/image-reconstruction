import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import (
    train_valid_split,
    cut_img,
    get_train_transform,
    get_valid_transform,
    Pix2PixDataset,
    CutImageDataset,
)
from utils import (
    get_optimizer,
    get_scheduler,
    get_model,
    set_seed,
    print_system_envs,
    psnr_score,
    GANLoss,
)
from networks import set_requires_grad


def train(args):
    print(
        f"<< Train Pix2Pix >>\n",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_envs()
    set_seed(args.seed)
    os.makedirs(args.checkpoint.save_dir, exist_ok=True)

    # compose dataset
    (
        train_input_paths,
        train_label_paths,
        valid_input_paths,
        valid_label_paths,
    ) = train_valid_split(
        args.data.dir, valid_type=args.data.valid_type, full_train=args.data.full_train
    )

    cut_img(
        img_path_list=train_input_paths,
        label_path_list=train_label_paths,
        save_dir=args.data.dir,
        stride=args.data.stride,
        patch_size=args.data.patch_size,
    )

    train_transform = get_train_transform(args.network.name)
    valid_transform = get_valid_transform(args.network.name)
    train_dataset = Pix2PixDataset(
        args.data.dir, args.data.patch_size, train_transform, "train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # compose train components
    G_model, D_model = get_model(args, mode="train")
    G_model.to(device)
    D_model.to(device)

    G_optimizer, D_optimizer = get_optimizer(args, G_model, D_model)
    G_scheduler, D_scheduler = get_scheduler(args, G_optimizer, D_optimizer)

    gan_loss_fn = GANLoss().to(device)
    l1_loss_fn = nn.L1Loss().to(device)

    start_epoch = 0
    if args.checkpoint.load_path is not None:
        ckpt = torch.load(args.checkpoint.load_path)
        start_epoch = ckpt["epoch"] + 1
        G_model.load_state_dict(ckpt["G_model"])
        D_model.load_state_dict(ckpt["D_model"])
        G_optimizer.load_state_dict(ckpt["G_optimizer"])
        D_optimizer.load_state_dict(ckpt["D_optimizer"])
        G_scheduler.load_state_dict(ckpt["G_scheduler"])
        D_scheduler.load_state_dict(ckpt["D_scheduler"])
        print(
            f"[+] Checkpoint\n",
            f"'{args.checkpoint.load_path}' loaded\n",
            f"Resume from epoch {start_epoch+1}\n",
        )

    # train
    best_score = 0
    for epoch in range(start_epoch, args.epochs):
        G_model.train()
        D_model.train()

        train_loss_D = []
        train_loss_G = []
        train_loss_l1 = []
        train_loss_gan = []
        for img, label in tqdm(train_loader, desc="[Train]"):
            img, label = img.float().to(device), label.float().to(device)

            fake_img = G_model(img)

            ## Step 1 : train discriminator
            set_requires_grad(D_model, True)
            D_optimizer.zero_grad()

            # Train with fake
            fake_set = torch.cat([img, fake_img], dim=1)
            pred_fake = D_model.forward(fake_set.detach())
            loss_D_fake = gan_loss_fn(pred_fake, False)

            # Train with real
            real_set = torch.cat([img, label], dim=1)
            pred_real = D_model.forward(real_set)
            loss_D_real = gan_loss_fn(pred_real, True)

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
            loss_G_gan = gan_loss_fn(pred_fake, True)

            # Second, G(A) = B
            loss_G_L1 = l1_loss_fn(fake_img, label)
            loss_G = loss_G_gan + loss_G_L1 * 100.0
            loss_G.backward()
            G_optimizer.step()

            train_loss_D.append(loss_D.item())
            train_loss_G.append(loss_G.item())

            train_loss_l1.append(loss_G_L1.item())
            train_loss_gan.append(loss_G_gan.item())

        train_loss_D = np.mean(train_loss_D)
        train_loss_G = np.mean(train_loss_G)
        train_loss_l1 = np.mean(train_loss_l1)
        train_loss_gan = np.mean(train_loss_gan)

        valid_psnr = validation(
            G_model,
            valid_input_paths,
            valid_label_paths,
            patch_size=args.data.patch_size,
            stride=args.data.stride,
            transforms=valid_transform,
            device=device,
        )

        # update scheduler
        G_scheduler.step(valid_psnr)
        D_scheduler.step(valid_psnr)

        print(
            f"[+] Epoch: {epoch}/{args.epochs}\n",
            f"Valid PSNR: {valid_psnr:.4f}\n",
            f"Train Generator Loss: {train_loss_G:.4f}\n",
            f" * L1 Loss: {train_loss_l1*100:.4f}\n",
            f" * GAN Loss: {train_loss_gan:.4f}\n",
            f"Train Discriminator Loss: {train_loss_D:.4f}\n",
        )

        # save best model
        if best_score < valid_psnr:
            best_score = valid_psnr
            ckpt_save_path = os.path.join(
                args.checkpoint.save_dir, "ckpt_best_pix2pix.pth"
            )
            ckpt = dict(
                epoch=epoch,
                G_model=G_model.state_dict(),
                G_optimizer=G_optimizer.state_dict(),
                G_scheduler=G_scheduler.state_dict(),
                D_model=D_model.state_dict(),
                D_optimizer=D_optimizer.state_dict(),
                D_scheduler=D_scheduler.state_dict(),
            )
            torch.save(ckpt, ckpt_save_path)
            print(
                f"[+] Best Score Updated!\n",
                f"Best PSNR: {best_score: .4f}\n",
                f"Checkpoint saved: '{ckpt_save_path}'\n",
            )


def validation(
    model,
    img_paths: list,
    label_paths: list,
    patch_size: int = 512,
    stride: int = 256,
    transforms=None,
    device=None,
):
    model.eval()
    batch_size = 32

    valid_psnr_list = []
    with torch.no_grad():
        for img_path, lbl_path in tqdm(
            zip(img_paths, label_paths), desc="[Validation]"
        ):
            ds = CutImageDataset(
                img_path, patch_size=patch_size, stride=stride, transforms=transforms
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

            # main light scattering reduction(pix2pix)
            preds = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            votes = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            for images, (x1, x2, y1, y2) in dl:
                pred = model(images.to(device).float())  # [C, W, H]
                pred = pred * 127.5 + 127.5
                for i in range(len(x1)):
                    preds[:, x1[i] : x2[i], y1[i] : y2[i]] += pred[i]
                    votes[:, x1[i] : x2[i], y1[i] : y2[i]] += 1
            preds /= votes
            preds = preds.cpu().detach().numpy().astype(np.uint8)
            preds = preds.transpose(1, 2, 0)
            preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)

            label = cv2.imread(lbl_path)
            valid_psnr_list.append(psnr_score(preds.astype(float), label.astype(float)))

    valid_psnr = np.mean(valid_psnr_list)
    return valid_psnr
