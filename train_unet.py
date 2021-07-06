import argparse
from glob import glob
import time
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16
import segmentation_models_pytorch as smp

from mlp_mixer import MLPMixer
from utils import seed_everything, print_gpu_status
from augmentations import get_train_transform, get_valid_transform
from dataset import train_valid_split, TrainShiftedDataset, ValidShiftedDataset
from validation import validate
from preprocessing import get_save_shifted_images
import wandb


def initialize_wandb(
    args,
    project="LightScatteringReduction",
    name="valid_type1",
    entity="smbly",
    network="UNet",
):
    wandb.init(project=project, name=name, entity=entity)
    config = wandb.config
    config.lr = args.lr
    config.batch_size = args.train_batch_size
    config.network = network


def train(args):
    initialize_wandb(args)
    seed_everything(41)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_input_paths,
        train_label_paths,
        valid_input_paths,
        valid_label_paths,
    ) = train_valid_split(
        args.data_dir, "./configs/train_meta.csv", valid_type=1, full_train=True
    )

    train_input_dir = os.path.join(args.data_dir, "train_shifted_inputs")
    train_label_dir = os.path.join(args.data_dir, "train_shifted_labels")
    valid_input_dir = os.path.join(args.data_dir, "valid_shifted_inputs")
    valid_label_dir = os.path.join(args.data_dir, "valid_shifted_labels")
    get_save_shifted_images(train_input_paths, save_dir=train_input_dir)
    get_save_shifted_images(train_label_paths, save_dir=train_label_dir)
    get_save_shifted_images(valid_input_paths, save_dir=valid_input_dir)
    get_save_shifted_images(valid_label_paths, save_dir=valid_label_dir)

    train_transform = get_train_transform()
    valid_transform = get_valid_transform()

    train_dataset = TrainShiftedDataset(
        source_dir=train_input_dir,
        label_dir=train_label_dir,
        transforms=train_transform,
    )
    valid_dataset = ValidShiftedDataset(
        source_dir=valid_input_dir,
        label_dir=valid_label_dir,
        transforms=valid_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False, drop_last=True
    )

    total_train_steps = len(train_loader) * args.epochs
    total_valid_steps = len(valid_loader)
    print(
        "[+] Train Description\n",
        f"Epochs: {args.epochs}\n",
        f"Train Batch Size: {args.train_batch_size}\n",
        f"Valid Batch Size: {args.valid_batch_size}\n",
        f"Train Data Size: {len(train_dataset)}\n",
        f"Valid Data Size: {len(valid_dataset)}\n",
        f"Total Train Steps: {total_train_steps}\n",
        f"Total Valid Steps: {total_valid_steps}\n",
    )

    model = smp.Unet(
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        decoder_attention_type="scse",
    )
    model.segmentation_head[2] = nn.Tanh()
    model.eval()
    model.to(device)

    num_params = sum([p.numel() for p in model.parameters()])
    print("[+] Model Description\n", f"Number of Params: {num_params}\n")
    print_gpu_status()
    wandb.watch(model)

    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_train_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )

    best_score = 0
    for epoch in range(args.epochs):

        time_check = time.time()
        model.train()
        train_loss = []

        for batch in tqdm(train_loader, desc="[Train]"):
            images = batch["image"]
            labels = batch["label"].to(device)

            preds = model(images.to(device))
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if isinstance(scheduler.get_last_lr(), list):
                wandb.log({"lr": scheduler.get_last_lr()[0]})
            else:
                wandb.log({"lr": scheduler.get_last_lr()})

            train_loss.append(loss.item())

        min_per_epoch = (time.time() - time_check) / 60  # MPE
        sec_per_batch = (time.time() - time_check) / len(train_loader)  # SPB

        time_check = time.time()
        valid_score, valid_psnr_each_sample = validate(
            model,
            valid_dataloader=valid_loader,
            device=device,
        )
        sec_per_inference = (time.time() - time_check) / len(valid_input_paths)
        print(
            f"[Epoch: {epoch+1}/{args.epochs}] Valid PSNR: {valid_score: .3f} Train Loss: {np.mean(train_loss): .3f}"
        )
        wandb.log(
            dict(
                epoch=epoch,
                valid_psnr=valid_score,
                train_loss=np.mean(train_loss),
                minutes_per_epoch=min_per_epoch,  # MPE
                seconds_per_batch=sec_per_batch,  # SPB
                seconds_per_inference=sec_per_inference,  # SPI
            )
        )
        # wandb.log(valid_psnr_each_sample) # TODO: 단일 이미지별 PSNR

        if best_score < valid_score:
            ckpt = {'model': model.state_dict(), 'scheduler': scheduler.state_dict(), 'optimier': optimizer.state_dict()}
            torch.save(ckpt, f"best_shifted_unet.pth")
            best_score = valid_score
            print(f"Best model saved: # Epoch {epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/content/data/")
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--train-batch-size", default=8)
    parser.add_argument("--valid-batch-size", default=32)
    parser.add_argument("--ckpt-path", default="./saved/unet-ckpt.pth")

    args = parser.parse_args()
    train(args)
