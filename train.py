import argparse
from glob import glob
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import pickle
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16

from mlp_mixer import MLPMixer
from utils import seed_everything
from augmentations import get_train_transform, get_valid_transform
from dataset import CustomDataset
from train_utils import cut_img, EarlyStopping
from validation import validate


def train(args):
    seed_everything(41)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_csv = pd.read_csv(os.path.join(args.data_dir, "train.csv"))

    train_all_input_files = train_csv["input_img"].apply(
        lambda x: os.path.join(args.data_dir, "train_input_img", x)
    )
    train_all_label_files = train_csv["label_img"].apply(
        lambda x: os.path.join(args.data_dir, "train_label_img", x)
    )

    train_input_files = train_all_input_files.tolist()[60:]
    train_label_files = train_all_label_files.tolist()[60:]

    valid_input_files = train_all_input_files.tolist()[:60]
    valid_label_files = train_all_label_files.tolist()[:60]

    # cut_img(train_input_files, os.path.join(args.data_dir, 'train_input_img_'), img_size=args.img_size, stride=args.stride)
    # cut_img(train_label_files, os.path.join(args.data_dir, 'train_label_img_'), img_size=args.img_size, stride=args.stride)
    # cut_img(valid_input_files, os.path.join(args.data_dir, 'val_input_img_'), img_size=args.img_size, stride=args.stride)
    # cut_img(valid_label_files, os.path.join(args.data_dir, 'val_label_img_'), img_size=args.img_size, stride=args.stride)

    train_transform = get_train_transform()
    valid_transform = get_valid_transform()
    train_dataset = CustomDataset(args.data_dir, "Train", args.img_size, train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )

    model = MLPMixer(
        in_channels=3,
        hidden_dim=768,
        patch_size=16,
        image_size=args.img_size,
        num_blocks=12,
        token_dim=384,
        channel_dim=4096,
    )
    model.to(device)

    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of Params\n", num_params)

    criterion = nn.L1Loss().to(device)
    early_stop = EarlyStopping(path=args.ckpt_path)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, 
        total_steps=total_steps,
        pct_start=0.1,
        # steps_per_epoch=10, 
        # epochs=10,
        anneal_strategy='linear'
        )

    best_score = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for step, (img, label) in enumerate(tqdm(train_loader)):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())

        valid_score, _, _ = validate(
            model,
            valid_input_files,
            valid_label_files,
            stride=args.stride,
            img_size=args.img_size,
            transforms=valid_transform,
            device=device,
        )
        print(f'[Epoch: {epoch+1}/{args.epochs}] Valid PSNR: {valid_score: .3f} Train Loss: {np.mean(train_loss): .3f}')
        if best_score < valid_score:
            torch.save(model.state_dict(), f'best_mlpmixer.pth')
            best_score = valid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='/content/data/')
    parser.add_argument("--img-size", default=512)
    parser.add_argument("--stride", default=512)
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--train-batch-size", default=16)
    parser.add_argument("--valid-batch-size", default=32)
    parser.add_argument("--ckpt-path", default="./saved/ckpt.pth")

    args = parser.parse_args()
    train(args)
