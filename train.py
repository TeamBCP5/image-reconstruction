import argparse
from glob import glob
import time
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
from dataset import CustomDataset, train_valid_split
from train_utils import cut_img, EarlyStopping
from validation import validate
import wandb

def initialize_wandb(args, project='LightScatteringReduction', name='valid_type1', entity='smbly', network='MLPMixer'):
    wandb.init(project=project, name=name, entity=entity)
    config = wandb.config
    config.img_size = args.img_size
    config.lr = args.lr
    config.batch_size = args.train_batch_size
    config.network = network


def train(args):
    initialize_wandb(args)
    seed_everything(41)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_input_paths, train_label_paths, valid_input_paths, valid_label_paths = \
        train_valid_split(args.data_dir, './configs/train_meta.csv', valid_type=1, full_train=False)

    cut_img(train_input_paths, os.path.join(args.data_dir, 'train_input_img_'), img_size=args.img_size, stride=args.stride)
    cut_img(train_label_paths, os.path.join(args.data_dir, 'train_label_img_'), img_size=args.img_size, stride=args.stride)
    cut_img(valid_input_paths, os.path.join(args.data_dir, 'val_input_img_'), img_size=args.img_size, stride=args.stride)
    cut_img(valid_label_paths, os.path.join(args.data_dir, 'val_label_img_'), img_size=args.img_size, stride=args.stride)

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
    wandb.watch(model)

    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of Params\n", num_params)

    criterion = nn.L1Loss().to(device)
    # early_stop = EarlyStopping(path=args.ckpt_path)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, 
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
        )

    best_score = 0
    for epoch in range(args.epochs):
        time_check = time.time()
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

            if isinstance(scheduler.get_last_lr(), list):
                wandb.log({'lr': scheduler.get_last_lr()[0]})
            else:
                wandb.log({'lr': scheduler.get_last_lr()})

            train_loss.append(loss.item())

        min_per_epoch = (time.time() - time_check) / 60 # MPE
        sec_per_batch = (time.time() - time_check) / len(train_loader) # SPB

        time_check = time.time()
        valid_score, _, _ = validate(
            model,
            valid_input_paths,
            valid_label_paths,
            stride=args.stride,
            img_size=args.img_size,
            transforms=valid_transform,
            device=device,
        )
        sec_per_inference = (time.time() - time_check) / len(valid_input_paths)

        print(f'[Epoch: {epoch+1}/{args.epochs}] Valid PSNR: {valid_score: .3f} Train Loss: {np.mean(train_loss): .3f}')
        wandb.log(dict(
            epoch=epoch, 
            valid_psnr=valid_score, 
            train_loss=np.mean(train_loss),
            minutes_per_epoch=min_per_epoch, # MPE
            seconds_per_batch=sec_per_batch, # SPB
            seconds_per_inference=sec_per_inference # SPI
            ))
        
        if best_score < valid_score:
            torch.save(model.state_dict(), f'best_mlpmixer.pth')
            best_score = valid_score
            print(f'Best model saved: # Epoch {epoch+1}')

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
    