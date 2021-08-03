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
from torch.utils.data import DataLoader, Dataset
import wandb

from utils import seed_everything, remove_all_files_in_dir, save_samples
from metrics import psnr_score
from dataset import train_valid_split, train_valid_unseen_split, ImageDataset
from preprocessing import cut_img
from model_utils import *


def inference(model, img_paths: list, stride: int=256, transforms=None, device=None, batch=128):
    results = []
    img_paths = [
        "/content/data/test_input_img/test_input_20003.png",
        "/content/data/test_input_img/test_input_20004.png",
        "/content/data/test_input_img/test_input_20005.png",
        "/content/data/test_input_img/test_input_20006.png",
        "/content/data/test_input_img/test_input_20009.png",
        "/content/data/test_input_img/test_input_20015.png",
        "/content/data/test_input_img/test_input_20016.png",
    ]
    for img_path in tqdm(img_paths, desc="[Inference]"):
        batch_count = 0
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = img.copy()

        img = transforms(image=img)["image"]

        crop = []
        position = []

        result_img = np.zeros_like(img.numpy().transpose(1, 2, 0))
        voting_mask = np.zeros_like(img.numpy().transpose(1, 2, 0))

        img = img.unsqueeze(0)
        for top in range(0, img.shape[2], stride):
            for left in range(0, img.shape[3], stride):
                if top + img_size > img.shape[2] or left + img_size > img.shape[3]:
                    continue

                piece = torch.zeros([1, 3, img_size, img_size])

                temp = img[:, :, top : top + img_size, left : left + img_size]
                piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                with torch.no_grad():
                    pred = model(piece.to(device))

                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = pred * 127.5 + 127.5

                crop.append(pred)
                position.append([top, left])
                batch_count += 1

                if batch_count == batch:
                    crop = np.array(crop).astype(np.float32)
                    for num, (t, l) in enumerate(position):
                        piece = crop[num]
                        h, w, c = result_img[
                            t : t + img_size, l : l + img_size, :
                        ].shape
                        result_img[t : t + img_size, l : l + img_size, :] += piece[
                            :h, :w, :
                        ]
                        voting_mask[t : t + img_size, l : l + img_size, :] += 1
                    crop = []
                    position = []
                    batch_count = 0

        # 가장 자리 1
        for left in range(0, img.shape[3], stride):
            if left + img_size > img.shape[3]:
                continue
            piece = torch.zeros([1, 3, img_size, img_size])
            temp = img[
                :, :, img.shape[2] - img_size : img.shape[2], left : left + img_size
            ]
            piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

            with torch.no_grad():
                pred = model(piece.to(device))

            pred = pred[0].cpu().clone().detach().numpy()
            pred = pred.transpose(1, 2, 0)
            pred = pred * 127.5 + 127.5

            crop.append(pred)
            position.append([img.shape[2] - img_size, left])

            batch_count += 1

            if batch_count == batch:
                crop = np.array(crop).astype(np.float32)
                for num, (t, l) in enumerate(position):
                    piece = crop[num]
                    h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                    result_img[t : t + img_size, l : l + img_size, :] += piece[
                        :h, :w, :
                    ]
                    voting_mask[t : t + img_size, l : l + img_size, :] += 1
                crop = []
                position = []
                batch_count = 0

        # 가장 자리 2
        for top in range(0, img.shape[2], stride):
            if top + img_size > img.shape[2]:
                continue
            piece = torch.zeros([1, 3, img_size, img_size])

            temp = img[
                :, :, top : top + img_size, img.shape[3] - img_size : img.shape[3]
            ]
            piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

            with torch.no_grad():
                pred = model(piece.to(device))
            pred = pred[0].cpu().clone().detach().numpy()
            pred = pred.transpose(1, 2, 0)
            pred = (pred * 127.5) + 127.5

            crop.append(pred)
            position.append([top, img.shape[3] - img_size])
            batch_count += 1

            if batch_count == batch:
                crop = np.array(crop).astype(np.float32)
                for num, (t, l) in enumerate(position):
                    piece = crop[num]
                    h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                    result_img[t : t + img_size, l : l + img_size, :] += piece[
                        :h, :w, :
                    ]
                    voting_mask[t : t + img_size, l : l + img_size, :] += 1
                crop = []
                position = []
                batch_count = 0

        # 오른쪽 아래
        piece = torch.zeros([1, 3, img_size, img_size])
        temp = img[
            :,
            :,
            img.shape[2] - img_size : img.shape[2],
            img.shape[3] - img_size : img.shape[3],
        ]
        piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

        with torch.no_grad():
            pred = model(piece.to(device))

        pred = pred[0].cpu().clone().detach().numpy()
        pred = pred.transpose(1, 2, 0)
        pred = pred * 127.5 + 127.5

        crop.append(pred)
        position.append([img.shape[2] - img_size, img.shape[3] - img_size])
        batch_count += 1

        if batch_count == batch:
            crop = np.array(crop).astype(np.float32)
            for num, (t, l) in enumerate(position):
                piece = crop[num]
                h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                result_img[t : t + img_size, l : l + img_size, :] += piece[:h, :w, :]
                voting_mask[t : t + img_size, l : l + img_size, :] += 1
            crop = []
            position = []
            batch_count = 0

        if batch_count > 0:
            crop = np.array(crop).astype(np.float32)
            for num, (t, l) in enumerate(position):
                piece = crop[num]
                h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                result_img[t : t + img_size, l : l + img_size, :] += piece[:h, :w, :]
                voting_mask[t : t + img_size, l : l + img_size, :] += 1

        result_img = result_img / voting_mask
        result_img = result_img.astype(np.uint8)

        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        results.append(result_img)

    return results


class InferenceDataset(Dataset):
    
    def __init__(self, img, pix2pix_transform, hinet_transform):
        super(InferenceDataset, self).__init__()
        pass

    def __getitem__(self, idx):
        return

    def __len__(self):
        return


if __name__ == '__main__':
    model = None
    transforms = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stride = 256
    max_batch = 10
    img_path = "/content/data/test_input_img/test_input_20003.png"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = img.copy()
    img = transforms(image=img)["image"]

    crop = []
    position = []

    result_img = np.zeros_like(img.numpy().transpose(1, 2, 0))
    voting_mask = np.zeros_like(img.numpy().transpose(1, 2, 0))

    img = img.unsqueeze(0)
    for top in range(0, img.shape[2], stride):
        for left in range(0, img.shape[3], stride):
            if top + img_size > img.shape[2] or left + img_size > img.shape[3]:
                continue

            piece = torch.zeros([1, 3, img_size, img_size])

            temp = img[:, :, top : top + img_size, left : left + img_size]
            piece[:, :, : temp.shape[2], : temp.shape[3]] = temp
            position.append([top, left])
            batch_count += 1

            if batch_count == batch:
                crop = np.array(crop).astype(np.float32)
                for num, (t, l) in enumerate(position):
                    piece = crop[num]
                    h, w, c = result_img[
                        t : t + img_size, l : l + img_size, :
                    ].shape
                    result_img[t : t + img_size, l : l + img_size, :] += piece[
                        :h, :w, :
                    ]
                    voting_mask[t : t + img_size, l : l + img_size, :] += 1
                crop = []
                position = []
                batch_count = 0

    # 가장 자리 1
    for left in range(0, img.shape[3], stride):
        if left + img_size > img.shape[3]:
            continue
        piece = torch.zeros([1, 3, img_size, img_size])
        temp = img[
            :, :, img.shape[2] - img_size : img.shape[2], left : left + img_size
        ]
        piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

        with torch.no_grad():
            pred = model(piece.to(device))

        pred = pred[0].cpu().clone().detach().numpy()
        pred = pred.transpose(1, 2, 0)
        pred = pred * 127.5 + 127.5

        crop.append(pred)
        position.append([img.shape[2] - img_size, left])

        batch_count += 1

        if batch_count == batch:
            crop = np.array(crop).astype(np.float32)
            for num, (t, l) in enumerate(position):
                piece = crop[num]
                h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                result_img[t : t + img_size, l : l + img_size, :] += piece[
                    :h, :w, :
                ]
                voting_mask[t : t + img_size, l : l + img_size, :] += 1
            crop = []
            position = []
            batch_count = 0

    # 가장 자리 2
    for top in range(0, img.shape[2], stride):
        if top + img_size > img.shape[2]:
            continue
        piece = torch.zeros([1, 3, img_size, img_size])

        temp = img[
            :, :, top : top + img_size, img.shape[3] - img_size : img.shape[3]
        ]
        piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

        with torch.no_grad():
            pred = model(piece.to(device))
        pred = pred[0].cpu().clone().detach().numpy()
        pred = pred.transpose(1, 2, 0)
        pred = (pred * 127.5) + 127.5

        crop.append(pred)
        position.append([top, img.shape[3] - img_size])
        batch_count += 1

        if batch_count == batch:
            crop = np.array(crop).astype(np.float32)
            for num, (t, l) in enumerate(position):
                piece = crop[num]
                h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
                result_img[t : t + img_size, l : l + img_size, :] += piece[
                    :h, :w, :
                ]
                voting_mask[t : t + img_size, l : l + img_size, :] += 1
            crop = []
            position = []
            batch_count = 0

    # 오른쪽 아래
    piece = torch.zeros([1, 3, img_size, img_size])
    temp = img[
        :,
        :,
        img.shape[2] - img_size : img.shape[2],
        img.shape[3] - img_size : img.shape[3],
    ]
    piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

    with torch.no_grad():
        pred = model(piece.to(device))

    pred = pred[0].cpu().clone().detach().numpy()
    pred = pred.transpose(1, 2, 0)
    pred = pred * 127.5 + 127.5

    crop.append(pred)
    position.append([img.shape[2] - img_size, img.shape[3] - img_size])
    batch_count += 1

    if batch_count == batch:
        crop = np.array(crop).astype(np.float32)
        for num, (t, l) in enumerate(position):
            piece = crop[num]
            h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
            result_img[t : t + img_size, l : l + img_size, :] += piece[:h, :w, :]
            voting_mask[t : t + img_size, l : l + img_size, :] += 1
        crop = []
        position = []
        batch_count = 0

    if batch_count > 0:
        crop = np.array(crop).astype(np.float32)
        for num, (t, l) in enumerate(position):
            piece = crop[num]
            h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
            result_img[t : t + img_size, l : l + img_size, :] += piece[:h, :w, :]
            voting_mask[t : t + img_size, l : l + img_size, :] += 1

    result_img = result_img / voting_mask
    result_img = result_img.astype(np.uint8)

    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    results.append(result_img)
