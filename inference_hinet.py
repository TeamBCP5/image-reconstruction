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
from networks.HINet import HINet
from utils.criterions import HINetLoss
from utils.utils import get_model
from utils.metrics import psnr_score
from data.dataset import HINetDataset, train_valid_split
from data.augmentations import get_train_transform, get_valid_transform


def inference(model, epoch, test_loader, device, save_dir: str='./inference/'):
    model.eval()

    save_paths = []
    with torch.no_grad():
        for img_id, img in tqdm(test_loader, desc="[Inference]"):
            img_id = img_id[0]
            pred = model(img.float().to(device))[0]
            pred = torch.clamp(pred[-1], 0, 1)
            pred = pred.cpu().clone().detach().numpy()
            pred = pred * 255.0
            pred = pred.transpose(1, 2, 0)
            pred = np.clip(pred, 0, 255)
            pred = pred.astype(np.uint8)
            pred = cv2.resize(pred, dsize=(3264, 2448), interpolation=cv2.INTER_CUBIC)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            img_save_path = os.path.join(save_dir, img_id)
            cv2.imwrite(img_save_path, pred)
            save_paths.append(img_save_path)

    save_zip_path = os.path.join(save_dir, f'inference_ep({epoch:0>2d}).zip')
    submission = zipfile.ZipFile(save_zip_path, 'w')
    for path in save_paths:
        submission.write(path)
    submission.close()