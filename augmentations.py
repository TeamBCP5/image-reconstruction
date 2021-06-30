import os
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        additional_targets={"image": "image", "label": "image"},
    )


def get_valid_transform():
    return A.Compose(
        [
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        additional_targets={"image": "image", "label": "image"},
    )
