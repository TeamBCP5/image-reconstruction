import math
import cv2
import numpy as np


def rmse_score(true, pred):
    if true.ndim == 4:  # 배치단위
        score = np.sqrt(np.mean((true - pred) ** 2, axis=(1, 2, 3)))
    else:
        score = np.sqrt(np.mean((true - pred) ** 2))
    return score


def psnr_score(true, pred, pixel_max: int = 255):
    score = 20 * np.log10(pixel_max / rmse_score(true, pred))
    return score