import math
import numpy as np


def rmse_score(true, pred):
    score = math.sqrt(np.mean((true - pred) ** 2))
    return score

def psnr_score(true, pred, pixel_max):
    score = 20 * np.log10(pixel_max / rmse_score(true, pred))
    return score
