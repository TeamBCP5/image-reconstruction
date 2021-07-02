import os
from tqdm import tqdm
import cv2
import torch
import numpy as np
from utils import save_pickle

def cut_img(img_path_list, save_path, img_size, stride):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        # img = reflection_pad(img, window_size=img_size)
        # assert img.shape[0] % img_size == 0
        # assert img.shape[1] % img_size == 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                pkl_save_path = f'{save_path}{img_size}/{num}.pickle'
                save_pickle(pkl_save_path, piece)
                num += 1

def reflection_pad(image, window_size: int):
    h, w, _ = image.shape
    residue_w = w % window_size
    residue_h = h % window_size

    pad_w, pad_h = (window_size-residue_w)//2, (window_size-residue_h)//2
    left_pad = image[:, ::-1, :][:, w - pad_w:, :]
    right_pad = image[:, ::-1, :][:, :pad_w, :]
    image = np.hstack([left_pad, image, right_pad])

    top_pad = image[::-1, :, :][h-pad_h:, :, :]
    bottom_pad = image[::-1, :, :][:pad_h, :, :]
    image = np.vstack([top_pad, image, bottom_pad])
    return image


class EarlyStopping:
    def __init__(
        self, patience: int = 5, verbose: bool = False, delta: int = 0, path=None
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        model.eval()
        torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
