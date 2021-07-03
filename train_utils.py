import os
from tqdm import tqdm
import cv2
import torch
import numpy as np



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
