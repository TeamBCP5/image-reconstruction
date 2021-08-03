import os
import cv2
import zipfile
from glob import glob
import pickle
import random
import numpy as np
import segmentation_models_pytorch as smp

# from unext import UneXt50
import torch
from torch import nn
from networks.Pix2Pix import define_D
from networks.HINet import HINet


def get_model(model_type: str, encoder_weights: str="imagenet", activation: str="tanh", mode="train") -> nn.Module:
    if model_type == "pix2pix":
        G_model = smp.Unet(
            encoder_name="se_resnext50_32x4d",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=3,
        )
        # define activation function
        if activation == "tanh":
            G_model.segmentation_head[2] = nn.Tanh()

        elif activation == "sigmoid":
            G_model.segmentation_head[2] = nn.Sigmoid()
        if mode=="train":
            D_model = define_D(ndf=64, n_layers_D=4)
            return G_model, D_model
        elif mode=="inference":
            return G_model

    elif model_type == "hinet":
        model = HINet(depth=4)
        return model

    else:
        raise NotImplementedError(f"There's no model_type '{model_type}'")


def remove_all_files_in_dir(dir):
    """dir 내 모든 파일을 제거"""
    for fpath in glob(os.path.join(dir, "*")):
        os.remove(fpath)


def save_samples(result, epoch, save_dir="inference_sample"):
    sub_imgs = []
    for i, img in enumerate(result):
        save_path = os.path.join(save_dir, f"test_{20000+i}.png")
        cv2.imwrite(save_path, img)
        sub_imgs.append(save_path)

    save_zip_path = os.path.join(save_dir, f"sample_epoch_{epoch}.zip")
    submission = zipfile.ZipFile(save_zip_path, "w")
    for path in sub_imgs:
        submission.write(path)
    submission.close()


def seed_everything(seed: int = 41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_gpu_status() -> None:
    """GPU 이용 상태를 출력"""
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 3)
    reserved = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 3)
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 3)
    free = round(reserved - allocated, 3)
    print(
        "[+] GPU Status\n",
        f"Total: {total_mem} GB\n",
        f"Reserved: {reserved} GB\n",
        f"Allocated: {allocated} GB\n",
        f"Residue: {free} GB\n",
    )


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_grid(shape, window=512, stride=256):
    x, y = shape
    nx = x // stride + 1
    ny = y // stride + 1
    slices = []
    x1 = 0
    for i in range(nx):
        x2 = min(x1 + window, x)
        y1 = 0
        for j in range(ny):
            y2 = min(y1 + window, y)
            if x2 - x1 != window:
                x1 = x2 - window
            if y2 - y1 != window:
                y1 = y2 - window
            slices.append([x1, x2, y1, y2])
            y1 += stride
        x1 += stride
    slices = np.array(slices)  
    return slices.reshape(-1,4)
