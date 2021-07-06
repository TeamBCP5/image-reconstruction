import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from typing import List
from utils import save_pickle
 
LARGE = 3264
SMALL = 1632

def get_save_shifted_images(img_path_list: List[str], save_dir: str, shuffle: bool=False, random_state: int=None):
    os.makedirs(save_dir, exist_ok=True)
    if random_state is not None:
        random.seed(random_state)

    for img_path in tqdm(img_path_list, desc='[Extract Shifted Imgs]'):
        img_name = os.path.basename(img_path)
        prefix = img_name.split('.')[0]

        img = cv2.imread(img_path)
        img = make_img_square(img)

        shift_factor = 6 if img.shape[0] == LARGE else 3

        if shuffle:
            shifts = [[(i, j) for i in range(shift_factor) for j in range(shift_factor)] for _ in range(3)] # for each channel
            for s in shifts:
                random.shuffle(s)

            for (w0, h0), (w1, h1), (w2, h2) in zip(*shifts):
                channel0 = img[:, w0::shift_factor, 0]
                channel0 = channel0[h0::shift_factor, :]
                channel0 = channel0[:, :, np.newaxis]

                channel1 = img[:, w1::shift_factor, 1]
                channel1 = channel1[h1::shift_factor, :]
                channel1 = channel1[:, :, np.newaxis]

                channel2 = img[:, w2::shift_factor, 2]
                channel2 = channel2[h2::shift_factor, :]
                channel2 = channel2[:, :, np.newaxis]

                shifted_img = np.concatenate([channel0, channel1, channel2], axis=-1)
                save_path = os.path.join(save_dir, f'{prefix}({w0}{w1}{w2},{h0}{h1}{h2}).png')
                cv2.imwrite(save_path, shifted_img)

        else:
            shifts = [(i, j) for i in range(shift_factor) for j in range(shift_factor)]
            for w, h in shifts:
                shifted_img = shifted_img[:, w::shift_factor, :]
                shifted_img = shifted_img[h::shift_factor, :, :]
                save_path = os.path.join(save_dir, f'{prefix}({w},{h}).png')
                cv2.imwrite(save_path, shifted_img)


def make_img_square(img: np.array): # reflection pad
    h, w, _ = img.shape

    # padding to make image square
    if h < w and (h - w) % 2 == 0:
        margin = (w - h) // 2
        lower_pad = img[::-1, :, :][:margin]
        upper_pad = img[::-1, :, :][h-margin:]
        img = np.vstack([upper_pad, img, lower_pad])
        
    return img

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