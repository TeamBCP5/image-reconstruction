import os
from glob import glob
from typing import List, Tuple
import pickle
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def train_valid_split(data_dir: str='/content/', meta_path: str='train_meta.csv', valid_type: int=1, full_train: bool=False) -> Tuple[List[str]]:
    """
    Example:
        train_input_paths, train_label_paths, valid_input_paths, valid_label_paths = \
            train_valid_split(data_dir='/content/', meta_path='train_meta.csv', valid_type=1, full_train=True)
    Args:
        data_dir (str): 학습 데이터 디렉토리 경로
        meta_path (str): 메타파일 경로. Defaults to 'train_meta.csv'.
        valid_type (int):
            - 1: 스프레드시트를 통해 결정한 validation set
            - 0: 기존 validation set(10000~10059 이미지로 검증)
        full_train (bool):
            - True: valid_type에 관계 없이 모든 데이터를 학습에 활용
    Returns:
        Tuple[List[str]]: [학습IMG경로], [학습GT경로], [검증IMG경로], [검증GT경로]
    """
    assert os.path.isfile(meta_path), f"'{meta_path}' not found"
    assert os.path.isdir(data_dir), f"'{data_dir}' is not a directory"
    meta = pd.read_csv(meta_path)

    # align data path
    meta['input_img'] = meta['input_img'].apply(lambda x: os.path.join(data_dir, 'train_input_img', x))
    meta['label_img'] = meta['label_img'].apply(lambda x: os.path.join(data_dir, 'train_label_img', x))

    # split train & valid
    if full_train:
        train_input_paths = meta['input_img'].tolist()
        train_label_paths = meta['label_img'].tolist()
    else:
        train_input_paths = meta[meta[f'valid_type{valid_type}']=='train']['input_img'].tolist()
        train_label_paths = meta[meta[f'valid_type{valid_type}']=='train']['label_img'].tolist()
    valid_input_paths = meta[meta[f'valid_type{valid_type}']=='valid']['input_img'].tolist()
    valid_label_paths = meta[meta[f'valid_type{valid_type}']=='valid']['label_img'].tolist()
    return train_input_paths, train_label_paths, valid_input_paths, valid_label_paths


class ShiftedDataset(Dataset):
    def __init__(self, source_dir, label_dir, transforms=None):
        self.sources = sorted(glob(os.path.join(source_dir, '*.png')))
        self.labels = sorted(glob(os.path.join(label_dir, '*.png')))
        assert len(self.sources) == len(self.labels)
        self.transforms = transforms

    def __getitem__(self, idx, channel_order='rgb'):
        src = cv2.imread(self.sources[idx])
        lbl = cv2.imread(self.labels[idx])
        assert src.shape == lbl.shape
        if channel_order == 'rgb':
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
        outputs = self.transforms(image=src, label=lbl)
        return outputs

    def __len__(self):
        return len(self.sources)

class EvalShiftedDataset(Dataset):
    def __init__(self, source_dir, transforms):
        self.sources = sorted(glob(os.path.join(source_dir, '*.png')))
        self.transforms = transforms

    def __getitem__(self, idx, channel_order='rgb'):
        img_name = os.path.basename(self.sources[idx])
        prefix =  img_name.split('_')[0] # 'test' or 'train'
        suffix = img_name.split('_')[-1].split('.')[0] # 10001(0,0), ...
        src = cv2.imread(self.sources[idx])
        if channel_order == 'rgb':
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        outputs = self.transforms(img_id=f'{prefix}_{suffix}.png', image=src)
        return outputs

    def __len__(self):
        return len(self.sources)


class CustomDataset(Dataset):
    def __init__(self, data_dir, train_mode, img_size, transforms):
        super().__init__()
        self.train_mode = train_mode
        self.transforms = transforms

        if self.train_mode=='Train':
            self.img_list = glob(os.path.join(data_dir, f'train_input_img_{img_size}/*.pickle'))
            self.label_list = glob(os.path.join(data_dir, f'train_label_img_{img_size}/*.pickle'))
            
            self.img_list.sort()
            self.label_list.sort()
        elif self.train_mode=='Val':
            self.img_list = glob(os.path.join(data_dir, f'val_input_img_{img_size}/*.pickle'))
            self.label_list = glob(os.path.join(data_dir, f'val_label_img_{img_size}/*.pickle'))
            
            self.img_list.sort()
            self.label_list.sort()
        else:
            self.img_list = glob(os.path.join(data_dir, 'test_input_img/*.*'))
    
    def load_pickle(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        return data
  
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_file_name = img_path.split('/')[-1]
        image = self.load_pickle(img_path)
        image = image.astype(np.float32)
        
        if self.train_mode != 'Test':
            label_path = self.label_list[index]
            label_file_name = label_path.split('/')[-1]
            label = self.load_pickle(label_path)
            label = label.astype(np.float32)
            transformed = self.transforms(image=image, label=label)
            image = transformed['image']
            label = transformed['label']
            return image, label
        else:
            transformed = self.transforms(image=image)
            image = transformed["image"]
            return image, img_file_name
    
    def __len__(self):
        return len(self.img_list)


class BaseDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=None):
        self.input_paths = sorted(glob(os.path.join(input_dir, "*.png")))
        self.target_paths = sorted(glob(os.path.join(target_dir, "*.png")))
        assert len(self.input_paths) == len(self.target_paths)

    def __getitem__(self, idx):
        # input_img = cv2.imread(self.input_paths[idx])
        target_img = cv2.imread(self.target_paths[idx])
        input_img = Image.open(self.input_paths[idx])
        target_img = Image.open(self.target_paths[idx])
        input_img, target_img = self.augmentation(input_img, target_img)
        return dict(input=input_img, target=target_img)

    def __len__(self):
        return len(self.input_paths)

    def augmentation(self, inp, targ):
        inp, targ = self.random_rot(inp, targ)
        inp, targ = self.random_flip(inp, targ)
        inp, targ = self.to_tensor(inp, targ)
        return inp, targ

    @staticmethod
    def random_rot(inp, targ):
        k = np.random.randint(4)
        inp = np.rot90(inp, k)
        targ = np.rot90(targ, k)
        return inp, targ

    @staticmethod
    def random_flip(inp, targ):
        f = np.random.randint(2)
        if f == 0:
            inp = np.fliplr(inp)
            targ = np.fliplr(targ)

        return inp, targ

    @staticmethod
    def to_tensor(input_img, target_img):
        input_img = transforms.ToTensor()(input_img)
        target_img = transforms.ToTensor()(target_img)
        return input_img, target_img

    @staticmethod
    def collate_fn(batch):
        input_imgs = [b["input"] for b in batch]
        target_imgs = [b["target"] for b in batch]

        input_imgs = torch.vstack(input_imgs)
        target_imgs = torch.vstack(target_imgs)
        return dict(input=input_imgs, target=target_imgs)
