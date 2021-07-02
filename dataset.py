import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle


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
        
        if self.train_mode!='Test':
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
