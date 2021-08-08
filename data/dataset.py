import os
from tqdm import tqdm
import shutil
from typing import Tuple, List
from glob import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import rasterio
from rasterio.windows import Window
from utils import get_model, truncate_aligned_model, Flags
from data import get_valid_transform


VALIDTYPE_DESC = {
    "valid_type1": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n",
    "valid_type2": "Valid:\n  - Train10000-10059\n Train:\n  - Valid 이미지 제외\n",
    "valid_type3": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n",
    "valid_type4": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - Test 이미지와 상이한 이미지 제외\n",
    "valid_type5": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - Test 이미지와 상이한 이미지 제외\n  - Valid와 유사한 이미지 제외\n",
    "valid_type6": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - Test 이미지와 상이한 이미지 제외\n  - Valid와 유사한 이미지 제외\n  - 1632x1224 제외\n",
    "valid_type7": "Valid:\n  - 녹색빛 & 빛이 강한 이미지\n Train:\n  - Valid 이미지 제외\n  - 녹색빛 & 빛이 강한 이미지\n",
    "valid_type8": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - 빠른 실험을 위한 소규모 학습 데이터\n",
    "valid_type9": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - 빠른 실험을 위한 소규모 학습 데이터\n  - Unseen 데이터로 일반화 성능 검증\n",
    "valid_type10": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - 20개 Unseen 데이터로 일반화 성능 검증\n",
    "valid_type11": "Valid:\n  - Test 이미지와 유사한 20장\n Train:\n  - Valid 이미지 제외\n  - 노이즈 이미지 제외\n  - 6개 Unseen 데이터로 일반화 성능 검증\n",
}


def train_valid_split(
    data_dir: str = "/content/data",
    meta_path: str = "./configs/train_meta.csv",
    valid_type: int = 1,
    full_train: bool = True,
) -> Tuple[List[str]]:
    assert os.path.isfile(meta_path), f"'{meta_path}' not found"
    assert os.path.isdir(data_dir), f"'{data_dir}' is not a directory"
    print(
        "[+] Train/Valid Type Desctription\n",
        f"Type: {valid_type}\n",
        f"Full Train: {full_train}\n",
        VALIDTYPE_DESC[f"valid_type{valid_type}"],
    )
    meta = pd.read_csv(meta_path)

    # align data path
    meta["input_img"] = meta["input_img"].apply(
        lambda x: os.path.join(data_dir, "train_input_img", x)
    )
    meta["label_img"] = meta["label_img"].apply(
        lambda x: os.path.join(data_dir, "train_label_img", x)
    )

    # filter image which is exist
    meta = meta.loc[meta["input_img"].apply(lambda x: os.path.isfile(x))].reset_index(
        drop=True
    )

    # split train & valid
    if full_train:
        train_input_paths = meta[meta[f"valid_type{valid_type}"] != "except"][
            "input_img"
        ].tolist()
        train_label_paths = meta[meta[f"valid_type{valid_type}"] != "except"][
            "label_img"
        ].tolist()
    else:
        train_input_paths = meta[meta[f"valid_type{valid_type}"] == "train"][
            "input_img"
        ].tolist()
        train_label_paths = meta[meta[f"valid_type{valid_type}"] == "train"][
            "label_img"
        ].tolist()

    valid_input_paths = meta[meta[f"valid_type{valid_type}"] == "valid"][
        "input_img"
    ].tolist()
    valid_label_paths = meta[meta[f"valid_type{valid_type}"] == "valid"][
        "label_img"
    ].tolist()
    return train_input_paths, train_label_paths, valid_input_paths, valid_label_paths


class Pix2PixDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 512,
        transforms=None,
        mode: str = "train",  # 'train', 'valid', 'test'
    ):
        super().__init__()
        self.mode = mode
        self.transforms = transforms

        self.src_paths = sorted(
            glob(os.path.join(data_dir, f"{mode}_input_img_{patch_size}", "*"))
        )
        self.lbl_paths = (
            sorted(glob(os.path.join(data_dir, f"{mode}_label_img_{patch_size}", "*")))
            if mode != "test"
            else None
        )
        assert len(self.src_paths) == len(self.lbl_paths)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.src_paths[idx])
        src = cv2.imread(self.src_paths[idx])
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        if self.lbl_paths is not None:
            lbl = cv2.imread(self.lbl_paths[idx])
            lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
            transformed = self.transforms(image=src, label=lbl)
            src = transformed["image"]
            lbl = transformed["label"]
            return src, lbl

        else:
            src = self.transforms(image=src)["image"]
            return src, img_name

    def __len__(self):
        return len(self.src_paths)


class HINetDataset(Dataset):
    def __init__(self, input_paths, label_paths=None, transforms=None, mode="train"):
        super().__init__()
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index):
        image = cv2.imread(self.input_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            image = A.GaussNoise(var_limit=5)(image=image)["image"]
            label = cv2.imread(self.label_paths[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

            transformed = self.transforms(image=image, label=label)
            image = transformed["image"] / 255.0
            label = transformed["label"] / 255.0
            return image, label

        elif self.mode == "valid":
            
            image = self.transforms(image=image)["image"]
            image = image / 255.0

            label = cv2.imread(self.label_paths[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            return image, label

        elif self.mode == "test":
            img_id = os.path.basename(self.input_paths[index])
            image = self.transforms(image=image)["image"]
            image = image / 255.0
            return img_id, image

    def __len__(self):
        return len(self.input_paths)


class CutImageDataset(Dataset):
    """
    input 이미지를 주어진 patch size로 자른 뒤, 좌표와 함께 반환
    Process:
        (1) 이미지로부터 patch size와 stride에 따라 x1, x2, y1, y2 좌표를 계산
        (2) x1, x2, y1, y2에 따라 이미지를 자른 뒤, 좌표와 함께 반환

    Args:
        img_path (str):
            - 자르고자 하는 이미지의 경로
        label_path(str, optional):
            - 자르고자 하는 라벨의 경로
            - 추론 시, 계산하지 않음
            - Defaults to None 
        patch_size (int, optional):
            - 반환받고자 하는 이미지의 크기
            - Defaults to 512.
        stride(int, optional):
            - window가 움직이는 길이, 짧을 수록 많은 양의 패치 생성
            - Defaults to 256
        transforms(optional):
            - 적용하고자 하는 Augmentation
            - Defaults to None
    Return:
        (np.array), (tuple): 패치 크기의 이미지, 원래 이미지에서의 좌표
    """
    def __init__(
        self,
        img_path: str,
        label_path: str = None,
        patch_size: int = 512,
        stride: int = 256,
        transforms=None,
    ):
        self.image = rasterio.open(img_path, num_threads="all_cpus")
        self.label = (
            rasterio.open(label_path, num_threads="all_cpus")
            if label_path is not None
            else None
        )
        self.shape = self.image.shape
        self.slices = self.make_grid(self.shape, patch_size, stride)
        self.transforms = transforms

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        x1, x2, y1, y2 = self.slices[index]
        image = self.image.read(
            [1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2))
        )
        image = np.moveaxis(image, 0, -1)

        # pix2pix에 사용하기 위한 학습데이터 생성
        if self.label is not None:
            label = self.label.read(
                [1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2))
            )
            label = np.moveaxis(label, 0, -1)
            return image, label

        # pix2pix에 사용될 추론 데이터 생성
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return image, (x1, x2, y1, y2)

    @staticmethod
    def make_grid(shape, patch_size=512, stride=256):
        x, y = shape
        # shape에 따른 패치 개수 계산
        nx = x // stride
        ny = y // stride
        slices = []
        x1 = 0
        for _ in range(nx):
            x2 = min(x1 + patch_size, x)
            y1 = 0
            for _ in range(ny):
                y2 = min(y1 + patch_size, y)
                # 패치의 끝 점이 shape의 크기를 넘어갈 시 패치의 시작점 조절 
                if x2 - x1 != patch_size:
                    x1 = x2 - patch_size
                if y2 - y1 != patch_size:
                    y1 = y2 - patch_size
                slices.append([x1, x2, y1, y2])
                y1 += stride
            x1 += stride
        slices = np.array(slices)
        return slices.reshape(-1, 4)

def compose_postprocessing_dataset(args, device):
    input_save_dir = os.path.join(args.data.dir, "train_input_img")
    label_save_dir = os.path.join(args.data.dir, "train_label_img")
    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    if len(os.listdir(input_save_dir)) != 0 and len(os.listdir(input_save_dir)) == len(
        os.listdir(label_save_dir)
    ):
        if len(os.listdir(input_save_dir)) < 622:
            import warnings
            warnings.warn(f"""
            The size of dataset is lower than 622, the original number of given train data.
            It's because you might stop process composing postprocessing dataset process.
            Train will be progressed without error, but if you want to train postprocessor with full dataset,
            remove all data in '{input_save_dir}' and '{label_save_dir}'
            """)
        return

    print(
        f"[+] Compose postprocessing dataset\n",
        "There's no train dataset to for postprocessor(HINet).\n",
        "Start composing dataset using main model(Pix2Pix).\n",
        f"Input images will be saved in '{input_save_dir}'.\n",
        f"Label images will be saved in '{label_save_dir}'.\n",
    )
    src_args = Flags(args.data.source.config).get()
    stride = src_args.data.stride
    patch_size = src_args.data.patch_size
    batch_size = 32

    G_model = get_model(src_args, mode="test")
    try:
        ckpt = torch.load(args.data.source.checkpoint)["G_model"]
    except:
        ckpt = torch.load(args.data.source.checkpoint)
    G_model.load_state_dict(ckpt)
    G_model.to(device)
    G_model.eval()
    print(f"[+] Generation model\n", f"Loaded from '{args.data.source.checkpoint}'\n")

    train_input_paths = sorted(
        glob(os.path.join(src_args.data.dir, "train_input_img", "*"))
    )
    train_label_paths = sorted(
        glob(os.path.join(src_args.data.dir, "train_label_img", "*"))
    )

    transforms = get_valid_transform(src_args.network.name)
    with torch.no_grad():
        for img_path, lbl_path in tqdm(
            zip(train_input_paths, train_label_paths), desc="[Compose Dataset]"
        ):
            ds = CutImageDataset(img_path, patch_size=patch_size, stride=stride, transforms=transforms)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
            # main light scattering reduction(pix2pix)
            preds = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            votes = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            for images, (x1, x2, y1, y2) in dl:
                pred = G_model(images.to(device).float())
                pred = pred * 127.5 + 127.5
                for i in range(len(x1)):
                    preds[:, x1[i] : x2[i], y1[i] : y2[i]] += pred[i]
                    votes[:, x1[i] : x2[i], y1[i] : y2[i]] += 1
            preds /= votes
            preds = preds.cpu().detach().numpy().astype(np.uint8)
            preds = preds.transpose(1, 2, 0)
            preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)

            # save input images
            img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(input_save_dir, img_name), preds)

            # save label images
            lbl_name = os.path.basename(lbl_path)
            shutil.copy(lbl_path, os.path.join(label_save_dir, lbl_name))

    truncate_aligned_model(G_model)