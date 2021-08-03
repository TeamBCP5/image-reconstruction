import os
from typing import Tuple, List
import pandas as pd
import cv2
from torch.utils.data import Dataset
import albumentations as A

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


def train_valid_unseen_split(
    data_dir: str = "/content/",
    meta_path: str = "./configs/train_meta.csv",
    valid_type: int = 11,
    full_train: bool = True,
) -> Tuple[List[str]]:
    assert os.path.isfile(meta_path), f"'{meta_path}' not found"
    assert os.path.isdir(data_dir), f"'{data_dir}' is not a directory"
    print(
        "[+] Train/Valid Type Desctription\n",
        f"Type: {valid_type}\n",
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

    # split train & valid
    if full_train:
        train_input_paths = meta[
            (meta[f"valid_type{valid_type}"] != "except")
            & (meta[f"valid_type{valid_type}"] != "unseen")
        ]["input_img"].tolist()
        train_label_paths = meta[
            (meta[f"valid_type{valid_type}"] != "except")
            & (meta[f"valid_type{valid_type}"] != "unseen")
        ]["label_img"].tolist()
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

    unseen_input_paths = meta[meta[f"valid_type{valid_type}"] == "unseen"][
        "input_img"
    ].tolist()
    unseen_label_paths = meta[meta[f"valid_type{valid_type}"] == "unseen"][
        "label_img"
    ].tolist()
    return (
        train_input_paths,
        train_label_paths,
        valid_input_paths,
        valid_label_paths,
        unseen_input_paths,
        unseen_label_paths,
    )


class Pix2PixDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        train_mode: str = "Train",
        split_size: int = 512,
        transforms=None,
    ):
        super().__init__()
        self.train_mode = train_mode
        self.transforms = transforms

        if self.train_mode == "Train":
            self.img_list = sorted(
                glob(os.path.join(data_dir, f"train_input_img_{split_size}", "*.png"))
            )
            self.label_list = sorted(
                glob(os.path.join(data_dir, f"train_label_img_{split_size}", "*.png"))
            )

        elif self.train_mode == "Val":
            self.img_list = sorted(
                glob(os.path.join(data_dir, f"val_input_img_{split_size}", "*.png"))
            )
            self.label_list = sorted(
                glob(os.path.join(data_dir, f"val_label_img_{split_size}", "*.png"))
            )

        else:
            self.img_list = sorted(
                glob(os.path.join(data_dir, "test_input_img", "*.png"))
            )

    def __getitem__(self, index):
        img_name = os.path.basename(self.img_list[index])
        image = cv2.imread(self.img_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train_mode != "Test":
            label = cv2.imread(self.label_list[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            transformed = self.transforms(image=image, label=label)
            image = transformed["image"]
            label = transformed["label"]
            return image, label

        # for test(inference)
        else:
            transformed = self.transforms(image=image)
            image = transformed["image"]
            return image, img_name

    def __len__(self):
        return len(self.img_list)


class HINetDataset(Dataset):
    def __init__(self, input_paths, label_paths=None, transforms=None, mode="Train"):
        super().__init__()
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index):
        image = cv2.imread(self.input_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "Train":
            image = A.GaussNoise(var_limit=5)(image=image)["image"]
            label = cv2.imread(self.label_paths[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

            transformed = self.transforms(image=image, label=label)
            image = transformed["image"] / 255.0
            label = transformed["label"] / 255.0
            return image, label

        elif self.mode == "Valid":
            image = A.GaussNoise(var_limit=5)(image=image)["image"]
            image = self.transforms(image=image)["image"]
            image = image / 255.0

            label = cv2.imread(self.label_paths[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            return image, label

        elif self.mode == "Test":
            img_id = os.path.basename(self.input_paths[index])
            image = self.transforms(image=image)["image"]
            image = image / 255.0
            return img_id, image

    def __len__(self):
        return len(self.input_paths)
