import argparse
import os
from glob import glob
from tqdm import tqdm
import cv2
import albumentations as A
from data import cut_img
from utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./camera_dataset/", help="이미지 데이터 경로"
    )
    parser.add_argument("--num_samples", type=int, default=5, help="생성할 샘플 수")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./sample_preprocessing/",
        help="Preprocessing 결과를 저장할 디렉토리 경로",
    )
    parser.add_argument(
        "--stride", type=int, default=256, help="Sliding Window 시 사용할 stride"
    )
    parser.add_argument(
        "--patch_size", type=int, default=512, help="Sliding Window 시 사용할 patch 사이즈"
    )
    args = parser.parse_args()

    set_seed(41)
    os.makedirs(args.save_dir, exist_ok=True)  # 원본 이미지가 저장될 디렉토리 경로
    original_dir = os.path.join(args.save_dir, "original")  # 원본 이미지가 저장될 디렉토리 경로
    pix2pix_preprocessing_dir = os.path.join(
        args.save_dir, "pix2pix"
    )  # Pix2Pix 전처리(sliding window) 결과가 저장될 디렉토리 경로
    hinet_preprocessing_dir = os.path.join(
        args.save_dir, "hinet"
    )  # HINet 전처리(resize) 결과가 저장될 디렉토리 경로
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(pix2pix_preprocessing_dir, exist_ok=True)
    os.makedirs(hinet_preprocessing_dir, exist_ok=True)
    os.makedirs(os.path.join(original_dir, "train_input_img"), exist_ok=True)
    os.makedirs(os.path.join(original_dir, "train_label_img"), exist_ok=True)
    os.makedirs(os.path.join(hinet_preprocessing_dir, "train_input_img"), exist_ok=True)
    os.makedirs(os.path.join(hinet_preprocessing_dir, "train_label_img"), exist_ok=True)

    train_input_paths = sorted(
        glob(os.path.join(args.data_dir, "train_input_img", "*"))
    )[: args.num_samples]
    train_label_paths = sorted(
        glob(os.path.join(args.data_dir, "train_label_img", "*"))
    )[: args.num_samples]

    # Pix2Pix
    cut_img(
        train_input_paths,
        train_label_paths,
        save_dir=pix2pix_preprocessing_dir,
        stride=args.stride,
        patch_size=args.patch_size,
    )

    # HINet
    for input_path, label_path in tqdm(
        zip(train_input_paths, train_label_paths), "[Resize]"
    ):
        input_img = cv2.imread(input_path)
        input_name = os.path.basename(input_path).split(".png")[0]
        label_img = cv2.imread(label_path)
        label_name = os.path.basename(label_path).split(".png")[0]

        cv2.imwrite(
            os.path.join(original_dir, "train_input_img", f"{input_name}.png"),
            input_img,
        )
        cv2.imwrite(
            os.path.join(original_dir, "train_label_img", f"{label_name}.png"),
            label_img,
        )

        input_img = A.Resize(1224, 1632, p=1.0)(image=input_img)["image"]
        label_img = A.Resize(1224, 1632, p=1.0)(image=label_img)["image"]

        cv2.imwrite(
            os.path.join(
                hinet_preprocessing_dir, "train_input_img", f"{input_name}_resized.png"
            ),
            input_img,
        )
        cv2.imwrite(
            os.path.join(
                hinet_preprocessing_dir, "train_label_img", f"{label_name}_resized.png"
            ),
            label_img,
        )

    print(f"Preprocessing samples saved in '{args.save_dir}'.")
