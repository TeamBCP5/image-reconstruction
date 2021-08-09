import os
from glob import glob
from tqdm import tqdm
import argparse
import cv2
import albumentations as A
from utils import set_seed


def demo_augmentation_pix2pix(probs: list) -> A.Compose:
    transform = A.Compose(
        [
            # B 채널 노이즈 삽입
            A.Cutout(
                num_holes=8,
                max_h_size=6,
                max_w_size=6,
                fill_value=[255, 0, 0],
                always_apply=False,
                p=probs[0],
            ),
            # G 채널 노이즈 삽입
            A.Cutout(
                num_holes=8,
                max_h_size=6,
                max_w_size=6,
                fill_value=[0, 255, 0],
                always_apply=False,
                p=probs[1],
            ),
            # R 채널 노이즈 삽입
            A.Cutout(
                num_holes=8,
                max_h_size=6,
                max_w_size=6,
                fill_value=[0, 0, 255],
                always_apply=False,
                p=probs[2],
            ),
            A.HorizontalFlip(p=probs[3]),  # Y축 대칭
            A.VerticalFlip(p=probs[4]),  # X축 대칭
            A.RandomRotate90(p=probs[5]),  # 90도 회전
        ],
        additional_targets={"image": "image", "label": "image"},
        p=1.0,
    )
    return transform


def demo_augmentation_hinet(probs: list):
    transform = A.Compose(
        [
            A.Resize(1224, 1632, p=1.0),
            A.HorizontalFlip(p=probs[0]),  # Y축 대칭
            A.ChannelShuffle(p=probs[1]),  # X축 대칭
        ],
        additional_targets={"label": "image"},
        p=1.0,
    )
    return transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./camera_dataset/", help="이미지 데이터 경로"
    )
    parser.add_argument("--num_samples", type=int, default=10, help="생성할 샘플 수")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./sample_augmentation/",
        help="Augmentation 적용 결과를 저장할 디렉토리 경로",
    )
    args = parser.parse_args()

    set_seed(41)
    os.makedirs(args.save_dir, exist_ok=True)
    original_dir = os.path.join(args.save_dir, "original")  # 원본 이미지가 저장될 디렉토리 경로
    pix2pix_aug_dir = os.path.join(
        args.save_dir, "pix2pix"
    )  # Pix2Pix augmentation 이미지가 저장될 디렉토리 경로
    hinet_aug_dir = os.path.join(
        args.save_dir, "hinet"
    )  # HINet augmentation 이미지가 저장될 디렉토리 경로
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(pix2pix_aug_dir, exist_ok=True)
    os.makedirs(hinet_aug_dir, exist_ok=True)

    train_input_paths = sorted(
        glob(os.path.join(args.data_dir, "train_input_img", "*"))
    )[: args.num_samples]

    # transform 할당
    aug_pix2pix = demo_augmentation_pix2pix("pix2pix")
    aug_hinet = demo_augmentation_hinet("hinet")

    for input_path in tqdm(train_input_paths, desc="[Sample Augmentation]"):
        img_name = os.path.basename(input_path).split(".png")[0]
        image = cv2.imread(input_path)
        cv2.imwrite(os.path.join(original_dir, f"{img_name}.png"), image)

        for i in range(6):
            pix2pix_aug = demo_augmentation_pix2pix(
                probs=[1.0 if k == i else 0.0 for k in range(6)]
            )
            aug_output = pix2pix_aug(image=image)["image"]
            cv2.imwrite(
                os.path.join(pix2pix_aug_dir, f"{img_name}_{i}.png"), aug_output
            )

        for i in range(2):
            hinet_aug = demo_augmentation_hinet(
                probs=[1.0 if k == i else 0.0 for k in range(2)]
            )
            aug_output = hinet_aug(image=image)["image"]
            cv2.imwrite(os.path.join(hinet_aug_dir, f"{img_name}_{i}.png"), aug_output)

    print(f"Augmentation samples saved in '{args.save_dir}'.")
