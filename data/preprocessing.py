import os
from tqdm import tqdm
from copy import deepcopy
from typing import List
import numpy as np
import cv2
from data import CutImageDataset


def cut_img(
    img_path_list: List[str],
    label_path_list: List[str],
    save_dir: str,
    stride: int = 256,
    patch_size: int = 512,
    denoise: str = False,
) -> None:
    # save directory 생성
    input_save_dir = os.path.join(save_dir, f"train_input_img_{patch_size}")
    label_save_dir = os.path.join(save_dir, f"train_label_img_{patch_size}")
    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    pbar = tqdm(
        zip(img_path_list, label_path_list),
        total=len(img_path_list),
        desc="[Cut & Save Images]",
    )
    for img_path, label_path in pbar:
        # 이미지를 슬라이딩 윈도우로 자르기 위한 Dataset 생성
        ds = CutImageDataset(img_path, label_path, patch_size=patch_size, stride=stride)

        # file명 추출
        img_name = os.path.basename(img_path).split(".png")[0]
        label_name = os.path.basename(label_path).split(".png")[0]

        for idx in range(len(ds)):
            image, label = ds[idx]  # Dataset으로부터 잘려진 이미지 load

            if denoise:
                image = apply_denoise(image)
                label = apply_denoise(label)

            # 이미지 저장
            image_save_path = os.path.join(input_save_dir, f"{img_name}_{idx:0>3d}.png")
            label_save_path = os.path.join(
                label_save_dir, f"{label_name}_{idx:0>3d}.png"
            )
            cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(label_save_path, cv2.cvtColor(label, cv2.COLOR_RGB2BGR))


def apply_naive_denoise(src: np.array) -> np.array:
    input_per_channel = [deepcopy(src[:, :, i]) for i in range(3)]
    blur_per_channel = []
    for idx in range(3):
        filter_size = 3 if idx == 1 else 5  # G 채널의 노이즈 크기는 1x1, R, B 채널의 노이즈 크기는 3x3
        blurred = cv2.medianBlur(input_per_channel[idx], filter_size)[:, :, np.newaxis]
        blur_per_channel.append(blurred)
    img = np.concatenate(blur_per_channel, axis=-1)
    return img


def apply_denoise(src: np.array, threshold: int = 30) -> np.array:
    """
    input 이미지의 노이즈를 제거. 3264x2448 기준 약 70ms 소요
    Process:
        (1) src 이미지로부터 median blur 이미지를 추출
            - 정보 손실을 최소화하기 위해 R, B 채널에는 5x5의 필터 사이즈를, G 채널에는 3x3의 필터 사이즈를 적용
        (2) src와 median blur 간 차이를 구하여 노이즈 발생 지점을 탐색
            - 어느 정도의 차이가 발생했을 때 노이즈로 간주할 것인지는 threshold에 의해 결정
        (3) 각 노이즈 지점을 각 위치에 대응되는 median blur 이미지의 픽셀값으로 대체

    Args:
        src (np.array):
            - 노이즈가 포함된 input 이미지
            - RGB(또는 BGR)의 3채널 이미지에 대한 디노이징을 지원
        threshold (int, optional):
            - 기준 픽셀이 주변 픽셀과 값이 몇 만큼 차이날 때 '노이즈'로 간주할 것인지 결정
            - Defaults to 50.

    Return:
        (np.array): 노이즈가 제거된 이미지. src와 형태 동일
    """
    src_each_channel = [deepcopy(src)[:, :, c] for c in range(3)]

    # extract statistics(local median)
    median_blur_each_channel = []
    for c in range(3):
        # Adopt filter size for B, R channel since noise size is 3x3
        # Adopt filter size for G channel since noise size is 1x1
        filter_size = 5 if c != 1 else 3
        median_blur_each_channel.append(
            cv2.medianBlur(src_each_channel[c], filter_size)
        )

    # calculate difference between origin and median-blurred image
    diff_each_channel = []
    for c in range(3):
        diff_each_channel.append(
            cv2.subtract(src_each_channel[c], median_blur_each_channel[c])
        )

    # detect noise locations according to threshold
    denoised_each_channel = []
    for c in range(3):
        diff = diff_each_channel[c]
        src_single_channel = src_each_channel[c]
        blur_single_channel = median_blur_each_channel[c]

        # use span_size to detect whole noise pixels for each noise location
        # for B, R channel, noise size is 3x3 => use span_size 3
        # for G channel, noise size is 1x1 => use span_size 1(actually not needed)
        span_size = 3 if c != 1 else 1
        diff_thresholded = (diff > threshold).astype(
            np.uint8
        ) * 100  # weight for convenience
        diff_span = cv2.blur(
            diff_thresholded, (span_size, span_size)
        )  # use mean blur for span non-zero pixel to detect whole noise locations
        detection_mask = diff_span > 0  # True for each noise position

        # imputate noise value with statistic(median)
        src_single_channel[detection_mask == True] = blur_single_channel[
            detection_mask == True
        ]
        denoised_each_channel.append(src_single_channel[:, :, np.newaxis])

    # recover denoised image as RGB(or BGR) channel
    denoised_src = np.concatenate(denoised_each_channel, axis=-1)
    return denoised_src
