import os
from tqdm import tqdm
from copy import deepcopy
from typing import List
import numpy as np
import cv2
from utils import remove_all_files_in_dir


def cut_img(
    img_path_list: List[str],
    save_path: str,
    stride: int = 256,
    split_size: int = 512,
    train_multiscale_mode: bool = False,
    denoise_method: str = None,  # 'naive', 'selective'
    remove_exist: bool = True,
):
    save_dir = f"{save_path}_{split_size}"
    os.makedirs(save_dir, exist_ok=True)
    if remove_exist:
        remove_all_files_in_dir(save_dir)

    num = 0
    for path in tqdm(img_path_list, desc="[Cut Images]"):
        for size_idx in range(2):
            img = cv2.imread(path)

            # denoise
            if denoise_method == "naive":
                img = apply_naive_denoise(img)
            elif denoise_method == "selective":
                img = apply_denoise(img)

            top_size, left_size = img.shape[0], img.shape[1]
            if size_idx == 1:
                if train_multiscale_mode is False:
                    break
                top_size, left_size = int(top_size / 2), int(left_size / 2)
                img = cv2.resize(
                    img, dsize=(top_size, left_size), interpolation=cv2.INTER_CUBIC
                )

            for top in range(0, img.shape[0], stride):
                for left in range(0, img.shape[1], stride):
                    if (
                        top + split_size > img.shape[0]
                        or left + split_size > img.shape[1]
                    ):
                        continue
                    piece = np.zeros((split_size, split_size, 3), dtype=np.uint8)
                    temp = img[top : top + split_size, left : left + split_size, :]
                    piece[: temp.shape[0], : temp.shape[1], :] = temp

                    # save png
                    png_save_path = os.path.join(save_dir, f"{num}.png")
                    cv2.imwrite(png_save_path, piece)
                    num += 1

            # 가장 자리 1
            for left in range(0, img.shape[1], stride):
                if left + split_size > img.shape[1]:
                    continue
                piece = np.zeros((split_size, split_size, 3), np.uint8)
                temp = img[
                    img.shape[0] - split_size : img.shape[0],
                    left : left + split_size,
                    :,
                ]
                piece[: temp.shape[0], : temp.shape[1], :] = temp

                # save png
                png_save_path = os.path.join(save_dir, f"{num}.png")
                cv2.imwrite(png_save_path, piece)
                num += 1

            # 가장 자리 2
            for top in range(0, img.shape[0], stride):
                if top + split_size > img.shape[0]:
                    continue
                piece = np.zeros([split_size, split_size, 3], np.uint8)
                temp = img[
                    top : top + split_size, img.shape[1] - split_size : img.shape[1], :
                ]
                piece[: temp.shape[0], : temp.shape[1], :] = temp

                # save png
                png_save_path = os.path.join(save_dir, f"{num}.png")
                cv2.imwrite(png_save_path, piece)
                num += 1

            # 오른쪽 아래
            piece = np.zeros([split_size, split_size, 3], np.uint8)
            temp = img[
                img.shape[0] - split_size : img.shape[0],
                img.shape[1] - split_size : img.shape[1],
                :,
            ]
            piece[: temp.shape[0], : temp.shape[1], :] = temp

            # save png
            png_save_path = os.path.join(save_dir, f"{num}.png")
            cv2.imwrite(png_save_path, piece)
            num += 1

    return num


def apply_naive_denoise(src):
    input_per_channel = [deepcopy(src[:, :, i]) for i in range(3)]
    blur_per_channel = []
    for idx in range(3):
        filter_size = 3 if idx == 1 else 5  # G 채널의 노이즈 크기는 1x1, R, B 채널의 노이즈 크기는 3x3
        blurred = cv2.medianBlur(input_per_channel[idx], filter_size)[:, :, np.newaxis]
        blur_per_channel.append(blurred)
    img = np.concatenate(blur_per_channel, axis=-1)
    return img


def apply_denoise(
    src: np.array, threshold: int = 30
) -> np.array:  # NOTE. now for BGR(or RGB) channel image
    """input 이미지의 노이즈를 제거. 3264x2448 기준 약 70ms 소요
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
