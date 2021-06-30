from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
import segmentation_models_pytorch as smp
from metrics import psnr_score
from train_utils import cut_img
from dataset import CustomDataset
from augmentations import get_valid_transform


def get_stats(image_array: np.array, prefix: str='input'): # NOTE. image_array: RGB Scale
    stat_dict = dict()
    avg_r = image_array[:, :, 0].flatten().mean()
    avg_g = image_array[:, :, 1].flatten().mean()
    avg_b = image_array[:, :, 2].flatten().mean()
    avg_all = image_array.flatten().mean()
    std_r = image_array[:, :, 0].flatten().std()
    std_g = image_array[:, :, 1].flatten().std()
    std_b = image_array[:, :, 2].flatten().std()
    std_all = image_array.flatten().std()
    stat_dict[f'{prefix}_avg_r'] = avg_r
    stat_dict[f'{prefix}_avg_g'] = avg_g
    stat_dict[f'{prefix}_avg_b'] = avg_b
    stat_dict[f'{prefix}_avg_all'] = avg_all
    stat_dict[f'{prefix}_std_r'] = std_r
    stat_dict[f'{prefix}_std_g'] = std_g
    stat_dict[f'{prefix}_std_b'] = std_b
    stat_dict[f'{prefix}_std_all'] = std_all
    return stat_dict

def validate(
    model,
    img_paths,
    label_paths,
    stride=512,
    img_size=256,
    transforms=None,
    device=None,
): 
    input_stats = dict()
    label_stats = dict()

    model.eval()
    results = []
    score = []
    for img_path, label_path in tqdm(zip(img_paths, label_paths)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        input_stats = get_stats(img, prefix='input') # input statistics - NOTE 샘플 단위 추론에만 활용
        label_stats = get_stats(img, prefix='label') # label statistics

        img = transforms(image=img)["image"]

        crop = []
        position = []

        result_img = np.zeros_like(img.numpy().transpose(1, 2, 0))
        voting_mask = np.zeros_like(img.numpy().transpose(1, 2, 0))

        img = img.unsqueeze(0)
        for top in range(0, img.shape[2], stride):
            for left in range(0, img.shape[3], stride):
                piece = torch.zeros([1, 3, img_size, img_size])

                temp = img[:, :, top : top + img_size, left : left + img_size]
                piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                with torch.no_grad():
                    pred = model(piece.to(device))
                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = (pred * 127.5) + 127.5
                crop.append(pred)

                position.append([top, left])

        crop = np.array(crop)
        for num, (t, l) in enumerate(position):
            piece = crop[num].astype(np.uint8)
            h, w, c = result_img[t : t + img_size, l : l + img_size, :].shape
            result_img[t : t + img_size, l : l + img_size, :] += piece[:h, :w, :]
            voting_mask[t : t + img_size, l : l + img_size, :] += 1

        result_img = result_img / voting_mask
        result_img = result_img.astype(np.uint8)
        results.append(result_img)

        score.append(psnr_score(result_img, label, 255))

    return np.mean(score), input_stats, label_stats


if __name__ == '__main__':
    import pandas as pd
    import os
    from glob import glob

    img_size = 512
    stride = 512
    place = 'parking_lot'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = './saved_models/baseline_ckpt.pth'
    input_dir = '/content/data/train_input_img/' # NOTE. absolute path
    label_dir = '/content/data/train_label_img/' # NOTE. absolute path
    annotations = pd.read_csv('./sketches/light-scattering-annotations.csv')

    # subsample = annotations[annotations['place'] == place].reset_index(drop=True)
    subsample = annotations # NOTE. 장소 관계없이 추론
    input_img_paths = subsample['input'].apply(lambda x: os.path.join(input_dir, x)).tolist()
    label_img_paths = subsample['label'].apply(lambda x: os.path.join(label_dir, x)).tolist()
    valid_transforms = get_valid_transform()

    model = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=3, classes=3, decoder_attention_type="scse")
    model.segmentation_head[2] = nn.Tanh()
    model.load_state_dict(torch.load(save_path))
    model.eval()
    model.to(device)

    # NOTE. global
    # psnr = validate(
    #         model=model, 
    #         img_paths=input_img_paths, 
    #         label_paths=label_img_paths, 
    #         stride=stride,
    #         img_size=img_size,
    #         transforms=valid_transforms,
    #         device=device
    #         )
    # print(f'PSNR: {psnr}')
    # NOTE. local
    stats = []
    for input_, target in zip(input_img_paths, label_img_paths):
        input_name = os.path.basename(input_)
        label_name = os.path.basename(target)
        place = annotations[annotations['input'] == input_name]['place'].values[0]
        min_light_num = annotations[annotations['input'] == input_name]['min_light_num'].values[0]

        psnr, input_stats, label_stats = validate(
            model=model, 
            img_paths=[input_], 
            label_paths=[target], 
            stride=stride,
            img_size=img_size,
            transforms=valid_transforms,
            device=device
            )
        tmp_stats = pd.concat([
            pd.Series(input_stats).to_frame().T, # 채널별 픽셀 평균/분산
            pd.Series(label_stats).to_frame().T # 채널별 픽셀 평균/분산
            ], axis=1)
        tmp_stats['input'] = input_name
        tmp_stats['label'] = label_name
        tmp_stats['psnr'] = psnr # PSNR
        stats.append(tmp_stats)
        print(f'SAMPLE NAME: {input_name}\n * PLACE: {place}\n * MIN LIGHT NUMS: {min_light_num}\n * PSNR: {psnr}')

    stats = pd.concat(stats, axis=0, ignore_index=True)
    subsample = subsample.merge(stats, how='left', on=['input', 'label'])
    subsample.to_csv('./sketches/distribution_comparison.csv', index=False)
