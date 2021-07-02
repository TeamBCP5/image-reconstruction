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
        label_stats = get_stats(label, prefix='label') # label statistics

        img = transforms(image=img)["image"]

        crop = []
        position = [] # (상좌표, 좌좌표)

        result_img = np.zeros_like(img.numpy().transpose(1, 2, 0)) # (H, W, C)
        voting_mask = np.zeros_like(img.numpy().transpose(1, 2, 0)) # (H, W, C)

        # (1) 왼쪽부터 오른쪽으로 stride
        # (2) 오른쪽 끝에 도달했을 경우 윗줄 가장 왼쪽에서부터 (1)을 수행
        img = img.unsqueeze(0) # [B=1, C, H, W]
        for top in range(0, img.shape[2], stride):
            for left in range(0, img.shape[3], stride):
                piece = torch.zeros([1, 3, img_size, img_size])

                temp = img[:, :, top : top + img_size, left : left + img_size] # [B=1, C, H, W]
                piece[:, :, : temp.shape[2], : temp.shape[3]] = temp

                with torch.no_grad():
                    pred = model(piece.to(device))
                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0) # [H, W, C]
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
        score.append(psnr_score(result_img.astype(float), label.astype(float), 255))
    return np.mean(score), input_stats, label_stats

def validate_each_patch(
    model,
    img_paths,
    label_paths,
    stride=512,
    img_size=256,
    transforms=None,
    device=None,
    input_save_dir='./patch_inputs/',
    label_save_dir='./patch_labels/',
) -> None: 
    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    input_stats = dict()
    label_stats = dict()
    stats = []

    model.eval()
    for img_path, label_path in tqdm(zip(img_paths, label_paths), desc='Validation'):
        input_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        img_id = int(os.path.basename(img_path).split('_')[-1].split('.')[0])

        input_img = cv2.imread(img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) # [H, W, C(RGB)]
        label_img = cv2.imread(label_path)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB) # [H, W, C(RGB)]

        input_tensor = transforms(image=input_img)["image"]

        # (1) 왼쪽부터 오른쪽으로 stride
        # (2) 오른쪽 끝에 도달했을 경우 윗줄 가장 왼쪽에서부터 (1)을 수행
        input_tensor = input_tensor.unsqueeze(0) # [B=1, C, H, W]
        for top in range(0, input_tensor.shape[2], stride):
            for left in range(0, input_tensor.shape[3], stride):
                input_tensor_piece = torch.zeros([1, 3, img_size, img_size])
                input_array_piece = np.zeros([img_size, img_size, 3])
                label_piece = np.zeros([img_size, img_size, 3]) # [H, W, C(RGB)]
                
                input_tensor_tmp = input_tensor[:, :, top : top + img_size, left : left + img_size] # [B=1, C, H, W]
                input_array_tmp = input_img[top : top + img_size, left : left + img_size, :] # [H, W, C(RGB)]
                label_tmp = label_img[top : top + img_size, left : left + img_size, :] # [H, W, C(RGB)]

                input_tensor_piece[:, :, :input_tensor_tmp.shape[2], :input_tensor_tmp.shape[3]] = input_tensor_tmp # [B=1, C, H, W]
                input_array_piece[:input_array_tmp.shape[0], :input_array_tmp.shape[1], :] = input_array_tmp
                label_piece[:label_tmp.shape[0], :label_tmp.shape[1], :] = label_tmp

                with torch.no_grad():
                    pred = model(input_tensor_piece.to(device))
                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0) # [H, W, C]
                pred = (pred * 127.5) + 127.5
                pred = pred.astype(np.int8)
                psnr = psnr_score(pred, label_piece, 255)

                # extrac stat
                input_stats = get_stats(input_array_piece, prefix='input') # input statistics - NOTE 샘플 단위 추론에만 활용
                label_stats = get_stats(label_piece, prefix='label') # label statistics
                tmp_stats = pd.concat([
                    pd.Series(input_stats).to_frame().T, # 채널별 픽셀 평균/분산
                    pd.Series(label_stats).to_frame().T # 채널별 픽셀 평균/분산
                    ], axis=1)
                tmp_stats['input'] = input_name
                tmp_stats['label'] = label_name
                tmp_stats['psnr'] = psnr # PSNR
                tmp_stats['top'] = top
                tmp_stats['left'] = top
                stats.append(tmp_stats)
                
                # save patch images - NOTE 작은 이미지로 저장함
                # save_input_piece = cv2.cvtColor(input_array_piece.astype(np.uint8), cv2.COLOR_RGB2BGR)
                # save_label_piece = cv2.cvtColor(label_piece.astype(np.uint8), cv2.COLOR_RGB2BGR)
                # save_input_piece = cv2.resize(save_input_piece, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
                # save_label_piece = cv2.resize(save_label_piece, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

                # cv2.imwrite(
                #     os.path.join(input_save_dir, f"train_{img_id}_input_patch(top({top})left({left}))_psnr({psnr:.2f}).png"), 
                #     save_input_piece
                #     )
                # cv2.imwrite(
                #     os.path.join(label_save_dir, f"train_{img_id}_label_patch(top({top})left({left}))_psnr({psnr:.2f}).png"), 
                #     save_label_piece
                #     )

    stats = pd.concat(stats, axis=0, ignore_index=True)
    stats.to_csv('patch_distribution_comparison.csv', index=False)


if __name__ == '__main__':
    import pandas as pd
    import os
    from glob import glob

    img_size = 512
    stride = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = './saved_models/baseline_ckpt.pth'
    input_dir = '/content/data/train_input_img/' # NOTE. absolute path
    label_dir = '/content/data/train_label_img/' # NOTE. absolute path
    # annotations = pd.read_csv('./sketches/light-scattering-annotations.csv')
    input_img_paths = sorted(glob(os.path.join(input_dir, '*.png')))
    label_img_paths = sorted(glob(os.path.join(label_dir, '*.png')))
    valid_transforms = get_valid_transform()

    # subsample = annotations[annotations['place'] == place].reset_index(drop=True)
    # subsample = annotations # NOTE. 장소 관계없이 추론
    # input_img_paths = subsample['input'].apply(lambda x: os.path.join(input_dir, x)).tolist()
    # label_img_paths = subsample['label'].apply(lambda x: os.path.join(label_dir, x)).tolist()
    
    model = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=3, classes=3, decoder_attention_type="scse")
    model.segmentation_head[2] = nn.Tanh()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    validate_each_patch(
        model=model, 
        img_paths=input_img_paths, 
        label_paths=label_img_paths,
        stride=stride,
        img_size=img_size,
        transforms=valid_transforms,
        device=device
    )
