import argparse
import os
from glob import glob
from tqdm import tqdm
import time
import zipfile
from copy import deepcopy
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import EvalShiftedDataset
from augmentations import get_valid_transform
from preprocessing import get_save_shifted_images

def remove_all_files_in_dir(dir):
    """앙상블 과정 중 활용되는 임시폴더 내 파일을 모두 제거"""
    for fpath in glob(os.path.join(dir, "*")):
        os.remove(fpath)

def inference(args):
    os.makedirs('/content/tmp_dir', exist_ok=True)
    remove_all_files_in_dir('tmp_dir')
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        decoder_attention_type="scse",
    )
    model.segmentation_head[2] = nn.Tanh()
    ckpt = torch.load(args.checkpoint)['model']
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)

    test_input_paths = glob(os.path.join(args.data_dir, 'test_input_img', '*.png'))
    test_dir = os.path.join(args.data_dir, "test_shifted_inputs")
    get_save_shifted_images(test_input_paths, save_dir=test_dir)

    eval_transform = get_valid_transform()
    eval_dataset = EvalShiftedDataset(source_dir=test_dir, transforms=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)

    total_eval_steps = len(eval_loader)
    print(
        "[+] Train Description\n",
        f"Test Data Size: {len(eval_dataset)}\n",
        f"Test Batch Size: {args.batch_size}\n",
    )
    
    # inference
    time_check = time.time()
    for batch in tqdm(eval_loader, desc='[Inference]'):
        img_names = batch['img_id']
        images = batch['image']
        batch_size = images.size(0)

        with torch.no_grad():
            preds = model(images.to(device))

        preds = preds.cpu().clone().detach().numpy()
        preds = preds.transpose(0, 2, 3, 1) # [B, H, W, C]
        preds = (preds * 127.5) + 127.5
        preds = preds.astype(np.uint8)

        for idx in range(batch_size):
            img_id = img_names[idx]
            predicted = preds[idx, :, :, :].squeeze()
            predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)
            save_path = os.path.join('/content/tmp_dir', img_id)
            cv2.imwrite(save_path, predicted)

    # group shifted images for recover
    shifted_img_paths = sorted(glob(os.path.join("/content/tmp_dir", '*')))
    img_id_groups = dict()
    img_id_group = []
    current_id = None
    for idx, path in enumerate(shifted_img_paths):
        img_id = os.path.basename(path).split('(')[0]
        if current_id is None:
            current_id = img_id
            img_id_group.append(path)
        elif img_id != current_id:
            img_id_groups[current_id] = deepcopy(img_id_group)
            current_id = img_id
            img_id_group = [path]
        elif idx == len(shifted_img_paths) - 1:
            img_id_group.append(path)
            img_id_groups[current_id] = deepcopy(img_id_group)
        else:
            img_id_group.append(path)

    # recover phase
    for img_id in tqdm(sorted(list(img_id_groups.keys())), desc='[Recover]'):
        group_paths = img_id_groups[img_id]
        square_length = (len(group_paths) // 6) * 544
        recover_h = (square_length // 4) * 3

        recover = np.zeros((square_length, square_length, 3), dtype=np.uint8)
        for path in group_paths:
            w, h = path.split(img_id)[-1].split('.')[0].split(',')
            w, h = int(w[-1]), int(h[0])
            shifted_img = cv2.imread(path)
            recover[:, w::6, :][h::6, :, :] = shifted_img
        
        recover = recover[(square_length-recover_h)//2:(square_length+recover_h)//2, :, :]
        cv2.imwrite(os.path.join(args.save_dir, f'{img_id}.png'), recover)
    
    # make zip file for submission
    submission = zipfile.ZipFile('submission.zip', 'w')
    for path in sorted(glob(os.path.join(args.save_dir, '*.png'))):
        submission.write(path)
    submission.close()


def _inference(model, img_paths, img_size=512, stride=512, transforms=None, device=None):
    results = []
    for img_path, label_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms(image=img)['image']

        crop = []
        position = []

        result_img = np.zeros_like(img.numpy().transpose(1,2,0))
        voting_mask = np.zeros_like(img.numpy().transpose(1,2,0))
        
        img = img.unsqueeze(0)
        for top in range(0, img.shape[2], stride):
            for left in range(0, img.shape[3], stride):
                piece = torch.zeros([1, 3, img_size, img_size])
                temp = img[:, :, top:top+img_size, left:left+img_size]
                piece[:, :, :temp.shape[2], :temp.shape[3]] = temp
                with torch.no_grad():
                    pred = model(piece.to(device))
                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = (pred*127.5)+127.5
                crop.append(pred)
                position.append([top, left])

        crop = np.array(crop).astype(np.uint16)
        for num, (t, l) in enumerate(position):
            piece = crop[num]
            h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
            result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w, :]
            voting_mask[t:t+img_size, l:l+img_size, :] += 1
        
        result_img = result_img / voting_mask
        result_img = result_img.astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        results.append(result_img)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/content/data/")
    parser.add_argument("--checkpoint", default="best_shifted_unet.pth")
    parser.add_argument("--batch-size", default=64)
    parser.add_argument("--save_dir", default="submission")
    args = parser.parse_args()
    inference(args)
    