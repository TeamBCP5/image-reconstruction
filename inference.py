from tqdm import tqdm
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torch.nn.functional as F
import argparse
import glob

from utils import save_samples, get_model
from data import get_valid_transform, EvalDataset


def predict(main_model, post_model, device, img_paths, transforms, arg):
    main_model.eval()
    post_model.eval()

    size = arg.size
    stride = arg.stride
    batch_size = arg.batch_size

    results = []

    pbar = tqdm(img_paths, total=len(img_paths), position=0, leave=True)
    for img_path in pbar:
        ds = EvalDataset(img_path, size=size, stride=stride, transforms=transforms)
        dl = DataLoader(ds, 
                        batch_size=batch_size, 
                        shuffle=False)
        preds = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
        votes = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
        
        for images, (x1, x2, y1, y2) in dl:
            with autocast():
                pred = main_model(images.to(device).float())
            pred = ((pred*0.5) + 0.5)
            for i in range(len(x1)):
                preds[:, x1[i]:x2[i], y1[i]:y2[i]] += pred[i]
                votes[:, x1[i]:x2[i], y1[i]:y2[i]] += 1
        preds /= votes
        preds = F.interpolate(preds.unsqueeze(0), size=(1224, 1632), mode='bilinear', align_corners=False)
        with autocast():
            post_preds = post_model(preds)
        post_preds = F.interpolate(post_preds[-1], size=(2448, 3264), mode='bicubic', align_corners=False)
        post_preds = torch.clamp(post_preds, 0, 1) * 255

        result_img = post_preds[0].cpu().detach().numpy()
        result_img = result_img.transpose(1,2,0)
        result_img = result_img.astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        results.append(result_img)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pix2pix",
        dest="pix2pix",
        default="./trained/pix2pix/pix2pix.pth",
        help="pix2pix 학습 모델 경로",
    )
    parser.add_argument(
        "--hinet",
        dest="hinet",
        default="./trained/hinet/hinet.pth",
        help="HiNet 학습 모델 경로",
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        default="./input/test_input_img",
        help="추론 시 활용할 데이터 경로",
    )
    parser.add_argument(
        "--size", dest="size", default=512, type=int, help="추론 시 사용될 윈도우의 크기"
    )
    parser.add_argument(
        "--stride", dest="stride", default=256, type=int, help="추론 시 사용될 stride의 크기"
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=32,
        type=int,
        help="추론 시 사용될 배치의 크기"
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default="./submission/",
        type=str,
        help="추론 결과를 저장할 디렉토리 경로",
    )

    parser = parser.parse_args()
    pix2pix = get_model("pix2pix", encoder_weights=None, mode="inference")
    pix2pix.load_state_dict(torch.load(parser.pix2pix))

    hinet = get_model("hinet")
    hinet.load_state_dict(torch.load(parser.hinet))

    test_img_paths = sorted(glob.glob(f"{parser.image_path}/*.png"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        results = predict(
            pix2pix.to(device), hinet.to(device), device, test_img_paths, get_valid_transform(), parser
        )

    save_samples(results, epoch=0, save_dir=parser.output_dir)
