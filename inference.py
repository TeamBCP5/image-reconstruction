import os
import argparse
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F
from utils import save_samples, get_model, Flags, print_arguments
from data import get_valid_transform, CutImageDataset


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    main_model = get_model(Flags(args.cfg_main).get(), mode="test")
    try:
        main_model.load_state_dict(torch.load(args.ckpt_main)["G_model"])
    except:
        main_model.load_state_dict(torch.load(args.ckpt_main))
    main_model.to(device)
    main_model.eval()

    post_model = get_model(Flags(args.cfg_post).get(), mode="test")
    try:
        post_model.load_state_dict(torch.load(args.ckpt_post)["model"])
    except:
        post_model.load_state_dict(torch.load(args.ckpt_post))
    post_model.to(device)
    post_model.eval()

    # set preprocessing process
    img_paths = sorted(glob(os.path.join(args.img_dir, "*.png")))
    patch_size = args.patch_size
    stride = args.stride
    batch_size = args.batch_size
    transforms = get_valid_transform("inference")

    # inference
    with torch.no_grad():
        results = []
        pbar = tqdm(
            img_paths, total=len(img_paths), position=0, leave=True, desc="[Inference]"
        )
        for img_path in pbar:
            ds = CutImageDataset(
                img_path, patch_size=patch_size, stride=stride, transforms=transforms
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

            # main light scattering reduction(pix2pix)
            preds = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            votes = torch.zeros(3, ds.shape[0], ds.shape[1]).to(device)
            for images, (x1, x2, y1, y2) in dl:
                with autocast():
                    pred = main_model(images.to(device).float())
                    pred = (pred * 0.5) + 0.5
                for i in range(len(x1)):
                    preds[:, x1[i] : x2[i], y1[i] : y2[i]] += pred[i]
                    votes[:, x1[i] : x2[i], y1[i] : y2[i]] += 1
            preds /= votes
            preds = F.interpolate(
                preds.unsqueeze(0),
                size=(1224, 1632),
                mode="bilinear",
                align_corners=False,
            )

            # postprocessing(hinet)
            with autocast():
                post_preds = post_model(preds)
            post_preds = F.interpolate(
                post_preds[-1], size=(2448, 3264), mode="bicubic", align_corners=False
            )
            post_preds = torch.clamp(post_preds, 0, 1) * 255

            result_img = post_preds[0].cpu().detach().numpy()
            result_img = result_img.transpose(1, 2, 0)
            result_img = result_img.astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

            results.append(result_img)  # (IMG_ID, np.array)

    # save predicted images
    save_samples(results, save_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_main",
        dest="cfg_main",
        default="./configs/Pix2Pix.yaml",
        help="Main 모델 config 파일 경로",
    )
    parser.add_argument(
        "--config_post",
        dest="cfg_post",
        default="./configs/HINet_phase1.yaml",
        help="Postprocessing 모델 config 파일 경로",
    )
    parser.add_argument(
        "--checkpoint_main",
        dest="ckpt_main",
        default="./checkpoints/pix2pix/pix2pix.pth",
        help="학습한 main 모델 경로",
    )
    parser.add_argument(
        "--checkpoint_post",
        dest="ckpt_post",
        default="./checkpoints/hinet/hinet.pth",
        help="학습한 postprocessing 모델 경로",
    )
    parser.add_argument(
        "--image_dir",
        dest="img_dir",
        default="/content/data/test_input_img",
        help="추론 시 활용할 데이터 경로",
    )
    parser.add_argument("--patch_size", default=512, type=int, help="추론 시 사용될 윈도우의 크기")
    parser.add_argument("--stride", default=256, type=int, help="추론 시 사용될 stride의 크기")
    parser.add_argument("--batch_size", default=32, type=int, help="추론 시 사용될 배치의 크기")
    parser.add_argument(
        "--output_dir", default="./submission/", type=str, help="추론 결과를 저장할 디렉토리 경로"
    )

    args = parser.parse_args()
    print_arguments(args)
    predict(args)
