import os
import gc
from glob import glob
from psutil import virtual_memory
import zipfile
import random
import numpy as np
import cv2
import torch
from torch import nn
import segmentation_models_pytorch as smp
from importlib import import_module
from networks import HINet, define_D


def get_model(args, mode="train") -> nn.Module:
    if args.network.name == "pix2pix":
        if mode == "train":
            config_G = args.network.generator._asdict()  # encoder_name, ...
            config_D = args.network.discriminator._asdict()  # ndf, ...
            G_model = smp.Unet(**config_G)
            G_model.segmentation_head[2] = nn.Tanh()
            D_model = define_D(**config_D)
            return G_model, D_model

        else:  # 'valid' or 'test'
            config_G = args.network.generator._asdict()  # encoder_name, ...
            config_G['encoder_weights'] = None
            G_model = smp.Unet(**config_G)
            return G_model

    elif args.network.name == "hinet":
        model = HINet(depth=4)
        return model

    else:
        raise NotImplementedError(f"There's no model_type '{args.network.name}'")


def get_optimizer(args, *models):
    optimizers = []
    for model in models:
        params = [p for p in model.parameters() if p.requires_grad]
        optim_module = getattr(
            import_module("torch.optim"), args.optimizer.name
        )  # Adam, AdamW, ...
        optimizer = optim_module(params, lr=args.optimizer.lr)
        optimizers.append(optimizer)

    if len(optimizers) == 1:
        optimizers = optimizers[0]

    return optimizers


def get_scheduler(args, *optimizers):
    config_dict = args.scheduler._asdict()
    scheduler_type = config_dict.pop("name")

    schedulers = []
    for opt in optimizers:
        scheduler_module = getattr(
            import_module("torch.optim.lr_scheduler"), scheduler_type
        )  # ReduceLROnPlateau, ...
        scheduler = scheduler_module(opt, **config_dict)
        schedulers.append(scheduler)

    if len(schedulers) == 1:
        schedulers = schedulers[0]

    return schedulers


def remove_all_files_in_dir(dir):
    """dir 내 모든 파일을 제거"""
    for fpath in glob(os.path.join(dir, "*")):
        os.remove(fpath)


def save_samples(result, save_dir="./submission/"):
    with zipfile.ZipFile(os.path.join(save_dir, "submission.zip"), "w") as img_out:
        for i, image in enumerate(result):
            image = cv2.imencode(".png", image)[1]
            img_out.writestr(f"test_{20000+i}.png", image)


def set_seed(seed: int = 41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_gpu_status() -> None:
    """GPU 이용 상태를 출력"""
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
    allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
    free = reserved - allocated
    print(
        "[+] GPU Status\n",
        f"Total: {total_mem:.4f} GB\n",
        f"Reserved: {reserved:.4f} GB\n",
        f"Allocated: {allocated:.4f} GB\n",
        f"Residue: {free:.4f} GB\n",
    )


def filter_img_id(img_name):
    return int(img_name.split("_")[2].split(".")[0])


def print_system_envs():
    """시스템 환경을 출력"""
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    cpu_mem_size = virtual_memory().available // (1024 ** 3)
    gpu_mem_size = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(
        "[+] System environments\n",
        f"Number of GPUs : {num_gpus}\n",
        f"Number of CPUs : {num_cpus}\n",
        f"CPU Memory Size : {cpu_mem_size:.4f} GB\n",
        f"GPU Memory Size : {gpu_mem_size:.4f} GB\n",
    )


def truncate_aligned_model(model: nn.Module) -> None:
    """입력된 모델을 kill하는 함수. 한정된 GPU 자원의 과부하 방지를 위해 사용"""
    del model
    gc.collect()
    torch.cuda.empty_cache()
