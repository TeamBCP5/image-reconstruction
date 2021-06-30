import os
import pickle
import random
import numpy as np
import torch


def set_seed(seed: int = 41):
    """시드값을 고정하는 함수. 실험 재현을 위해 사용"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int = 41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)