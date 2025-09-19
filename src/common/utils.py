
import numpy as np
import skimage as ski
import os
from colorama import Fore
import sys
import torch

def image_to_patches(img, patch_size=224):
    # Setting the offset of 11 due to initial columns with no data
    img = img[:, 11:, :]
    H, W, C = img.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    cropped_img = img[:n_rows * patch_size, :n_cols * patch_size, :]
    patches = cropped_img.reshape(n_rows, patch_size, n_cols, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, patch_size, patch_size, C)
    print(patches.shape)
    return patches

def makedir(path_dir):
    try:
        os.makedirs(path_dir)
        print(f'OK: Directory {path_dir} created')
    except OSError as error:
        print(Fore.YELLOW + f'Diretório {path_dir} já existe')
        print(error)
        print(Fore.WHITE)

def check_dir_exists(input_path):
    try:
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f'Input directory {input_path} does not exist')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

def normalize_rgb_image(image, min_val=0, max_val=255):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (max_val - min_val) + min_val
    image = np.clip(image, min_val, max_val)
    image = image.astype(np.uint8)
    return image

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


def update_metrics(preds, targets, metrics, mode):
    for metric in metrics:
        metric.update(preds, targets)


def compute_metrics(metrics, mode):
    metrics_dict = {}
    for metric in metrics:
        metric_comp = metric.compute().item()    
        metrics_dict[f'{metric.__class__.__name__}_{mode}'] = metric.compute().item()
    return metrics_dict





def log_metrics(writer, metrics: dict, step: int, prefix: str = ""):
    """
    metrics: e.g. {"loss": 0.5, "acc": 0.82, "lr": 1e-3}
    step: global step or epoch
    prefix: optional string like "train/" or "val/"
    """
    for k, v in metrics.items():
        if k != 'epoch':
            tag = f"{prefix}{k}" if prefix else k
            # If value is a single scalar (int/float), log it
            writer.add_scalar(tag, float(v), step)