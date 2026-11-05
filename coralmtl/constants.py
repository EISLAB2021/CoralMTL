import os
import random

import numpy as np
import torch

SCALES_PX = [64, 128, 256, 512]

PIXEL_PITCH = 0.0005

LOG_L_X = torch.tensor(
    [np.log10(s * PIXEL_PITCH) for s in SCALES_PX], dtype=torch.float32
)

CURVE_COLS = ["Log_H_64px", "Log_H_128px", "Log_H_256px", "Log_H_512px"]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_r2(y_true, y_pred):
    y_true = y_true.view(-1).cpu()
    y_pred = y_pred.view(-1).cpu()
    target_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - target_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-6)
    return r2.item()
