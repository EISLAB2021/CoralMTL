import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from coralmtl.constants import LOG_L_X, seed_everything
from coralmtl.data import CoralDataset
from coralmtl.engine import run_epoch
from coralmtl.losses import (
    CurveRegressionLoss,
    DifferentiableFDSlopeLoss,
    PixelWiseRugosityLoss,
    SurfaceNormalLoss,
)
from coralmtl.model import CoralFullMultiTaskModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(os.cpu_count())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

BATCH_SIZE = 8
EPOCHS_MAX = 100
ADAM_LR = 1e-4
WEIGHT_DECAY = 1e-4
ENCODER = "resnet34"
ARCH = "UnetPlusPlus"
NUM_CLASSES = 8
NUM_WORKERS = 4

LOSS_WEIGHTS = {
    "seg": 1.0,
    "normal": 1.0,
    "fd_curve": 5.0,
    "fd_slope": 1.0,
    "ru": 2.0,
    "edge": 1.0,
}


def main():
    seed_everything(42)

    x_train_dir = os.path.join(DATA_ROOT, "train", "images")
    y_train_dir = os.path.join(DATA_ROOT, "train", "mask")
    z_train_dir = os.path.join(DATA_ROOT, "train", "dems")
    csv_train_path = os.path.join(DATA_ROOT, "train", "train_fd.csv")

    x_val_dir = os.path.join(DATA_ROOT, "val", "images")
    y_val_dir = os.path.join(DATA_ROOT, "val", "mask")
    z_val_dir = os.path.join(DATA_ROOT, "val", "dems")
    csv_val_path = os.path.join(DATA_ROOT, "val", "val_fd.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(
        f"\n{'='*50}\nStarting Full Model Training "
        f"(No Depth Loss, Encoder: {ENCODER})\n{'='*50}"
    )

    spatial_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
        ],
        additional_targets={"dem": "image"},
    )
    color_transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=20, b_shift_limit=20, p=0.5),
        ]
    )

    train_dataset = CoralDataset(
        x_train_dir,
        y_train_dir,
        z_train_dir,
        csv_train_path,
        color_aug=color_transform,
        spatial_aug=spatial_transform,
    )
    valid_dataset = CoralDataset(x_val_dir, y_val_dir, z_val_dir, csv_val_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = CoralFullMultiTaskModel(
        ARCH, ENCODER, in_channels=3, out_classes=NUM_CLASSES
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_MAX, eta_min=1e-6
    )
    scaler = GradScaler()
    log_l_device = LOG_L_X.to(device)

    loss_funcs = {
        "seg": nn.CrossEntropyLoss(ignore_index=255),
        "normal": SurfaceNormalLoss().to(device),
        "fd_curve": CurveRegressionLoss().to(device),
        "fd_slope": DifferentiableFDSlopeLoss(log_l_device).to(device),
        "rugo_geo": PixelWiseRugosityLoss().to(device),
    }

    metrics_log = []
    best_seg_loss = float("inf")
    best_fd_r2 = -float("inf")
    best_ru_r2 = -float("inf")

    for epoch in range(EPOCHS_MAX):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS_MAX}")

        train_metrics = run_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            loss_funcs,
            LOSS_WEIGHTS,
            device,
            scaler,
            log_l_device,
            is_train=True,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                valid_dataloader,
                optimizer,
                scheduler,
                loss_funcs,
                LOSS_WEIGHTS,
                device,
                scaler,
                log_l_device,
                is_train=False,
            )

        logging.info(
            f"[Train] Total: {train_metrics['total']:.4f} | "
            f"Seg: {train_metrics['seg']:.4f}"
        )
        logging.info(
            f"[Valid] Seg: {val_metrics.get('seg',0):.4f} | "
            f"FD R2: {val_metrics.get('fd_r2',0):.4f} | "
            f"RU R2: {val_metrics.get('ru_r2',0):.4f}"
        )

        if val_metrics.get("seg", float("inf")) < best_seg_loss:
            best_seg_loss = val_metrics["seg"]
            torch.save(
                model.state_dict(), os.path.join(OUTPUT_DIR, "best_seg_model.pth")
            )

        if val_metrics.get("fd_r2", -100) > best_fd_r2:
            best_fd_r2 = val_metrics["fd_r2"]
            torch.save(
                model.state_dict(), os.path.join(OUTPUT_DIR, "best_fd_r2_model.pth")
            )

        if val_metrics.get("ru_r2", -100) > best_ru_r2:
            best_ru_r2 = val_metrics["ru_r2"]
            torch.save(
                model.state_dict(), os.path.join(OUTPUT_DIR, "best_ru_r2_model.pth")
            )

        metrics_log.append(
            {
                "epoch": epoch + 1,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        pd.DataFrame(metrics_log).to_csv(
            os.path.join(OUTPUT_DIR, "metrics.csv"), index=False
        )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
