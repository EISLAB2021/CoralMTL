import os

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset as BaseDataset

from .constants import CURVE_COLS, PIXEL_PITCH


def compute_normal_numpy(dem_np):
    dzdx = cv2.Sobel(dem_np, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(dem_np, cv2.CV_32F, 0, 1, ksize=3)
    pixel_size = PIXEL_PITCH
    z_component = np.ones_like(dem_np) * 8.0 * pixel_size
    normal = np.stack([-dzdx, -dzdy, z_component], axis=-1)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    return normal / (norm + 1e-6)


class CoralDataset(BaseDataset):

    def __init__(
        self,
        images_dir,
        masks_dir,
        dems_dir,
        csv_file,
        input_image_reshape=(512, 512),
        color_aug=None,
        spatial_aug=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.dems_dir = dems_dir
        self.input_image_reshape = input_image_reshape
        self.color_aug = color_aug
        self.spatial_aug = spatial_aug

        self.ids = [
            x
            for x in os.listdir(images_dir)
            if x.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
        ]
        self.labels_df = pd.read_csv(csv_file)
        self.labels_df["stem"] = self.labels_df["Filename"].apply(
            lambda x: os.path.splitext(x)[0]
        )
        self.label_map = self.labels_df.set_index("stem").to_dict("index")

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __getitem__(self, i):
        img_stem = os.path.splitext(self.ids[i])[0]

        image = cv2.imread(
            os.path.join(self.images_dir, self.ids[i]), cv2.IMREAD_COLOR
        )
        image = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.input_image_reshape
        )

        mask_path = os.path.join(self.masks_dir, img_stem + ".png")
        mask = (
            cv2.resize(
                cv2.imread(mask_path, 0),
                self.input_image_reshape,
                interpolation=cv2.INTER_NEAREST,
            )
            if os.path.exists(mask_path)
            else np.zeros(self.input_image_reshape, dtype=np.uint8)
        )

        dem_path = os.path.join(self.dems_dir, img_stem + ".tif")
        if not os.path.exists(dem_path):
            dem_path = os.path.join(self.dems_dir, img_stem + ".tiff")
        dem = (
            cv2.resize(
                np.nan_to_num(tifffile.imread(dem_path), nan=0.0),
                self.input_image_reshape,
                interpolation=cv2.INTER_LINEAR,
            )
            if os.path.exists(dem_path)
            else np.zeros(self.input_image_reshape, dtype=np.float32)
        )

        row = self.label_map[img_stem]
        target_fd = float(row.get("FD_Variation", row.get("fractal_dimension", 0)))
        t_curve = torch.tensor([float(row[c]) for c in CURVE_COLS], dtype=torch.float32)
        valid = 1.0 if float(row.get("R_Squared", 1.0)) >= 0.9 else 0.0
        t_valid = torch.full((4,), valid, dtype=torch.float32)

        if self.spatial_aug:
            aug = self.spatial_aug(image=image, mask=mask, dem=dem)
            image, mask, dem = aug["image"], aug["mask"], aug["dem"]

        normal = compute_normal_numpy(dem)
        if self.color_aug:
            image = self.color_aug(image=image)["image"]

        image = torch.from_numpy(image.astype("float32") / 255.0).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()
        dem_tensor = (
            torch.tensor(np.expand_dims(dem, axis=-1) if dem.ndim == 2 else dem)
            .float()
            .permute(2, 0, 1)
        )
        normal_tensor = torch.tensor(normal).float().permute(2, 0, 1)

        return (
            image,
            mask,
            dem_tensor,
            normal_tensor,
            torch.tensor(target_fd, dtype=torch.float32),
            t_curve,
            t_valid,
        )

    def __len__(self):
        return len(self.ids)
