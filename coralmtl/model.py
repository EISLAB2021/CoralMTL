import copy

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralFullMultiTaskModel(nn.Module):
    def __init__(
        self,
        arch="UnetPlusPlus",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=8,
        num_scales=4,
        **kwargs,
    ):
        super().__init__()

        base_model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.encoder = base_model.encoder
        try:
            decoder_channels = base_model.segmentation_head[0].in_channels
        except Exception:
            decoder_channels = base_model.segmentation_head.in_channels

        self.seg_decoder = base_model.decoder
        self.seg_head = base_model.segmentation_head
        self.edge_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

        self.geo_decoder = copy.deepcopy(base_model.decoder)
        self.normal_head = nn.Conv2d(decoder_channels, 3, kernel_size=3, padding=1)

        self.metric_decoder = copy.deepcopy(base_model.decoder)
        self.gga_attn = nn.Sequential(
            nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.fd_curve_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_scales, kernel_size=1),
        )

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image):
        image = (image - self.mean) / self.std
        features = self.encoder(image)

        seg_feat = self.seg_decoder(features)
        seg_mask = self.seg_head(seg_feat)
        pred_edges = self.edge_head(seg_feat)

        geo_feat = self.geo_decoder(features)
        normal_out = self.normal_head(geo_feat)
        normal_map = F.normalize(normal_out, dim=1)

        metric_feat = self.metric_decoder(features)
        attention_map = self.gga_attn(geo_feat.detach())
        metric_feat = metric_feat * attention_map

        fd_curve_map = self.fd_curve_head(metric_feat)

        return seg_mask, normal_map, fd_curve_map, pred_edges
