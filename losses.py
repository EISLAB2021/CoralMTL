import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred_map, gt_curve, valid_mask):
        pred_vec = torch.mean(pred_map, dim=(2, 3))
        loss = self.mse(pred_vec, gt_curve)
        loss = loss * valid_mask
        return torch.sum(loss) / (torch.sum(valid_mask) + 1e-6)


def predict_fd_from_curve(pred_map, log_l):
    pred_curve = torch.mean(pred_map, dim=(2, 3))
    x = log_l
    x_mean = x.mean()
    y_mean = pred_curve.mean(dim=1, keepdim=True)
    num = torch.sum((pred_curve - y_mean) * (x - x_mean), dim=1)
    den = torch.sum((x - x_mean) ** 2)
    pred_slope = num / (den + 1e-8)
    return 3.0 - pred_slope


class DifferentiableFDSlopeLoss(nn.Module):
    def __init__(self, log_l_x):
        super().__init__()
        self.register_buffer("log_l", log_l_x)

    def forward(self, pred_map, gt_fd):
        return F.smooth_l1_loss(predict_fd_from_curve(pred_map, self.log_l), gt_fd)


class SurfaceNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_normal, gt_normal):
        cosine = torch.sum(pred_normal * gt_normal, dim=1)
        loss = 1.0 - torch.mean(cosine)
        return loss


class PixelWiseRugosityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_rugo_map, gt_normals):
        nz = gt_normals[:, 2:3, :, :]
        nz = torch.clamp(nz, min=0.1, max=1.0)
        gt_rugo_map = 1.0 / nz
        return self.mse(pred_rugo_map, gt_rugo_map)


def compute_sobel_edges(z_map):
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=z_map.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=z_map.device
    ).view(1, 1, 3, 3)
    edge_x = F.conv2d(z_map, sobel_x, padding=1)
    edge_y = F.conv2d(z_map, sobel_y, padding=1)
    edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
    return edge_mag / (edge_mag.max() + 1e-6)
