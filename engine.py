from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from .constants import compute_r2
from .losses import compute_sobel_edges, predict_fd_from_curve


def run_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_funcs,
    weights,
    device,
    scaler,
    log_l_device,
    is_train=True,
):
    if is_train:
        model.train()
    else:
        model.eval()

    metrics = defaultdict(float)
    ru_gts, ru_preds, fd_gts, fd_preds = [], [], [], []

    pbar = tqdm(dataloader, desc="Train" if is_train else "Valid")

    for batch in pbar:
        images, masks, dems, gt_normals, gt_fd, gt_curve, gt_valid = [
            b.to(device) for b in batch
        ]

        if is_train:
            optimizer.zero_grad()

        with autocast():
            pred_masks, pred_normals, pred_fd_map, pred_edges = model(images)

            l_seg = loss_funcs["seg"](pred_masks, masks)
            l_norm = loss_funcs["normal"](pred_normals, gt_normals)
            l_fd_curve = loss_funcs["fd_curve"](pred_fd_map, gt_curve, gt_valid)
            l_fd_slope = loss_funcs["fd_slope"](pred_fd_map, gt_fd)

            pred_norm_z = torch.clamp(pred_normals[:, 2:3, :, :], 0.1, 1.0)
            derived_rugo = 1.0 / pred_norm_z
            l_ru = loss_funcs["rugo_geo"](derived_rugo, gt_normals)

            gt_edges = compute_sobel_edges(gt_normals[:, 2:3, :, :])
            l_edge = F.mse_loss(torch.sigmoid(pred_edges), gt_edges)

            total_loss = (
                weights["seg"] * l_seg
                + weights["normal"] * l_norm
                + weights["fd_curve"] * l_fd_curve
                + weights["fd_slope"] * l_fd_slope
                + weights["ru"] * l_ru
                + weights["edge"] * l_edge
            )

        if is_train:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        metrics["total"] += total_loss.item()
        metrics["seg"] += l_seg.item()
        metrics["edge"] += l_edge.item()
        metrics["fd_cv"] += l_fd_curve.item()
        metrics["fd_sl"] += l_fd_slope.item()
        metrics["ru"] += l_ru.item()

        pbar.set_postfix(
            {"Total": f"{total_loss.item():.2f}", "Seg": f"{l_seg.item():.3f}"}
        )

        if not is_train:
            ru_gt = torch.mean(
                1.0 / torch.clamp(gt_normals[:, 2, :, :], 0.1, 1.0), dim=(1, 2)
            )
            ru_pred = torch.mean(
                1.0 / torch.clamp(pred_normals[:, 2, :, :], 0.1, 1.0), dim=(1, 2)
            )
            ru_gts.extend(ru_gt.cpu().tolist())
            ru_preds.extend(ru_pred.cpu().tolist())

            p_fd = predict_fd_from_curve(pred_fd_map, log_l_device)
            fd_gts.extend(gt_fd.cpu().tolist())
            fd_preds.extend(p_fd.cpu().tolist())

    if is_train and scheduler:
        scheduler.step()

    avg_metrics = {k: v / len(dataloader) for k, v in metrics.items()}

    if not is_train and len(ru_gts) > 1:
        avg_metrics["ru_r2"] = compute_r2(torch.tensor(ru_gts), torch.tensor(ru_preds))
        avg_metrics["fd_r2"] = compute_r2(torch.tensor(fd_gts), torch.tensor(fd_preds))

    return avg_metrics
