import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ALOCC, DEVICE
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve
)
from scipy.optimize import brentq
from piq import ssim, psnr, vif_p, multi_scale_gmsd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from utils import timer
from contextlib import contextmanager, nullcontext
# ============================================================================
# 自研的多尺度 VIF（不依赖库，适配小分辨率）
# ============================================================================
# 基于局部高斯窗口的统计和多尺度降采样（接近 VIFp 思路，但窗口可控、不受 41×41 限制）
def _gaussian_kernel2d(window_size: int = 7, sigma: float = 1.5, device=DEVICE, dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g = g / (g.sum() + 1e-12)
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return k2d


def _local_stats_xy(x: torch.Tensor, y: torch.Tensor, kernel2d: torch.Tensor):
    # x, y: [B, C, H, W]; kernel2d: [1,1,kh,kw]
    C = x.shape[1]
    kh, kw = kernel2d.shape[-2], kernel2d.shape[-1]
    k = kernel2d.expand(C, 1, kh, kw)
    pad = (kw // 2, kw - kw // 2, kh // 2, kh - kh // 2)
    mu_x = F.conv2d(x, k, padding=(kh // 2, kw // 2), groups=C)
    mu_y = F.conv2d(y, k, padding=(kh // 2, kw // 2), groups=C)
    sigma_x_sq = F.conv2d(x * x, k, padding=(kh // 2, kw // 2), groups=C) - mu_x * mu_x
    sigma_y_sq = F.conv2d(y * y, k, padding=(kh // 2, kw // 2), groups=C) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, k, padding=(kh // 2, kw // 2), groups=C) - mu_x * mu_y
    return mu_x, mu_y, sigma_x_sq, sigma_y_sq, sigma_xy


def _vif_ms_custom(ref: torch.Tensor, dist: torch.Tensor, levels: int = 4, window_size: int = 5, sigma: float = 1.5, eps: float = 1e-10) -> torch.Tensor:
    """Multi-scale VIF (custom) on [0,1] images. Returns per-sample VIF, shape [B].
    标准VIF公式实现：
    - ref: reference/clean image (原图)
    - dist: distorted/test image (重建图)

    VIF measures how much information from reference is preserved in distorted image.
    Higher VIF = better quality (more information preserved).
    """
    assert ref.shape == dist.shape, 'Shapes of ref and dist must match'
    device, dtype = ref.device, ref.dtype
    B = ref.shape[0]
    num_sum = torch.zeros(B, device=device, dtype=dtype)
    den_sum = torch.zeros(B, device=device, dtype=dtype)

    cur_ref, cur_dist = ref, dist
    for s in range(levels):
        H, W = cur_ref.shape[-2], cur_ref.shape[-1]
        if min(H, W) < window_size:
            break
        k2d = _gaussian_kernel2d(window_size, sigma, device=device, dtype=dtype)
        _, _, sigma_ref_sq, sigma_dist_sq, sigma_ref_dist = _local_stats_xy(cur_ref, cur_dist, k2d)

        # 标准VIF公式：
        # g = sigma_ref_dist / sigma_ref_sq  (增益基于参考图像方差)
        # sigma_n_sq = sigma_dist_sq - g^2 * sigma_ref_sq  (噪声方差)
        # I_ref = log(1 + sigma_ref_sq / sigma_n_sq)  (参考图像信息)
        # I_dist = log(1 + g^2 * sigma_ref_sq / sigma_n_sq)  (失真图像信息)
        # VIF = sum(I_dist) / sum(I_ref)

        g = sigma_ref_dist / (sigma_ref_sq + eps)
        # g = torch.clamp(g, min=0.0, max=1.0)
        sigma_n_sq = torch.clamp(sigma_dist_sq - (g * g) * sigma_ref_sq, min=eps)

        # 信息量计算
        num_term = torch.log1p((g * g) * sigma_ref_sq / (sigma_n_sq + eps))
        den_term = torch.log1p(sigma_ref_sq / (sigma_n_sq + eps))

        num_sum = num_sum + num_term.sum(dim=(1, 2, 3))
        den_sum = den_sum + den_term.sum(dim=(1, 2, 3))

        if s < levels - 1:
            cur_ref = F.avg_pool2d(cur_ref, kernel_size=2, stride=2)
            cur_dist = F.avg_pool2d(cur_dist, kernel_size=2, stride=2)

    vif = num_sum / (den_sum + eps)
    return vif


@contextmanager
def _eval_bn_use_batch_stats(module: nn.Module):
    bns: list[tuple[nn.BatchNorm2d, bool, float | None]] = []
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            bns.append((m, bool(m.training), m.momentum))
            m.train(True)
            m.momentum = 0.0
    try:
        yield
    finally:
        for m, training, momentum in bns:
            m.train(training)
            m.momentum = momentum

def calculate_metrics(model: ALOCC, data_loader: DataLoader, inner_class=0, verbose=False, eval_bn_batch_stats: bool = False):
    all_dis_fake, all_labels, all_ssim_scores, all_vif_scores, all_gmsd_scores = [], [], [], [], []
    
    model.G.eval()
    model.D.eval()
    ctx = _eval_bn_use_batch_stats(model) if bool(eval_bn_batch_stats) else nullcontext()
    with torch.inference_mode(), ctx:
        for data in data_loader:
            img, noisy_img, labels = data            
            # 模型前向传播
            gen_img = model.G(noisy_img)
            dis_fake = torch.sigmoid(model.D(gen_img)).squeeze()
            # 反归一化
            img = (img + 1.0) / 2.0
            gen_img = (gen_img + 1.0) / 2.0
            # 批量计算SSIM和VIF
            if img.shape[-1] == 28:
                ssim_scores = ssim(img, gen_img, data_range=1.0, reduction='none', downsample=False, kernel_size=7)
                vif_scores = _vif_ms_custom(img, gen_img)
            else:
                ssim_scores = ssim(img, gen_img, data_range=1.0, reduction='none')
                vif_scores = vif_p(img, gen_img, data_range=1.0, reduction='none', sigma_n_sq=10000)
            gmsd_scores = multi_scale_gmsd(img, gen_img, data_range=1.0, reduction='none')
            
            # 在GPU上收集结果，减少CPU传输
            all_dis_fake.append(dis_fake)
            all_labels.append(labels)
            all_ssim_scores.append(ssim_scores)
            all_vif_scores.append(vif_scores)
            all_gmsd_scores.append(gmsd_scores)
        # 一次性合并所有结果（仍在GPU上）
        dis_fake_tensor = torch.cat(all_dis_fake)
        labels_tensor = torch.cat(all_labels)
        ssim_tensor = torch.cat(all_ssim_scores)
        vif_tensor = torch.cat(all_vif_scores)
        all_gmsd_scores = torch.cat(all_gmsd_scores)
        # 在GPU上进行掩码操作
        # 计算指标（仍在GPU上）
        inlier_mask = (labels_tensor == inner_class)
        outlier_mask = ~inlier_mask
        
        mean_ssim_in = ssim_tensor[inlier_mask].mean().cpu().numpy()
        mean_vif_ic = vif_tensor[inlier_mask].mean().cpu().numpy()
        mean_gmsd_ic = all_gmsd_scores[inlier_mask].mean().cpu().numpy()
        
        mean_ssim_out = ssim_tensor[outlier_mask].mean().cpu().numpy()
        mean_vif_oc = vif_tensor[outlier_mask].mean().cpu().numpy()
        mean_gmsd_oc = all_gmsd_scores[outlier_mask].mean().cpu().numpy()
    # 转移到CPU进行scikit-learn计算
    y_score = dis_fake_tensor.cpu().numpy()
    y_true = (labels_tensor == inner_class).cpu().numpy()
    # 原有的ROC计算（这部分在CPU上）
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # 向量化F1计算
    y_pred = (y_score >= thresholds[:, None])
    tp, fp, fn = (y_true & y_pred).sum(1), (~y_true & y_pred).sum(1), (y_true & ~y_pred).sum(1)
    f1_scores = 2 * tp / (2 * tp + fp + fn + 1e-8)
    best_idx = np.argmax(f1_scores)
    f1 = f1_scores[best_idx]
    acc = accuracy_score(y_true, y_score >= thresholds[best_idx])
    eer = brentq(lambda x: 1. - x - np.interp(x, fpr, tpr), 0., 1.)
    auc = roc_auc_score(y_true, y_score)
    
    if verbose:
        print(
            f"Metrics -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
            f"SSIM-IC: {mean_ssim_in:.3f}, SSIM-OC: {mean_ssim_out:.3f}, "
            f"VIF-IC: {mean_vif_ic:.3f}, VIF-OC: {mean_vif_oc:.3f}, "
            f"GMSD-IC: {mean_gmsd_ic:.3f}, GMSD-OC: {mean_gmsd_oc:.3f}"
        )
    
    return f1, acc, eer, auc, mean_ssim_in, mean_ssim_out, mean_vif_ic, mean_vif_oc, mean_gmsd_ic, mean_gmsd_oc

def show_class_metrics(metrics):
    metrics = np.array(metrics)
    index = metrics[:, 0]
    
    # 打印各指标平均值
    mean_acc = float(np.mean(metrics[:, 1])) if metrics.size else 0.0
    mean_f1 = float(np.mean(metrics[:, 2])) if metrics.size else 0.0
    mean_eer = float(np.mean(metrics[:, 3])) if metrics.size else 0.0
    mean_auc = float(np.mean(metrics[:, 4])) if metrics.size else 0.0
    mean_ssim_in = float(np.mean(metrics[:, 5])) if metrics.shape[1] > 5 else 0.0
    mean_ssim_out = float(np.mean(metrics[:, 6])) if metrics.shape[1] > 6 else 0.0
    mean_vif_ic = float(np.mean(metrics[:, 7])) if metrics.shape[1] > 7 else 0.0
    mean_vif_oc = float(np.mean(metrics[:, 8])) if metrics.shape[1] > 8 else 0.0
    print(
        f"平均值 -> F1: {mean_f1:.3f}, Acc: {mean_acc:.3f}, EER: {mean_eer:.3f}, AUC: {mean_auc:.3f}, "
        f"SSIM-IC: {mean_ssim_in:.3f}, SSIM-OC: {mean_ssim_out:.3f}, "
        f"VIF-IC: {mean_vif_ic:.3f}, VIF-OC: {mean_vif_oc:.3f}"
    )
    
    # 绘图配置 - 减少子图数量，合并SSIM和PSNR
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('ALOCC Metrics over Epochs', fontsize=14)
    
    # 第一行：分类指标
    classification_cfg = [
        (1, 'g', 's', 'F1-score'),
        (2, 'b', 'o', 'Accuracy'),
        (3, 'r', '^', 'EER'),
        (4, 'm', 'D', 'AUC')
    ]
    
    # 第二行：合并的SSIM和PSNR
    reconstruction_cfg = [
        (5, 'c', 'v', 'SSIM-IC'),
        (6, 'y', 'p', 'SSIM-OC'),
        (7, 'k', '*', 'VIF-IC'),
        (8, 'g', 'x', 'VIF-OC')
    ]
    
    # 绘制分类指标（前4个）
    for i, (idx, color, marker, name) in enumerate(classification_cfg):
        ax = axs[0, i] if i < 3 else axs[1, 0]  # 第一行3个，第二行1个
        ax.plot(index, metrics[:, idx], f'{color}-{marker}', label=name)
        ax.set(xlabel='Index', ylabel=name, title=name)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(index)
        ax.grid(True)
        ax.legend()
    
    # 绘制合并的SSIM图
    ax_ssim = axs[1, 1]  # 第一行第3个位置放合并的SSIM
    for idx, color, marker, name in reconstruction_cfg[:2]:  # SSIM-in和SSIM-out
        ax_ssim.plot(index, metrics[:, idx], f'{color}-{marker}', label=name)
    ax_ssim.set(xlabel='Index', ylabel='SSIM Value', title='SSIM Comparison')
    ax_ssim.set_ylim(0, 1.0)
    ax_ssim.set_xticks(index)
    ax_ssim.grid(True)
    ax_ssim.legend()
    
    # 绘制合并的PSNR图
    ax_psnr = axs[1, 2]  # 第二行第2个位置放合并的PSNR
    for idx, color, marker, name in reconstruction_cfg[2:]:  # PSNR-in和PSNR-out
        ax_psnr.plot(index, metrics[:, idx], f'{color}-{marker}', label=name)
    ax_psnr.set(xlabel='Index', ylabel='PSNR Value', title='PSNR Comparison')
    ax_psnr.set_ylim(0, 1.0)
    ax_psnr.set_xticks(index)
    ax_psnr.grid(True)
    ax_psnr.legend()
    
    plt.tight_layout()
    plt.show()
