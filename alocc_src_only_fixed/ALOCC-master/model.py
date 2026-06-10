import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import piq
import matplotlib.pyplot as plt
from utils import timer, DEVICE
from torch.utils.data import DataLoader

def _weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if getattr(m, 'weight', None) is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, 'weight', None) is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if getattr(m, 'bias', None) is not None and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# 生成器

# ---------------------------------------------------------------------------
# CBAM — 来自 ECCV 2018，插入编码器每个 block 后以增强细粒度特征区分能力
# [CBAM] use_cbam=True 时生效，默认关闭保持向后兼容
# ---------------------------------------------------------------------------

class _CBAMChannelAttn(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(-2, -1), keepdim=True))
        mx = self.mlp(x.amax(dim=(-2, -1), keepdim=True))
        return torch.sigmoid(avg + mx)


class _CBAMSpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """轻量 CBAM：通道注意力 → 空间注意力，reduction=8 适配小通道数（32/64/128）。"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.ca = _CBAMChannelAttn(channels, reduction)
        self.sa = _CBAMSpatialAttn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling (GeM)."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=(-1, -2), keepdim=True)
        return x.pow(1.0 / self.p)


class CosinePrototypeClassifier(nn.Module):
    """Cosine classifier with a learnable prototype and (optional) learnable logit scale.

    Input: feature map [B, C, H, W]
    Output: logits [B, 1]
    """
    def __init__(self, in_channels: int, proj_channels: int, p: float = 3.0, scale: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.pool = GeM(p=p, eps=eps)
        # Prototype vector (normalized in forward)
        self.weight = nn.Parameter(torch.randn(proj_channels))
        # Logit scale; clamped in forward for stability
        self.logit_scale = nn.Parameter(torch.tensor(float(scale)))
        self.eps = eps

    def forward(self, feat_map: torch.Tensor, return_feat: bool = False):
        z = self.proj(feat_map)                   # [B, P, H, W]
        v = self.pool(z).flatten(1)               # [B, P]
        v = F.normalize(v, dim=1, eps=self.eps)   # unit feature
        w = F.normalize(self.weight, dim=0, eps=self.eps)  # unit prototype
        cos = (v * w).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # [B, 1]
        scale = self.logit_scale.clamp(1.0, 50.0)
        logits = scale * cos
        if return_feat:
            return logits, v, cos
        return logits

class LowRankNoisyBottleneck(nn.Module):  # [S1-BOT] low-rank 1x1 sandwich + optional noise
    def __init__(self, channels: int, rank: int = 0, dropout: float = 0.0, noise_type: str = "dropout"):
        super().__init__()
        self.rank = int(rank)
        self.dropout_p = float(dropout)
        self.noise_type = str(noise_type)
        if self.rank > 0:
            self.down = nn.Conv2d(channels, self.rank, kernel_size=1, stride=1, padding=0, bias=False)
            self.up   = nn.Conv2d(self.rank, channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.down = None
            self.up = None
        if self.dropout_p > 0.0 and self.noise_type == "dropout":
            self.drop = nn.Dropout2d(p=self.dropout_p)
        else:
            self.drop = None

    def forward(self, x):
        if self.down is not None:
            x = self.up(self.down(x))
        if self.drop is not None:
            x = self.drop(x)
        elif self.dropout_p > 0.0 and self.noise_type == "gaussian" and self.training:
            x = x + torch.randn_like(x) * self.dropout_p
        return x


class Generator(nn.Module):
    def __init__(self, c_dim, gf_dim, df_dim, in_h, in_w=None, out_h=None, out_w=None, classify=False,
                 bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout",
                 use_cbam: bool = False):  # [CBAM]
        super(Generator, self).__init__()
        in_h1 = (in_h - 5 + 2 * 2) // 2 + 1
        in_h2 = (in_h1 - 5 + 2 * 2) // 2 + 1
        in_h3 = (in_h2 - 5 + 2 * 2) // 2 + 1
        # 编码器 — use_cbam=True 时在每个 block 后插入 CBAM
        enc_layers = [
            nn.Conv2d(c_dim, df_dim*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_cbam:
            enc_layers.append(CBAM(df_dim*2))
        enc_layers += [
            nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_cbam:
            enc_layers.append(CBAM(df_dim*4))
        enc_layers += [
            nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_cbam:
            enc_layers.append(CBAM(df_dim*8))
        self.encoder = nn.Sequential(*enc_layers)
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(df_dim*8, gf_dim*2, 5, 2, 2, output_padding=(in_h2+1 - in_h3*2)%2, bias=False),
            nn.BatchNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(gf_dim*2, gf_dim, 5, 2, 2, output_padding=(in_h1+1 - in_h2*2)%2, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(gf_dim, c_dim, 5, 2, 2, output_padding=(in_h+1 - in_h1*2)%2, bias=True),
            nn.Tanh()
        )
        # 分类器
        self.classify = classify
        self.classifier = CosinePrototypeClassifier(
            in_channels=df_dim * 8,
            proj_channels=df_dim * 4,
            p=3.0,
            scale=10.0,
            eps=1e-6,
        )
        # hook出卷积层的特征图
        self.target_layer_handle = self.decoder[-2].register_forward_hook(self._hook_conv)
        # [S1-BOT] bottleneck at encoder/decoder seam; Identity when all knobs off -> bitwise parity with baseline
        if int(bottleneck_rank) > 0 or float(bottleneck_dropout) > 0.0:
            self.bottleneck = LowRankNoisyBottleneck(df_dim * 8, rank=bottleneck_rank,
                                                    dropout=bottleneck_dropout, noise_type=bottleneck_noise_type)
        else:
            self.bottleneck = nn.Identity()
        self.apply(_weights_init_normal)

    def _hook_conv(self, module, input, output):
        self.feature_maps = output

    def forward_classify(self, x, detach_encoder: bool = False, return_feat: bool = False):
        feat = self.encoder(x)
        if detach_encoder:
            feat = feat.detach()
        if return_feat:
            return self.classifier(feat, return_feat=True)
        return self.classifier(feat)

    def forward(self, x, classify=False):
        enc_output = self.encoder(x)
        enc_output = self.bottleneck(enc_output)  # [S1-BOT]
        if classify:
                                self.classifier_output = self.classifier(enc_output)
        return self.decoder(enc_output)

import itertools

# 判别器
class Discriminator(nn.Module):
    def __init__(self, c_dim, df_dim, in_h, in_w=None):
        super(Discriminator, self).__init__()
        if in_w is None:
            in_w = in_h
        # 计算经过 4 个步长为 2 的卷积后的特征图大小
        # 每个卷积层步长为 2，所以总共缩小 2^4 = 16 倍
        h_after_conv = math.ceil(in_h / 16)
        w_after_conv = math.ceil(in_w / 16)
        self.logits = nn.Sequential(
            nn.Conv2d(c_dim, df_dim, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim, df_dim*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(df_dim * 8 * h_after_conv * w_after_conv, 1, bias=True)
        )
        self.apply(_weights_init_normal)

    def forward(self, x):
        return self.logits(x)
# 基础模型
class ALOCC(nn.Module):
    def __init__(
        self,
        c_dim=1,
        gf_dim=16,
        df_dim=16,
        in_h=None,
        in_w=None,
        out_h=None,
        out_w=None,
        lr=0.002,
        lr_d=None,  # [TTUR] Discriminator learning rate (if None, use lr)
        lr_g=None,  # [TTUR] Generator learning rate (if None, use lr)
        classify=False,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        bottleneck_rank: int = 0,  # [S1-BOT]
        bottleneck_dropout: float = 0.0,  # [S1-BOT]
        bottleneck_noise_type: str = "dropout",  # [S1-BOT]
        tf_verbatim_rmsprop: bool = False,  # [BASELINE-A][D2-B][ADR-007] opt-in TF1.15 RMSprop defaults
        use_cbam: bool = False,  # [CBAM] 编码器注意力增强
    ):
        super(ALOCC, self).__init__()
        self.device = DEVICE
        self.label_smoothing = float(label_smoothing)
        self.tf_verbatim_rmsprop = bool(tf_verbatim_rmsprop)  # [BASELINE-A][D2-B]
        # 损失函数
        self.criterion_bce = nn.BCEWithLogitsLoss().to(DEVICE, non_blocking=True)
        # Paper-aligned refinement objective: denoise noisy in-class samples back to clean targets.
        self.criterion_refinement = nn.MSELoss().to(DEVICE, non_blocking=True)
        # 生成器+判别器
        self.G = Generator(c_dim, gf_dim, df_dim, in_h, classify=classify,  # [S1-BOT]
                           bottleneck_rank=bottleneck_rank,
                           bottleneck_dropout=bottleneck_dropout,
                           bottleneck_noise_type=bottleneck_noise_type,
                           use_cbam=bool(use_cbam)).to(DEVICE, non_blocking=True)  # [CBAM]
        self.D = Discriminator(c_dim, df_dim, in_h=in_h).to(DEVICE, non_blocking=True)
        # 优化器 [TTUR] 支持分离的 D/G 学习率
        _rms_alpha = 0.9 if self.tf_verbatim_rmsprop else 0.99
        _rms_eps   = 1e-10 if self.tf_verbatim_rmsprop else 1e-8
        # 如果未指定 lr_d/lr_g，则使用统一的 lr
        _lr_d = float(lr_d) if lr_d is not None else float(lr)
        _lr_g = float(lr_g) if lr_g is not None else float(lr)
        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=_lr_d, alpha=_rms_alpha, eps=_rms_eps, weight_decay=weight_decay, foreach=True)
        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=_lr_g, alpha=_rms_alpha, eps=_rms_eps, weight_decay=weight_decay, foreach=True)

    def _save_checkpoint(self, path):
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict()
        }, path, _use_new_zipfile_serialization=False)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G.eval(); self.D.eval()

    def _refinement_loss(self, fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
        return self.criterion_refinement(fake_imgs, real_imgs)

    def _train(
        self,
        data_loader: DataLoader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
    ):
        # 进度条
        progress_bar = tqdm(total=epoch, desc="Train", leave=False)
        # 训练
        self.G.train(); self.D.train()
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count, epoch_d_loss_sum, epoch_g_loss_sum, epoch_g_r_total, epoch_g_gen_total = 0, .0, .0, .0, .0
                for real_imgs, noisy_imgs, *_ in data_loader:
                    batch_size = real_imgs.size(0)
                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    # 标签
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)
                    # 准备数据
                    # -----------------
                    # 训练判别器 D
                    # -----------------
                    self.D.zero_grad(set_to_none=True)
                    # D on real
                    real_logits = self.D(real_imgs)
                    d_loss_real = self.criterion_bce(real_logits, real_label)
                    # D on fake
                    with torch.no_grad():
                        fake_imgs = self.G(noisy_imgs)
                    fake_logits = self.D(fake_imgs)  # ✅ 无需 detach
                    d_loss_fake = self.criterion_bce(fake_logits, fake_label)
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.optim_D.step()
                    # -----------------
                    # 训练生成器 G (2次)
                    # -----------------
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    for _ in range(2):
                        self.G.zero_grad(set_to_none=True)
                        fake_imgs_new = self.G(noisy_imgs)
                        fake_logits_new = self.D(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)
                        # [BASELINE-A][D1-B][TF1.15 models.py L131] verbatim refinement:
                        #   tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G, labels=self.z)
                        # i.e. BCE-with-logits between G(noisy) and the noisy input itself.
                        g_loss_r = F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r
                        g_loss.backward()
                        self.optim_G.step()
                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()
                    epoch_d_loss_sum += d_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total
                    batch_count += 1
                    f.write(f"iter, d_loss={d_loss.item():.3f}, g_gen={g_gen_total / 2:.3f}, g_r={g_r_total / 2:.3f}\n")
                # 更新进度
                cnt = data_loader.__len__() * 2
                progress_bar.update(1)
                progress_bar.set_postfix(d_loss=epoch_d_loss_sum / cnt, g_gen=epoch_g_gen_total / cnt, g_r=epoch_g_r_total / cnt)
                # 保存检查点
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir,f"{i+1}.pth"))
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)
# ---------------------------------------------------------------------------
# AGC helpers — used by ALOCC_LOSS._train
# ---------------------------------------------------------------------------

def _total_grad_norm(parameters) -> float:
    """Compute the global L2 norm of all parameter gradients."""
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += p.grad.detach().norm(2).item() ** 2
    return total_sq ** 0.5


def _agc_update_and_clip(parameters, mu, var, grad_norm: float,
                         
                          decay: float, k: float, min_clip: float):
    """Update EMA stats and apply adaptive gradient clipping.

    First call (mu is None): seeds EMA with current grad_norm.
    Returns (new_mu, new_var, clip_threshold).
    """
    if mu is None:
        mu, var = grad_norm, 0.0
    else:
        delta = grad_norm - mu
        mu = mu + (1.0 - decay) * delta
        var = decay * (var + (1.0 - decay) * delta * delta)
    sigma = var ** 0.5
    threshold = max(float(min_clip), mu + k * sigma)
    if grad_norm > threshold:
        torch.nn.utils.clip_grad_norm_(parameters, threshold)
    return mu, var, threshold


class ALOCC_LOSS(ALOCC):
    """标准ALOCC_LOSS：单D架构，D见过外类"""
    def _train(
        self,
        data_loader: DataLoader,
        outclass_loader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        d_outclass_loss_scale=0.1,
        outclass_every: int = 20,
        g_steps: int = 2,
        d_steps: int = 1,  # [TTUR] Discriminator training steps per batch
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
        agc_ema_decay: float = 0.99,   # [AGC] EMA 衰减系数
        agc_k: float = 3.0,            # [AGC] 阈值 = mu + k * sigma
        agc_min_clip: float = 1.0,     # [AGC] 裁剪阈值下限，防止初期过激裁剪
    ):
        # 进度条
        progress_bar = tqdm(total=epoch, desc="Train", leave=False)
        # 训练
        self.G.train(); self.D.train()
        os.makedirs(checkpoint_dir, exist_ok=True)

        outclass_cycler = itertools.cycle(outclass_loader)

        # [AGC] 独立维护 D 和 G 的梯度范数 EMA 统计 (mu, var)
        # 初始为 None，首个 batch 后用实际梯度范数播种
        _d_mu, _d_var = None, 0.0
        _g_mu, _g_var = None, 0.0

        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count, epoch_d_loss_sum, epoch_g_loss_sum, epoch_g_r_total, epoch_g_gen_total = 0, .0, .0, .0, .0

                outclass_iter = iter(outclass_cycler)

                outclass_every_k = max(1, int(outclass_every))
                g_steps_k = max(1, int(g_steps))
                for in_step, (real_imgs, noisy_imgs, *_) in enumerate(data_loader, start=1): # 内类数据迭代
                    batch_size = real_imgs.size(0)
                    if real_imgs.device != DEVICE:
                        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                    if noisy_imgs.device != DEVICE:
                        noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)
                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    # 标签
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)
                    # -----------------
                    # 训练判别器 D (多步训练) [TTUR]
                    # -----------------
                    d_steps_k = max(1, int(d_steps))
                    d_clip_thresh = float(agc_min_clip)  # 记录最后一步的阈值用于日志
                    for _ in range(d_steps_k):
                        self.D.zero_grad(set_to_none=True)
                        # D on real
                        real_logits = self.D(real_imgs)
                        d_loss_real = self.criterion_bce(real_logits, real_label)
                        # D on fake
                        fake_imgs = self.G(noisy_imgs).detach()
                        fake_logits = self.D(fake_imgs)
                        d_loss_fake = self.criterion_bce(fake_logits, fake_label)

                        d_loss = d_loss_fake + d_loss_real
                        if (in_step % outclass_every_k) == 0:
                            outclass_imgs, outclass_noisy_imgs, *_ = next(outclass_iter)
                            if outclass_imgs.device != DEVICE:
                                outclass_imgs = outclass_imgs.to(DEVICE, non_blocking=True)
                            if outclass_noisy_imgs.device != DEVICE:
                                outclass_noisy_imgs = outclass_noisy_imgs.to(DEVICE, non_blocking=True)
                            outclass_fake_label = torch.full((outclass_imgs.size(0), 1), eps, device=DEVICE)

                            outclass_logits = self.D(outclass_imgs)
                            d_loss_outclass = self.criterion_bce(outclass_logits, outclass_fake_label)
                            d_loss = d_loss + d_loss_outclass * d_outclass_loss_scale
                        d_loss.backward()
                        # [AGC] 计算 D 梯度范数，更新 EMA，自适应裁剪
                        _d_grad_norm = _total_grad_norm(self.D.parameters())
                        _d_mu, _d_var, d_clip_thresh = _agc_update_and_clip(
                            self.D.parameters(), _d_mu, _d_var,
                            _d_grad_norm, agc_ema_decay, agc_k, agc_min_clip
                        )
                        self.optim_D.step()
                    # -----------------
                    # 训练生成器 G — 纯内类重建
                    # -----------------
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    g_clip_thresh = float(agc_min_clip)  # 记录最后一步的阈值用于日志
                    for _ in range(g_steps_k):
                        self.G.zero_grad(set_to_none=True)
                        fake_imgs_new = self.G(noisy_imgs)
                        fake_logits_new = self.D(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)
                        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r
                        g_loss.backward()
                        # [AGC] 计算 G 梯度范数，更新 EMA，自适应裁剪
                        _g_grad_norm = _total_grad_norm(self.G.parameters())
                        _g_mu, _g_var, g_clip_thresh = _agc_update_and_clip(
                            self.G.parameters(), _g_mu, _g_var,
                            _g_grad_norm, agc_ema_decay, agc_k, agc_min_clip
                        )
                        self.optim_G.step()
                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()
                    epoch_d_loss_sum += d_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total
                    f.write(
                        f"iter, d_loss={d_loss.item():.3f}, g_gen={g_gen_total / g_steps_k:.3f}, "
                        f"g_r={g_r_total / g_steps_k:.3f}, "
                        f"agc_d_thresh={d_clip_thresh:.3f}, agc_g_thresh={g_clip_thresh:.3f}\n"
                    )
                # 更新进度
                cnt = data_loader.__len__() * g_steps_k
                progress_bar.update(1)
                progress_bar.set_postfix(d_loss=epoch_d_loss_sum / cnt, g_gen=epoch_g_gen_total / cnt, g_r=epoch_g_r_total / cnt)
                # 保存检查点
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir,f"{i+1}.pth"))
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)
        return int(epoch)


class ALOCC_LOSS_CLIP(ALOCC_LOSS):
    """在 ALOCC_LOSS 基础上添加梯度裁剪，防止D的梯度爆炸导致训练崩溃"""
    def _train(
        self,
        data_loader: DataLoader,
        outclass_loader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        d_outclass_loss_scale=0.1,
        outclass_every: int = 20,
        g_steps: int = 2,
        d_steps: int = 1,
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
        clip_grad_norm: float = 5.0,
    ):
        progress_bar = tqdm(total=epoch, desc="Train (GradClip)", leave=False)
        self.G.train(); self.D.train()
        os.makedirs(checkpoint_dir, exist_ok=True)
        outclass_cycler = itertools.cycle(outclass_loader)
        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count, epoch_d_loss_sum, epoch_g_loss_sum, epoch_g_r_total, epoch_g_gen_total = 0, .0, .0, .0, .0
                outclass_iter = iter(outclass_cycler)
                outclass_every_k = max(1, int(outclass_every))
                g_steps_k = max(1, int(g_steps))
                d_steps_k = max(1, int(d_steps))
                for in_step, (real_imgs, noisy_imgs, *_) in enumerate(data_loader, start=1):
                    batch_size = real_imgs.size(0)
                    if real_imgs.device != DEVICE:
                        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                    if noisy_imgs.device != DEVICE:
                        noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)
                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)
                    # --- D 训练（带梯度裁剪） ---
                    for _ in range(d_steps_k):
                        self.D.zero_grad(set_to_none=True)
                        real_logits = self.D(real_imgs)
                        d_loss_real = self.criterion_bce(real_logits, real_label)
                        fake_imgs = self.G(noisy_imgs).detach()
                        fake_logits = self.D(fake_imgs)
                        d_loss_fake = self.criterion_bce(fake_logits, fake_label)
                        d_loss = d_loss_fake + d_loss_real
                        if (in_step % outclass_every_k) == 0:
                            outclass_imgs, outclass_noisy_imgs, *_ = next(outclass_iter)
                            if outclass_imgs.device != DEVICE:
                                outclass_imgs = outclass_imgs.to(DEVICE, non_blocking=True)
                            if outclass_noisy_imgs.device != DEVICE:
                                outclass_noisy_imgs = outclass_noisy_imgs.to(DEVICE, non_blocking=True)
                            outclass_fake_label = torch.full((outclass_imgs.size(0), 1), eps, device=DEVICE)
                            outclass_logits = self.D(outclass_imgs)
                            d_loss_outclass = self.criterion_bce(outclass_logits, outclass_fake_label)
                            d_loss = d_loss + d_loss_outclass * d_outclass_loss_scale
                        d_loss.backward()
                        # 梯度裁剪：防止D梯度爆炸
                        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=clip_grad_norm)
                        self.optim_D.step()
                    # --- G 训练 ---
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    for _ in range(g_steps_k):
                        self.G.zero_grad(set_to_none=True)
                        fake_imgs_new = self.G(noisy_imgs)
                        fake_logits_new = self.D(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)
                        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r
                        g_loss.backward()
                        self.optim_G.step()
                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()
                    epoch_d_loss_sum += d_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total
                    f.write(f"iter, d_loss={d_loss.item():.3f}, g_gen={g_gen_total / g_steps_k:.3f}, g_r={g_r_total / g_steps_k:.3f}\n")
                cnt = data_loader.__len__() * g_steps_k
                progress_bar.update(1)
                progress_bar.set_postfix(d_loss=epoch_d_loss_sum / cnt, g_gen=epoch_g_gen_total / cnt, g_r=epoch_g_r_total / cnt)
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)
        return int(epoch)


class ALOCC_LOSS_BASELINE_REF(ALOCC_LOSS):
    """验证消融结论：在ALOCC_LOSS基础上把重建损失改为基线的 noisy-target BCE
    
    基线 (ALOCC._train):
        g_loss_r = F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)
    
    ALOCC_LOSS._train:
        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs) = MSE(G(noisy), clean)
    
    本类:
        g_loss_r = F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)
    
    预期结果: AUC 应恢复到基线水平，验证消融结论
    """
    def _train(
        self,
        data_loader: DataLoader,
        outclass_loader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        d_outclass_loss_scale=0.1,
        outclass_every: int = 20,
        g_steps: int = 2,
        d_steps: int = 1,
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
    ):
        progress_bar = tqdm(total=epoch, desc="Train (Baseline-Ref)", leave=False)
        self.G.train(); self.D.train()
        os.makedirs(checkpoint_dir, exist_ok=True)
        outclass_cycler = itertools.cycle(outclass_loader)
        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count, epoch_d_loss_sum, epoch_g_loss_sum, epoch_g_r_total, epoch_g_gen_total = 0, .0, .0, .0, .0
                outclass_iter = iter(outclass_cycler)
                outclass_every_k = max(1, int(outclass_every))
                g_steps_k = max(1, int(g_steps))
                d_steps_k = max(1, int(d_steps))
                for in_step, (real_imgs, noisy_imgs, *_) in enumerate(data_loader, start=1):
                    batch_size = real_imgs.size(0)
                    if real_imgs.device != DEVICE:
                        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                    if noisy_imgs.device != DEVICE:
                        noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)
                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)
                    # --- D 训练 ---
                    for _ in range(d_steps_k):
                        self.D.zero_grad(set_to_none=True)
                        real_logits = self.D(real_imgs)
                        d_loss_real = self.criterion_bce(real_logits, real_label)
                        fake_imgs = self.G(noisy_imgs).detach()
                        fake_logits = self.D(fake_imgs)
                        d_loss_fake = self.criterion_bce(fake_logits, fake_label)
                        d_loss = d_loss_fake + d_loss_real
                        if (in_step % outclass_every_k) == 0:
                            outclass_imgs, outclass_noisy_imgs, *_ = next(outclass_iter)
                            if outclass_imgs.device != DEVICE:
                                outclass_imgs = outclass_imgs.to(DEVICE, non_blocking=True)
                            if outclass_noisy_imgs.device != DEVICE:
                                outclass_noisy_imgs = outclass_noisy_imgs.to(DEVICE, non_blocking=True)
                            outclass_fake_label = torch.full((outclass_imgs.size(0), 1), eps, device=DEVICE)
                            outclass_logits = self.D(outclass_imgs)
                            d_loss_outclass = self.criterion_bce(outclass_logits, outclass_fake_label)
                            d_loss = d_loss + d_loss_outclass * d_outclass_loss_scale
                        d_loss.backward()
                        self.optim_D.step()
                    # --- G 训练 ---
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    for _ in range(g_steps_k):
                        self.G.zero_grad(set_to_none=True)
                        fake_imgs_new = self.G(noisy_imgs)
                        fake_logits_new = self.D(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)
                        g_loss_r = F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r
                        g_loss.backward()
                        self.optim_G.step()
                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()
                    epoch_d_loss_sum += d_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total
                    f.write(f"iter, d_loss={d_loss.item():.3f}, g_gen={g_gen_total / g_steps_k:.3f}, g_r={g_r_total / g_steps_k:.3f}\n")
                cnt = data_loader.__len__() * g_steps_k
                progress_bar.update(1)
                progress_bar.set_postfix(d_loss=epoch_d_loss_sum / cnt, g_gen=epoch_g_gen_total / cnt, g_r=epoch_g_r_total / cnt)
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)
        return int(epoch)

class ALOCC_LOSS_CLS(ALOCC):
    def _train(
        self,
        data_loader: DataLoader,
        outclass_loader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        d_outclass_loss_scale=0.1,
        outclass_every: int = 20,
        g_steps: int = 2,
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
    ):
        self.optim_CLS = torch.optim.RMSprop(list(self.G.encoder.parameters()) + list(self.G.classifier.parameters()), lr=0.00001, foreach=True)
        self.criterion_cls = nn.BCEWithLogitsLoss().to(DEVICE, non_blocking=True)
        # 进度条
        progress_bar = tqdm(total=epoch, desc="Train", leave=False)
        # 训练
        self.G.train(); self.D.train()
        os.makedirs(checkpoint_dir, exist_ok=True)

        outclass_cycler = itertools.cycle(outclass_loader)

        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count, epoch_d_loss_sum, epoch_g_loss_sum, epoch_g_r_total, epoch_g_gen_total = 0, .0, .0, .0, .0

                outclass_iter = iter(outclass_cycler)

                outclass_every_k = max(1, int(outclass_every))
                g_steps_k = max(1, int(g_steps))
                for in_step, (real_imgs, noisy_imgs, *_) in enumerate(data_loader, start=1): # 内类数据迭代
                    batch_size = real_imgs.size(0)
                    if real_imgs.device != DEVICE:
                        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                    if noisy_imgs.device != DEVICE:
                        noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)
                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    # 标签
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)
                    # -----------------
                    # 训练判别器 D
                    # -----------------
                    self.D.zero_grad(set_to_none=True)
                    # D on real
                    real_logits = self.D(real_imgs)
                    d_loss_real = self.criterion_bce(real_logits, real_label)
                    # D on fake
                    fake_imgs = self.G(noisy_imgs).detach()
                    fake_logits = self.D(fake_imgs)  # ✅ 无需 detach
                    d_loss_fake = self.criterion_bce(fake_logits, fake_label)

                    d_loss = d_loss_fake + d_loss_real
                    outclass_imgs = None
                    outclass_fake_label = None
                    if (in_step % outclass_every_k) == 0:
                        outclass_imgs, outclass_noisy_imgs, *_ = next(outclass_iter)
                        if outclass_imgs.device != DEVICE:
                            outclass_imgs = outclass_imgs.to(DEVICE, non_blocking=True)
                        if outclass_noisy_imgs.device != DEVICE:
                            outclass_noisy_imgs = outclass_noisy_imgs.to(DEVICE, non_blocking=True)
                        outclass_fake_label = torch.full((outclass_imgs.size(0), 1), eps, device=DEVICE)

                        outclass_logits = self.D(outclass_imgs)

                        d_loss_outclass = self.criterion_bce(outclass_logits, outclass_fake_label)
                        d_loss = d_loss + d_loss_outclass * d_outclass_loss_scale
                    d_loss.backward()
                    self.optim_D.step()
                    # -----------------
                    # 训练生成器 G (2次) — 纯内类重建
                    # -----------------
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    for _ in range(g_steps_k):
                        self.G.zero_grad(set_to_none=True)
                        fake_imgs_new = self.G(noisy_imgs)
                        fake_logits_new = self.D(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)
                        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r
                        g_loss.backward()
                        self.optim_G.step()
                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()

                    # -----------------
                    # 训练分类分支（Cosine-Prototype 方案：更强的特征约束）
                    # -----------------
                    cls_alpha = 0.15        # BCE 分类损失权重
                    proto_alpha = 0.05      # in-class 拉近 prototype
                    margin_alpha = 0.05     # 负样本推离间隔
                    noisy_neg_alpha = 0.5   # noisy 作为“弱负样本”的权重
                    cos_margin = 0.2        # 负样本 cosine 上限（越小越严格）

                    self.optim_CLS.zero_grad(set_to_none=True)

                    in_logits, _, in_cos = self.G.forward_classify(real_imgs, detach_encoder=False, return_feat=True)
                    noisy_logits, _, noisy_cos = self.G.forward_classify(noisy_imgs, detach_encoder=False, return_feat=True)
                    if outclass_imgs is not None:
                        out_logits, _, out_cos = self.G.forward_classify(outclass_imgs, detach_encoder=False, return_feat=True)

                    # BCE：in=1, out/noisy=0（用 noisy 做 hard-negative 的补充）
                    if outclass_imgs is not None:
                        cls_loss = (
                            self.criterion_cls(in_logits, real_label) +
                            self.criterion_cls(out_logits, outclass_fake_label) +
                            noisy_neg_alpha * self.criterion_cls(noisy_logits, fake_label)
                        ) / (2.0 + noisy_neg_alpha)
                    else:
                        cls_loss = (
                            self.criterion_cls(in_logits, real_label) +
                            noisy_neg_alpha * self.criterion_cls(noisy_logits, fake_label)
                        ) / (1.0 + noisy_neg_alpha)

                    # Prototype 吸引：cos 越接近 1 越好
                    proto_loss = (1.0 - in_cos).mean()

                    # Margin 推离：负样本 cos 必须 <= cos_margin
                    if outclass_imgs is not None:
                        margin_loss = F.relu(out_cos - cos_margin).mean() + noisy_neg_alpha * F.relu(noisy_cos - cos_margin).mean()
                    else:
                        margin_loss = noisy_neg_alpha * F.relu(noisy_cos - cos_margin).mean()

                    g_cls_loss = cls_alpha * cls_loss + proto_alpha * proto_loss + margin_alpha * margin_loss
                    g_cls_loss.backward()
                    self.optim_CLS.step()

                    epoch_d_loss_sum += d_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total

                    f.write(f"iter, d_loss={d_loss.item():.3f}, g_gen={g_gen_total / g_steps_k:.3f}, g_r={g_r_total / g_steps_k:.3f}\n")
                # 更新进度
                cnt = data_loader.__len__() * g_steps_k
                progress_bar.update(1)
                progress_bar.set_postfix(d_loss=epoch_d_loss_sum / cnt, g_gen=epoch_g_gen_total / cnt, g_r=epoch_g_r_total / cnt)
                # 保存检查点
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir,f"{i+1}.pth"))
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)
        return int(epoch)

class ALOCC_LOSS_DUAL_D(ALOCC):
    """
    双判别器架构：解决外类训练对G的过度约束问题

    核心思想：
    - D_main: 见过外类，用于异常检测（测试阶段）
    - D_for_G: 不见过外类，用于训练G（提供温和的对抗信号）

    优势：
    - G不被过于严格的D约束，保持泛化能力
    - D_main仍然学习严格的判别边界，用于异常检测
    - 避免G过拟合训练数据
    """

    def __init__(
        self,
        c_dim=1,
        gf_dim=16,
        df_dim=16,
        in_h=None,
        in_w=None,
        out_h=None,
        out_w=None,
        lr=0.002,
        lr_d=None,
        lr_g=None,
        classify=False,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        bottleneck_rank: int = 0,
        bottleneck_dropout: float = 0.0,
        bottleneck_noise_type: str = "dropout",
        tf_verbatim_rmsprop: bool = False,
        sync_d_every: int = 0,  # 0表示不同步，>0表示每N个epoch同步一次
    ):
        super().__init__(
            c_dim=c_dim, gf_dim=gf_dim, df_dim=df_dim,
            in_h=in_h, in_w=in_w, out_h=out_h, out_w=out_w,
            lr=lr, lr_d=lr_d, lr_g=lr_g, classify=classify,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            bottleneck_rank=bottleneck_rank,
            bottleneck_dropout=bottleneck_dropout,
            bottleneck_noise_type=bottleneck_noise_type,
            tf_verbatim_rmsprop=tf_verbatim_rmsprop
        )

        # 创建第二个判别器D_for_G（不见外类）
        self.D_for_G = Discriminator(c_dim, df_dim, in_h=in_h).to(DEVICE, non_blocking=True)
        self.D_for_G.apply(_weights_init_normal)

        # D_for_G的优化器
        _rms_alpha = 0.9 if self.tf_verbatim_rmsprop else 0.99
        _rms_eps = 1e-10 if self.tf_verbatim_rmsprop else 1e-8
        _lr_d = float(lr_d) if lr_d is not None else float(lr)
        self.optim_D_for_G = torch.optim.RMSprop(
            self.D_for_G.parameters(),
            lr=_lr_d,
            alpha=_rms_alpha,
            eps=_rms_eps,
            weight_decay=weight_decay,
            foreach=True
        )

        self.sync_d_every = sync_d_every

    def _save_checkpoint(self, path):
        """保存包含两个D的checkpoint"""
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'D_for_G': self.D_for_G.state_dict()
        }, path, _use_new_zipfile_serialization=False)

    def _load_checkpoint(self, path):
        """加载包含两个D的checkpoint"""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        if 'D_for_G' in checkpoint:
            self.D_for_G.load_state_dict(checkpoint['D_for_G'])
        self.G.eval()
        self.D.eval()
        self.D_for_G.eval()

    def _train(
        self,
        data_loader: DataLoader,
        outclass_loader,
        epoch,
        checkpoint_dir,
        step=20,
        r_alpha=0.2,
        d_outclass_loss_scale=0.1,
        outclass_every: int = 20,
        g_steps: int = 2,
        d_steps: int = 1,
        stop_recon_threshold: float | None = None,
        stop_min_epoch: int = 1,
    ):
        """
        双D训练流程：
        1. D_main训练：见过外类，学习严格的判别边界
        2. D_for_G训练：不见外类，提供温和的对抗信号
        3. G训练：使用D_for_G的梯度，避免过度约束
        """
        progress_bar = tqdm(total=epoch, desc="Train (Dual-D)", leave=False)
        self.G.train()
        self.D.train()
        self.D_for_G.train()
        os.makedirs(checkpoint_dir, exist_ok=True)

        outclass_cycler = itertools.cycle(outclass_loader)

        with open(os.path.join(checkpoint_dir, 'debug.log'), 'w') as f:
            for i in range(epoch):
                batch_count = 0
                epoch_d_main_loss_sum = 0.0
                epoch_d_for_g_loss_sum = 0.0
                epoch_g_loss_sum = 0.0
                epoch_g_r_total = 0.0
                epoch_g_gen_total = 0.0

                outclass_iter = iter(outclass_cycler)
                outclass_every_k = max(1, int(outclass_every))
                g_steps_k = max(1, int(g_steps))
                d_steps_k = max(1, int(d_steps))

                for in_step, (real_imgs, noisy_imgs, *_) in enumerate(data_loader, start=1):
                    batch_size = real_imgs.size(0)
                    if real_imgs.device != DEVICE:
                        real_imgs = real_imgs.to(DEVICE, non_blocking=True)
                    if noisy_imgs.device != DEVICE:
                        noisy_imgs = noisy_imgs.to(DEVICE, non_blocking=True)

                    eps = max(0.0, min(0.5, float(self.label_smoothing)))
                    real_label = torch.full((batch_size, 1), 1.0 - eps, device=DEVICE)
                    fake_label = torch.full((batch_size, 1), eps, device=DEVICE)

                    # ==========================================
                    # 1. 训练D_main（见过外类）
                    # ==========================================
                    for _ in range(d_steps_k):
                        self.D.zero_grad(set_to_none=True)

                        # D_main on real
                        real_logits = self.D(real_imgs)
                        d_loss_real = self.criterion_bce(real_logits, real_label)

                        # D_main on fake
                        with torch.no_grad():
                            fake_imgs = self.G(noisy_imgs)
                        fake_logits = self.D(fake_imgs)
                        d_loss_fake = self.criterion_bce(fake_logits, fake_label)

                        d_main_loss = d_loss_real + d_loss_fake

                        # D_main on outclass（关键：只有D_main见外类）
                        if (in_step % outclass_every_k) == 0:
                            outclass_imgs, outclass_noisy_imgs, *_ = next(outclass_iter)
                            if outclass_imgs.device != DEVICE:
                                outclass_imgs = outclass_imgs.to(DEVICE, non_blocking=True)
                            if outclass_noisy_imgs.device != DEVICE:
                                outclass_noisy_imgs = outclass_noisy_imgs.to(DEVICE, non_blocking=True)
                            outclass_fake_label = torch.full((outclass_imgs.size(0), 1), eps, device=DEVICE)

                            outclass_logits = self.D(outclass_imgs)
                            d_loss_outclass = self.criterion_bce(outclass_logits, outclass_fake_label)
                            d_main_loss = d_main_loss + d_loss_outclass * d_outclass_loss_scale

                        d_main_loss.backward()
                        self.optim_D.step()

                    # ==========================================
                    # 2. 训练D_for_G（不见外类）
                    # ==========================================
                    for _ in range(d_steps_k):
                        self.D_for_G.zero_grad(set_to_none=True)

                        # D_for_G on real
                        real_logits_for_g = self.D_for_G(real_imgs)
                        d_for_g_loss_real = self.criterion_bce(real_logits_for_g, real_label)

                        # D_for_G on fake
                        with torch.no_grad():
                            fake_imgs_for_d = self.G(noisy_imgs)
                        fake_logits_for_g = self.D_for_G(fake_imgs_for_d)
                        d_for_g_loss_fake = self.criterion_bce(fake_logits_for_g, fake_label)

                        # 关键：D_for_G不见外类，只有两项损失
                        d_for_g_loss = d_for_g_loss_real + d_for_g_loss_fake

                        d_for_g_loss.backward()
                        self.optim_D_for_G.step()

                    # ==========================================
                    # 3. 训练G（使用D_for_G的梯度）
                    # ==========================================
                    g_loss_total, g_gen_total, g_r_total = 0, 0, 0
                    for _ in range(g_steps_k):
                        self.G.zero_grad(set_to_none=True)

                        fake_imgs_new = self.G(noisy_imgs)

                        # 关键：使用D_for_G而非D_main
                        # D_for_G没见过外类，提供温和的对抗信号
                        fake_logits_new = self.D_for_G(fake_imgs_new)
                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)

                        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs)
                        g_loss = g_loss_gan + r_alpha * g_loss_r

                        g_loss.backward()
                        self.optim_G.step()

                        g_loss_total += g_loss.item()
                        g_gen_total += g_loss_gan.item()
                        g_r_total += g_loss_r.item()

                    # 累计损失
                    epoch_d_main_loss_sum += d_main_loss.item()
                    epoch_d_for_g_loss_sum += d_for_g_loss.item()
                    epoch_g_r_total += g_r_total
                    epoch_g_gen_total += g_gen_total
                    batch_count += 1

                    # 日志
                    f.write(f"iter, d_main={d_main_loss.item():.3f}, d_for_g={d_for_g_loss.item():.3f}, "
                           f"g_gen={g_gen_total / g_steps_k:.3f}, g_r={g_r_total / g_steps_k:.3f}\n")

                # 更新进度条
                cnt = len(data_loader) * g_steps_k
                progress_bar.update(1)
                progress_bar.set_postfix(
                    d_main=epoch_d_main_loss_sum / cnt,
                    d_for_g=epoch_d_for_g_loss_sum / cnt,
                    g_gen=epoch_g_gen_total / cnt,
                    g_r=epoch_g_r_total / cnt
                )

                # 定期同步D_for_G（可选）
                if self.sync_d_every > 0 and (i + 1) % self.sync_d_every == 0:
                    # 将D_main的参数同步到D_for_G
                    self.D_for_G.load_state_dict(self.D.state_dict())

                # 保存checkpoint
                if (i + 1) % step == 0:
                    self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))

                # 早停
                if stop_recon_threshold is not None and (i + 1) >= int(stop_min_epoch):
                    mean_g_r = float(epoch_g_r_total / cnt)
                    if mean_g_r < float(stop_recon_threshold):
                        if (i + 1) % step != 0:
                            self._save_checkpoint(os.path.join(checkpoint_dir, f"{i+1}.pth"))
                        return int(i + 1)

        return int(epoch)