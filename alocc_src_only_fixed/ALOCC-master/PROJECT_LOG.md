# ALOCC 项目总控文档

> 最后更新：2026-05-30

---

## 1. 项目概况

**论文基础**：Sabokrou et al., *Adversarially Learned One-Class Classifier for Novelty Detection*, CVPR 2018

**核心问题**：Generator R 退化为"通用复印机"（identity shortcut）——对异常样本也能高质量重建，导致 D 无法区分内外类。

**Python 环境**：`C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`（torch 2.12+cu128，CUDA 可用）

---

## 2. 项目结构

```
ALOCC-master/
├── model.py                   # 所有模型变体
├── mnist_experiment.py        # 实验入口（CLI）
├── Metrics.py                 # AUC / SSIM / VIF / GMSD
├── utils.py                   # DEVICE / seed / timer
├── MNIST.py                   # MNIST 数据集
├── FashionMNIST.py            # FashionMNIST 数据集
├── CIFAR.py / COIL.py / UCSD.py
├── run_full_40ep.sh           # 批量实验脚本（bash）
├── run_full_40ep.ps1          # 批量实验脚本（PowerShell）
├── analyze_results.py         # 结果分析脚本
└── runs/                      # 实验输出（不入版本控制）
```

---

## 3. 已实现模型变体

| 类名 | 说明 | CLI 参数 |
|---|---|---|
| `ALOCC` | 基线：G+D，noisy→clean 重建，MSE 损失 | `--variant alocc` |
| `ALOCC_LOSS` | 扩展：D 见外类 + G 端 hinge 扭曲损失 | `--variant alocc_loss` |
| `ALOCC_LOSS_CLIP` | 在 ALOCC_LOSS 基础上加固定阈值梯度裁剪 | `--variant alocc_loss_clip` |
| `ALOCC_LOSS_BASELINE_REF` | 消融：外类 D + noisy-target BCE（对照组） | `--variant alocc_loss_baseline_ref` |
| `ALOCC_LOSS_CLS` | 加 Cosine Prototype 分类分支 | `--variant alocc_loss_cls` |
| `ALOCC_LOSS_DUAL_D` | 双判别器：D_main 见外类，D_for_G 不见 | `--variant alocc_loss_dual_d` |

---

## 4. 已实现结构改进

### S1 低秩噪声瓶颈（LowRankNoisyBottleneck）
- **位置**：`model.py` Generator encoder/decoder 接缝处
- **机制**：Conv1×1 降维→升维（rank 压缩）+ Dropout2d，阻断 identity shortcut 的容量条件
- **参数**：`--bottleneck-rank`（默认 0=关）、`--bottleneck-dropout`（默认 0.0）、`--bottleneck-noise-type {dropout,gaussian}`
- **等价性**：rank=0 且 dropout=0 时退化为 nn.Identity()，与无瓶颈基线 bitwise 一致

### CBAM 编码器注意力
- **位置**：`model.py` Generator encoder 每个 Conv block 后
- **机制**：通道注意力（MLP + avg/max pool）+ 空间注意力（7×7 conv）
- **参数**：`--use-cbam`（flag，默认关）
- **实验结论（2026-05-30，digit 0/2/8，seed 42/2026，40 epochs）**：

| 实验 | 基线 AUC | CBAM AUC | 变化 |
|---|---|---|---|
| digit8_seed42 | 0.858 | **0.913** | +0.055 ✓ |
| digit8_seed2026 | 0.772 | **0.838** | +0.066 ✓ |
| digit0_seed42 | 0.968 | **0.982** | +0.014 ✓ |
| digit0_seed2026 | 0.947 | **0.981** | +0.034 ✓ |
| digit2_seed42 | **0.907** | 0.877 | -0.031 ✗ |
| digit2_seed2026 | 0.809 | **0.927** | +0.117 ✓ |
| **平均** | 0.877 | **0.920** | **+0.043 ✓** |

### AGC 自适应梯度裁剪
- **位置**：`model.py` `_total_grad_norm` / `_agc_update_and_clip`
- **机制**：EMA 统计梯度范数，阈值 = mu + k×sigma，超标时裁剪
- **参数**：`--agc-ema-decay`（默认 0.99）、`--agc-k`（默认 3.0）、`--agc-min-clip`（默认 1.0）
- **推荐参数**：`--agc-ema-decay 0.999 --agc-k 1.0 --agc-min-clip 5.0`
- **结论**：AGC 机制生效但 AUC 未提升，训练不稳定根源是 GAN 对抗动态而非梯度爆炸

---

## 5. 当前实验基线（2026-05-30）

**配置**：`alocc_loss`，无 S1 瓶颈，无 CBAM，10 digits × 3 seeds（42/1337/2026），40 epochs

| digit | AUC 均值±std | raw_AUC | ssim_oc | ssim_gap |
|---|---|---|---|---|
| 0 | 0.839±0.111 | 0.854±0.099 | 0.885±0.032 | 0.049±0.018 |
| 1 | 0.983±0.023 | 0.983±0.023 | 0.626±0.329 | 0.114±0.013 |
| 2 | 0.778±0.142 | 0.805±0.120 | 0.691±0.280 | 0.044±0.030 |
| 3 | 0.801±0.142 | 0.808±0.137 | 0.908±0.015 | 0.031±0.004 |
| 4 | 0.842±0.115 | 0.856±0.108 | 0.911±0.009 | 0.028±0.003 |
| 5 | 0.717±0.144 | 0.741±0.160 | 0.722±0.279 | 0.046±0.034 |
| 6 | 0.915±0.050 | 0.932±0.043 | 0.890±0.025 | 0.054±0.018 |
| 7 | 0.778±0.204 | 0.804±0.169 | 0.892±0.020 | 0.051±0.016 |
| 8 | 0.600±0.089 | 0.458±0.162 | 0.646±0.389 | 0.040±0.017 |
| 9 | 0.679±0.069 | 0.704±0.104 | 0.447±0.344 | 0.055±0.016 |
| **均值** | **0.793** | — | **0.762** | 0.051 |

**关键诊断**：
- ssim_oc 均值 0.762，24/30 组 > 0.80 → **identity shortcut 严重**
- ssim_ic 与 ssim_oc 同步上涨（coupling_ratio ≈ 1.0），G 退化为通用复印机
- 红线通过率（ssim_oc ≤ 0.15）：**2/30**，远未达标

---

## 6. 北极星指标

| 指标 | 方向 | 当前均值 | 目标 |
|---|---|---|---|
| `ssim_oc` | ↓ | 0.762 | **≤ 0.15**（质量红线） |
| `ssim_gap = ssim_ic − ssim_oc` | ↑ | 0.051 | ≥ 0.15 |
| `raw_auc` | ↑ | — | ≥ 0.60（判别力下限） |
| `refined_auc` | ↑（不能塌） | 0.793 | 维持或提升 |

---

## 7. CBAM + S1 低秩瓶颈组合效果分析（2026-05-30）

### 7.1 数据流与位置关系

```
input [B,1,28,28]
  → Conv(1→32) + BN + LReLU → CBAM(32)      ← 第1层注意力门控
  → Conv(32→64) + BN + LReLU → CBAM(64)     ← 第2层注意力门控
  → Conv(64→128) + BN + LReLU → CBAM(128)   ← 第3层注意力门控
  → S1 Bottleneck: Conv1×1(128→8→128) + Dropout2d  ← 容量压缩
  → decoder → output
```

两者**串联**，CBAM 在前，S1 在后，攻击的是 identity shortcut 的不同维度。

### 7.2 各自的作用机制

| 机制 | 攻击点 | 对 inlier | 对 outlier |
|---|---|---|---|
| **CBAM** | 特征**质量**：通道+空间注意力，只见过 inlier，权重偏向 inlier 判别性方向 | 强化关键通道 | 对 outlier 特异结构响应弱 |
| **S1 瓶颈** | 特征**容量**：rank-8 投影强制压缩，阻断 identity shortcut 的容量条件 | 保留主要 inlier 方向 | 结构信号被截断 |

### 7.3 协同效应分析

**正向叠加的机制**：

CBAM 先对 encoder 输出的 128 通道做注意力加权，inlier 相关通道权重更高。这 128 维特征进入 S1 的 rank-8 投影时，rank-8 子空间会优先保留 CBAM 强化过的 inlier 方向——相当于 CBAM 帮 S1 "预筛选"了最重要的维度，让有限的 8 个 rank 用得更准。

对 outlier 的双重过滤：
- CBAM 阶段：outlier 特异通道权重低，特征被压制
- S1 阶段：rank-8 投影进一步截断 outlier 的残余结构信号

理论上 `ssim_oc` 应比单独使用任一机制更低。

**潜在冲突与风险**：

1. **CBAM 的注意力是联合训练的**：GAN 损失下 CBAM 可能学到"帮助重建任意输入"的注意力，而非"区分内外类"的注意力。若 identity shortcut 已经建立，CBAM 反而可能强化它。

2. **过度约束风险**：两者叠加会进一步压低 `ssim_ic`。从当前基线数据看，d2_s42 在 ep2 时 ssim_ic 仅 0.38，若 CBAM+S1 同时压缩，ssim_ic 可能跌到 0.2 以下，AUC 有塌的风险。

3. **digit 2 的特殊性**：CBAM 单独在 d2_s42 已退步 -0.031，说明 digit 2 的特征空间对注意力机制不友好（弧形笔画与多个外类共享），组合后风险最高。

### 7.4 从当前实验数据推断

d2_s42 的 per-epoch 曲线揭示了 shortcut 建立的时序：

```
ep1: ssim_ic=0.16, ssim_oc=0.11 → 差距 0.05，AUC=0.43（G 还没学好）
ep2: ssim_ic=0.38, ssim_oc=0.30 → 差距 0.08，AUC=0.64（最佳点）
ep5: ssim_ic=0.74, ssim_oc=0.66 → 差距 0.08，AUC=0.43（shortcut 建立中）
ep10+: ssim_ic≈ssim_oc≈0.9      → 差距 <0.02，AUC<0.3（完全退化）
```

CBAM 的作用是让 encoder 在早期就提取更有判别性的特征，可能延长"好窗口"（ep1-2）的持续时间。S1 的作用是从容量上限制 shortcut 的建立速度。**两者组合有望把"好窗口"从 1-2 个 epoch 延长到 5-10 个 epoch**，让选模器更容易找到好的 checkpoint。

### 7.5 预期结论

| digit | 预期 | 理由 |
|---|---|---|
| **8** | ✅ 正向叠加最强 | CBAM 聚焦双环结构，S1 限制容量，两者方向一致 |
| **0** | ✅ 正向叠加 | CBAM 单独已很好，S1 提供额外保障，边际收益递减 |
| **2** | ⚠️ 风险最高 | CBAM 单独已退步，组合可能过度约束；需降低 rank 或 dropout |
| **3/4** | 🔍 待验证 | ssim_oc 长期 >0.90，两者组合有望突破单一机制的天花板 |

### 7.6 建议消融实验

4 组 × 3 digits（0/2/8）× 2 seeds（42/2026）= 24 runs，约 20 分钟：

| 组 | 配置 | 目的 |
|---|---|---|
| A | 无 S1，无 CBAM | 对照基线 |
| B | S1 only（rank=8, p=0.3） | S1 单独贡献 |
| C | CBAM only | CBAM 单独贡献 |
| D | S1 + CBAM | 组合效果 |

**判定标准**：
- `ssim_oc(D) < min(ssim_oc(B), ssim_oc(C))` → 正向叠加
- `AUC(D) < AUC(A)` → 过度约束，需降低 rank
- `ssim_ic(D) < 0.15` → 过度压缩，G 已无法重建 inlier

---

## 8. 待办与下一步

| 优先级 | 任务 | 说明 |
|---|---|---|
| **P0** | 开启 S1 瓶颈重跑基线 | `--bottleneck-rank 8 --bottleneck-dropout 0.3`|
| **P0** | 开启 hinge 损失 | `--g-outclass-distortion-scale 0.3 --g-outclass-distortion-margin 0.6` |
| **P1** | S1 + CBAM 组合实验 | 验证两者是否正交叠加 |
| **P1** | 选模策略评估 | `best_auc` 在 identity shortcut 严重时选到后期 epoch，考虑加 ssim_oc 约束 |
| **P2** | ALOCC_LOSS_DUAL_D 实验 | 双判别器是否能稳定训练动力学 |
| **P2** | seed 稳健性验证 | 当前 3 seeds，建议扩到 5 seeds |

---

## 8. 可复现命令

```bash
# S1+Distortion 完整配置（推荐）
python mnist_experiment.py \
  --variant alocc_loss --specific 8 --epochs 40 --seed 42 \
  --train-count 4096 --batch-size 64 --out-per-class-count 300 \
  --bottleneck-rank 8 --bottleneck-dropout 0.3 \
  --d-outclass-loss-scale 0.1 \
  --g-outclass-distortion-scale 0.3 \
  --g-outclass-distortion-margin 0.6 \
  --selection-strategy best_auc \
  --output-dir runs/s1d_digit8_s42

# CBAM（digit8 推荐）
python mnist_experiment.py \
  --variant alocc_loss --specific 8 --epochs 40 --seed 42 \
  --use-cbam --output-dir runs/cbam_digit8_s42 \
  --selection-strategy best_auc
```
