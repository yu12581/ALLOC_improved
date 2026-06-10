# ALOCC 复现与改进项目

> CVPR 2018 论文 *Adversarially Learned One-Class Classifier for Novelty Detection* 的 PyTorch 复现与结构性改进。

---

## 1. 项目简介

本项目基于 ALOCC（Adversarially Learned One-Class Classifier）实现了一分类异常检测框架，并在原论文基础上引入了三项结构改进（S1 低秩瓶颈、CBAM 注意力、AGC 梯度裁剪）。

**核心问题：Identity Shortcut（恒等捷径）**

ALOCC 的 Generator R 在训练后期会退化为"通用复印机"——对内类和外类均做高质量重建，导致 Discriminator D 无法区分两者，AUC 崩溃。表现为 `ssim_ic` 与 `ssim_oc` 同步上涨，`coupling_ratio ≈ 1.0`。

**论文框架（Eq. 3-5）：**

```
min_R max_D ( E[log D(X)] + E[log(1 − D(R(X̃)))] )
L_R = ‖X − X'‖²        （MSE 重建损失，target 为干净 X）
L = L_{R+D} + λ·L_R    （λ = 0.2）
```

---

## 2. 环境依赖

| 组件 | 要求 |
|------|------|
| Python | 3.10+（开发环境 3.12） |
| PyTorch | 2.x，需 CUDA 支持 |
| CUDA | 需要 GPU（RTX 系列验证通过） |
| 操作系统 | Windows / Linux |

**Python 路径（当前环境）：**
```
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe
```

**安装依赖：**
```bash
pip install -r requirements.txt
```

**主要依赖包：**

| 包 | 用途 |
|---|---|
| `torch` / `torchvision` | 模型训练与数据加载 |
| `piq` | SSIM / VIF / GMSD 图像质量指标 |
| `scikit-learn` | AUC / ROC 计算 |
| `scipy` | EER 计算（brentq 求根） |
| `numpy` | 数值计算 |
| `tqdm` | 训练进度条 |
| `matplotlib` | 可视化 |

---

## 3. 项目结构

```
ALOCC-master/
├── model.py                    # 所有模型变体 + 结构组件（CBAM / S1 / AGC）
├── mnist_experiment.py         # 实验入口，支持完整 CLI 参数
├── Metrics.py                  # AUC / SSIM / VIF / GMSD 评估
├── MNIST.py                    # MNIST 数据集封装
├── FashionMNIST.py             # FashionMNIST 数据集封装
├── CIFAR.py                    # CIFAR 数据集封装
├── COIL.py                     # COIL-100 数据集封装
├── UCSD.py                     # UCSD Ped2 数据集封装
├── utils.py                    # DEVICE 检测 / 随机种子 / 计时装饰器
├── requirements.txt            # 依赖列表
├── PROJECT_LOG.md              # 实验总控文档（结论 / 待办 / 数据）
├── run_full_40ep.ps1           # 批量基线实验脚本（PowerShell）
├── run_full_40ep.sh            # 批量基线实验脚本（bash）
├── run_ablation_cbam_s1.ps1   # CBAM+S1 消融实验脚本（PowerShell）
├── run_ablation_cbam_s1.sh    # CBAM+S1 消融实验脚本（bash）
├── run_ablation_b_s1.ps1      # S1-only 补跑脚本
├── analyze_results.py          # 基线实验结果分析
├── analyze_ablation.py         # 消融实验结果分析
└── runs/                       # 实验输出（不入版本控制）
    ├── full_40ep_bestauc/      # 30 组基线结果
    └── ablation_cbam_s1/       # 24 组消融实验结果
```

---

## 4. 模型变体说明

| 变体 | CLI 参数 | 设计目的 |
|---|---|---|
| `ALOCC` | `--variant alocc` | 论文原版基线：G+D，仅训练内类，MSE 重建损失 |
| `ALOCC_LOSS` | `--variant alocc_loss` | 主力变体：D 额外见外类 + G 端 Hinge 扭曲损失，主动推开外类重建 |
| `ALOCC_LOSS_CLIP` | `--variant alocc_loss_clip` | 在 ALOCC_LOSS 基础上加固定阈值梯度裁剪，防止 D 梯度爆炸 |
| `ALOCC_LOSS_BASELINE_REF` | `--variant alocc_loss_baseline_ref` | 消融对照：外类 D + noisy-target BCE 重建（验证 MSE vs BCE 的差异） |
| `ALOCC_LOSS_CLS` | `--variant alocc_loss_cls` | 附加 Cosine Prototype 分类分支（GeM 池化 + 余弦相似度），提供额外特征约束 |
| `ALOCC_LOSS_DUAL_D` | `--variant alocc_loss_dual_d` | 双判别器：D_main 见外类用于测试，D_for_G 不见外类用于训练 G，解耦 G/D 的优化目标 |

### ALOCC_LOSS 的损失函数

```
D 损失：L_D = (L_real + L_fake) × (1−d_scale) + L_outclass × d_scale
G 损失：L_G = L_adv + r_alpha × L_MSE + g_scale × ReLU(margin − |R(X_out)−X_out|_L1)
```

其中 Hinge 项 `ReLU(margin − L1)` 在重建误差未达到 margin 时才激活，强迫 G 对外类产生足够大的重建误差。

---

## 5. 结构改进说明

### 5.1 S1 低秩噪声瓶颈（LowRankNoisyBottleneck）

**位置**：Generator encoder/decoder 接缝（`model.py:121`）

**原理**：在编码器输出的 128 通道特征上插入 `Conv1×1(128→r→128)` 降维-升维夹层，将有效秩强制压缩到 r（默认 8）。Identity shortcut 要求瓶颈矩阵满秩；rank-r 约束从容量层面阻断 shortcut 的建立条件。可叠加 `Dropout2d` 增加噪声扰动。

```
encoder输出 [B,128,4,4]
  → Conv1×1(128→8)   # 降维，秩压缩
  → Conv1×1(8→128)   # 升维，信息瓶颈
  → Dropout2d(p)      # 可选噪声扰动
  → decoder输入
```

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--bottleneck-rank` | `0`（关） | 低秩瓶颈的秩 r，0 = 退化为 Identity |
| `--bottleneck-dropout` | `0.0`（关） | Dropout 概率 |
| `--bottleneck-noise-type` | `dropout` | `dropout` 或 `gaussian` |

> rank=0 且 dropout=0 时与无瓶颈基线 **bitwise 完全一致**。

### 5.2 CBAM 编码器注意力（CBAM）

**位置**：Generator encoder 每个 Conv block 后（`model.py:33`），共 3 处

**原理**：来自 ECCV 2018。先做通道注意力（avg-pool + max-pool 经 MLP 求权重），再做空间注意力（沿通道维 avg/max 拼接后经 7×7 Conv），两者相乘作为门控信号。在 one-class 训练下，注意力权重自然偏向内类判别性方向。

```
特征图 x
  → 通道注意力：sigmoid(MLP(avg(x)) + MLP(max(x))) × x
  → 空间注意力：sigmoid(Conv7×7([avg_C(x), max_C(x)])) × x
```

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use-cbam` | 关（flag） | 开启后 G 参数量从 ~330K 增至 ~387K |

### 5.3 AGC 自适应梯度裁剪

**位置**：`model.py` `_agc_update_and_clip`

**原理**：用 EMA 统计梯度 L2 范数的均值 μ 和方差 σ²，动态裁剪阈值 = μ + k×σ。相比固定阈值裁剪，能自适应训练阶段的梯度量级变化。

**CLI 参数：**

| 参数 | 默认值 | 推荐值 | 说明 |
|---|---|---|---|
| `--agc-ema-decay` | `0.99` | `0.999` | EMA 衰减系数，越大阈值越平滑 |
| `--agc-k` | `3.0` | `1.0` | 裁剪灵敏度，越小越激进 |
| `--agc-min-clip` | `1.0` | `5.0` | 阈值下限，防止初期过激裁剪 |

> 仅 `ALOCC_LOSS` 及其子类支持 AGC，`ALOCC` 基线不支持。

---

## 6. 快速开始

### 单次实验

```powershell
# 基线 ALOCC（论文原版）
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe mnist_experiment.py `
  --variant alocc --specific 0 --epochs 40 --seed 42 `
  --train-count 4096 --batch-size 64 `
  --selection-strategy best_auc `
  --output-dir runs/alocc_d0_s42

# ALOCC_LOSS + S1 瓶颈（推荐配置）
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe mnist_experiment.py `
  --variant alocc_loss --specific 8 --epochs 40 --seed 42 `
  --train-count 4096 --batch-size 64 --out-per-class-count 300 `
  --bottleneck-rank 8 --bottleneck-dropout 0.3 `
  --d-outclass-loss-scale 0.1 `
  --g-outclass-distortion-scale 0.3 `
  --g-outclass-distortion-margin 0.6 `
  --selection-strategy best_auc `
  --output-dir runs/s1d_d8_s42

# CBAM（digit 8 推荐）
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe mnist_experiment.py `
  --variant alocc_loss --specific 8 --epochs 40 --seed 42 `
  --train-count 4096 --batch-size 64 --out-per-class-count 300 `
  --use-cbam --selection-strategy best_auc `
  --output-dir runs/cbam_d8_s42
```

### 批量实验（10 digits × 3 seeds = 30 runs）

```powershell
cd "d:\codeVS\ALLOC_mzy-master\ALLOC_mzy-master\alocc_src_only_fixed\ALOCC-master"
.\run_full_40ep.ps1
```

### 消融实验（CBAM vs S1 vs CBAM+S1）

```powershell
.\run_ablation_cbam_s1.ps1
```

### 结果分析

```powershell
# 基线实验分析
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe analyze_results.py

# 消融实验分析
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe analyze_ablation.py
```

---

## 7. 实验结论摘要

### 7.1 基线实验（30 组：10 digits × 3 seeds，alocc_loss，无 S1/CBAM，40 epochs）

| digit | AUC 均值±std | ssim_oc 均值 | 问题 |
|---|---|---|---|
| 0 | 0.865±0.111 | 0.892 | seed 敏感 |
| 1 | 0.983±0.023 | 0.626 | ✅ 较好 |
| 2 | 0.778±0.142 | 0.691 | 不稳定 |
| 3 | 0.801±0.142 | 0.908 | shortcut 严重 |
| 4 | 0.842±0.115 | 0.911 | shortcut 严重 |
| 5 | 0.717±0.144 | 0.722 | AUC 低 |
| 6 | 0.915±0.050 | 0.890 | ✅ 较好 |
| 7 | 0.778±0.204 | 0.892 | 方差大 |
| 8 | 0.600±0.089 | 0.921 | ❌ 最差 |
| 9 | 0.679±0.069 | 0.447 | AUC 低 |
| **均值** | **0.793** | **0.762** | 红线通过率 2/30 |

**核心诊断**：ssim_ic 与 ssim_oc 同步上涨（coupling_ratio ≈ 1.0），identity shortcut 严重。24/30 组 ssim_oc > 0.80。

### 7.2 消融实验结论（3 digits × 2 seeds，A/B/C/D 四组）

| digit | A（基线） | B（S1） | C（CBAM） | D（S1+CBAM） | 判定 |
|---|---|---|---|---|---|
| **0** | AUC=0.865, oc=0.892 | AUC=0.956, oc=0.818 | AUC=0.995, oc=0.899 | AUC=0.982, oc=0.849 | ➡️ 部分叠加 |
| **2** | AUC=0.803, oc=0.612 | AUC=0.946, oc=0.866 | AUC=0.707, oc=0.551 | AUC=0.925, oc=0.870 | ➡️ 部分叠加 |
| **8** | AUC=0.549, oc=0.921 | AUC=0.685, oc=0.508 | AUC=0.568, oc=0.526 | AUC=**0.701**, oc=**0.099** | ✅ 正向叠加 |

**关键发现：**
- **digit 8**：S1+CBAM 组合将 ssim_oc 从 0.921 压至 **0.099**（红线以下），AUC 最高。CBAM 聚焦双环结构，S1 限制容量，两者方向高度一致
- **digit 2**：S1 单独 AUC 最高（0.946），CBAM 在此类上有退步风险
- **digit 0**：CBAM 单独已是天花板（AUC=0.995），S1 边际收益递减
- **选模是瓶颈**：D 在 digit 8 上的最优窗口在 ep1-2，`best_auc` 策略能选到，其他策略可能错过

---

## 8. 指标说明

| 指标 | 含义 | 优化方向 | 计算方式 |
|---|---|---|---|
| `ssim_ic` | R(X_inlier) 与 X_inlier 的 SSIM | 越高越好（重建质量） | piq.ssim，kernel=7，[0,1] 归一化 |
| `ssim_oc` | R(X_outlier) 与 X_outlier 的 SSIM | **越低越好**（扭曲度） | 同上；≤ 0.15 为质量红线 |
| `ssim_gap` | ssim_ic − ssim_oc | **越高越好**（可分离度） | 直接差值；≥ 0.15 为目标 |
| `raw_auc` | D(X) 分数的 AUC（未经 R） | 越高越好 | sklearn.roc_auc_score；Oracle 选模主指标 |
| `refined_auc` | D(R(X)) 分数的 AUC（经过 R） | 越高越好 | 同上；论文 Figure 7 主指标 |
| `auc_gain` | refined_auc − raw_auc | > 0 为正（R 有助于判别） | 直接差值 |
| `score_gap` | D(R(inlier)) 均值 − D(R(outlier)) 均值 | 越高越好 | 分组均值差 |

> **注意**：ssim_oc 高不一定 AUC 低（见 digit 2 的 B_s1 组），两者在某些类上解耦。
> ssim_oc ≤ 0.15 是**结构质量红线**，但 AUC 仍是最终分类性能的唯一可信指标。

---

## 9. 已知问题与下一步

| 优先级 | 任务 | 说明 |
|---|---|---|
| **P0** | S1+Hinge 完整配置重跑 | 补加 `--g-outclass-distortion-scale 0.3 --g-outclass-distortion-margin 0.6`，验证 hinge 激活对 digit 2/3/5 的效果 |
| **P0** | S1+CBAM+Hinge 组合 | 在 digit 8 的正向叠加基础上加 hinge，看能否进一步稳定 |
| **P1** | digit 8 深度实验 | rank ∈ {4,8,16} × dropout ∈ {0.1,0.3,0.5} 网格，找最优瓶颈配置 |
| **P1** | 选模策略优化 | `best_auc` 在 identity shortcut 严重时选后期 epoch，考虑加 ssim_oc 约束或 redline 策略 |
| **P2** | ALOCC_LOSS_DUAL_D 实验 | 双判别器是否能稳定 digit 8 的训练动力学 |
| **P2** | seed 稳健性扩展 | 当前 2-3 seeds，建议扩到 5 seeds 再下结论 |
| **P3** | FashionMNIST / CIFAR 扩展 | 验证改进方案的跨数据集泛化能力 |

---

## 10. 引用

```bibtex
@inproceedings{sabokrou2018adversarially,
  title     = {Adversarially Learned One-Class Classifier for Novelty Detection},
  author    = {Sabokrou, Mohammad and Khalooei, Mohammad and Fathy, Mahmood and Adeli, Ehsan},
  booktitle = {CVPR},
  year      = {2018}
}
```

**CBAM 参考**：Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018

