# S1D-ALOCC：结构化实验方法文档

**版本**: 1.0  
**日期**: 2026-05-01  
**项目**: S1D-MNIST — S1 结构性瓶颈增强 ALOCC 异常检测  
**评审目标**: 通过同行评审 / 内部评审验收标准

---

## 1. 实验框架

### 1.1 核心目标

证明在 ALOCC 生成器（Generator）的编码器/解码器接缝处插入 S1 低秩瓶颈（Conv1×1 下投影 + Conv1×1 上投影 + Dropout2d）能够显著提升 MNIST 一分类异常检测性能，与无瓶颈的基线 ALOCC 对比。

**交付物**: 一篇可发表的对比实验论文。

### 1.2 科学假设

| 编号 | 假设 | 验证方式 |
|------|------|----------|
| **H1**（分类性能） | S1D 在 9/10 个 MNIST 一分类任务上 raw_auc 高于基线 ALOCC | Oracle "Best of 20 Epochs" raw_auc，每种条件 50 runs 聚合 |
| **H2**（结构性） | S1D 抑制"复印机效应"（恒等捷径）：其 R（精炼器）无法完美重建外类图像，而基线退化为复印机 | ssim_oc（R(外类) 与原外类的 SSIM）；预期 S1D << 基线 |
| **H3**（稳定性） | S1D 在所有 20 个训练 epoch 上的平均 AUC 高于基线 | 按 epoch 的 AUC 经 20 epochs × 50 runs 平均 |

### 1.3 变量定义

#### 自变量（受操控）

| 变量 | 水平 | 备注 |
|------|------|------|
| 模型变体 | `alocc`（基线）, `alocc_loss`（S1D） | S1D = `alocc_loss` + S1 瓶颈启用 |
| S1 瓶颈秩 | 基线: 0（恒等映射）, S1D: 8（数字 7: 4） | 通过 `--bottleneck-rank` 控制 |
| S1 瓶颈 Dropout | 基线: 0.0, S1D: 0.3（数字 7: 0.5） | 通过 `--bottleneck-dropout` 控制 |
| 数字（内类） | 0–9（10 个水平） | 一分类：在单个数字上训练，在全部 10 个数字上测试 |

#### 因变量（被测量）

| 类别 | 指标 | 符号 | 范围 | 方向 |
|------|------|------|------|------|
| **主要** | D(X) 的 AUC | `raw_auc` | [0, 1] | 越高越好 |
| **主要** | D(R(X)) 的 AUC | `refined_auc` | [0, 1] | 越高越好 |
| 分类 | 准确率 | `ACC` | [0, 1] | 越高越好 |
| 分类 | F1 分数 | `F1` | [0, 1] | 越高越好 |
| 分类 | 等错误率 | `EER` | [0, 1] | 越低越好 |
| 质量（内类） | R(X_in) 与 X_in 的 SSIM | `ssim_ic` | [0, 1] | 越高越好 |
| 质量（外类） | R(X_out) 与 X_out 的 SSIM | `ssim_oc` | [0, 1] | 越低越好（扭曲） |
| 质量 | SSIM 差距 | `ssim_gap` = ssim_ic − ssim_oc | [−1, 1] | 越高越好 |
| 质量 | VIF（内/外） | `vif_ic`, `vif_oc` | [0, ∞) | — |
| 质量 | GMSD（内/外） | `gmsd_ic`, `gmsd_oc` | [0, ∞) | — |
| 分数差距 | D(R(内)) 均值 − D(R(外)) 均值 | `score_gap` | [−1, 1] | 越高越好 |
| 分数差距（原始） | D(内) 均值 − D(外) 均值 | `raw_score_gap` | [−1, 1] | 越高越好 |
| 改进 | refined_score_gap − raw_score_gap | `score_gap_gain` | [−2, 2] | 越高越好 |
| 改进 | refined_auc − raw_auc | `auc_gain` | [−1, 1] | 越高越好 |

#### 受控变量（固定）

| 参数 | 值 | 理由 |
|------|-----|------|
| 数据集 | MNIST（28×28 灰度） | 一分类异常检测的公认基准 |
| 训练集大小 | 由 `--train-count 10000` 自动限制到该类全量（约 5,400–6,700） | BL-B 约定 |
| 测试内类数 | 200 | 匹配项目历史约定 |
| 噪声标准差 | 0.31 | ALOCC 论文默认值 |
| 输入归一化 | [−1, +1]（tanh 输出范围） | 生成器通过 Tanh 输出 |
| 优化器 | RMSprop（PyTorch 默认: α=0.99, ε=1e-8） | BL-B；BL-A 使用 TF1.15 verbatim（α=0.9, ε=1e-10） |
| 学习率 | 0.002 | 项目历史默认值 |
| 权重衰减 | 0.0 | 禁用 |
| 标签平滑 | 0.0 | 禁用 |
| 评估批大小 | 128 | 独立于训练批大小 |
| 设备 | CUDA | 需要 GPU（单 GPU 上测试） |

### 1.4 评估指标注册表

所有指标均在**测试集**上计算，测试集包含内类样本（训练所用的数字）和外类样本（其余 9 个数字）。

**主终点**: `raw_auc` — 未经精炼的二分类器 D(X) 的 AUC。该指标用于 Oracle 选模。

**共终点**: `refined_auc` — D(R(X)) 的 AUC。

**关键佐证**:
- `ssim_oc`：结构失真度指标 — 低值表明 S1D 打破了复印机效应。
- `score_gap_gain`：经过 R 后 D 的分离度提升了多少。

---

## 2. 架构总览

### 2.1 系统级数据流图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          实验流水线（EXPERIMENT PIPELINE）                 │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────────────┐   │
│  │ 数据加载  │──▶│ 模型构建  │──▶│ 训练循环   │──▶│ 检查点评估        │   │
│  │ Data     │   │ Model    │   │ Training  │   │ (epoch 1..N)      │   │
│  │ Loading  │   │ Build    │   │ Loop      │   └────────┬──────────┘   │
│  └──────────┘   └──────────┘   └───────────┘            │               │
│                                                          ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │               Oracle 选模（raw_auc）                              │    │
│  │  ↓ best_epoch → best.pth → Figure 7 分数 + 三联图导出           │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │              跨实验聚合（50 runs）                                 │    │
│  │  每种条件 10 个数字 × 5 个种子 的 mean ± std                      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 单次运行数据流（详细）

```
                             训练阶段（20 epochs）
┌────────────┐   ┌──────────────────────────────────────────────────────────┐
│ train_     │   │  每个 epoch:                                            │
│ loader     │──▶│                                                        │
│（内类：     │   │  ┌──────────────────────────────────────────────────┐  │
│ Baseline   │   │  │  Batch 循环:                                      │  │
│ 仅内类，    │   │  │  ┌────────┐   ┌──────────┐   ┌────────────────┐ │  │
│ S1D 内外兼 │   │  │  │加高斯   │──▶│ G(加噪)  │──▶│ 训练 D:        │ │  │
│ 有）       │   │  │  │噪声     │   │ = R(X_n) │   │ BCE(D(R),真实) │ │  │
│            │   │  │  │X → X_n  │   │          │   │ + BCE(D(R),假) │ │  │
│  ────────  │   │  │  └────────┘   └──────────┘   └────────┬───────┘ │  │
│ outclass_  │   │  │                                        ▼          │  │
│ loader     │──▶│  │  ┌──────────────────────────────────────────────┐ │  │
│（仅 S1D）  │   │  │  │ 训练 G（×2 步）:                             │ │  │
│ 外类样本   │   │  │  │   L_G = BCE(D(R), 真实)                     │ │  │
│            │   │  │  │        + r_alpha · MSE(R, X)  (精炼项)      │ │  │
│            │   │  │  │        + distortion_scale ·                 │ │  │
│  ────────  │   │  │  │          ReLU(margin − |R(X_out)−X_out|_L1) │ │  │
│ 全部 9 个  │   │  │  │         (仅 S1D: 外类扭曲)                  │ │  │
│ 外类数字   │   │  │  └──────────────────────────────────────────────┘ │  │
│            │   │  └──────────────────────────────────────────────────┘  │
│            │   │  Step=1: 每个 epoch 保存检查点                        │
└────────────┘   └──────────────────────────────────────────────────────────┘

                             评估阶段
┌────────────┐   ┌──────────────────────────────────────────────────────────┐
│ test_      │──▶│  对每个保存的 epoch 检查点 (1..20):                     │
│ loader     │   │  1. 加载 epoch.pth                                      │
│（内+外）   │   │  2. G(加噪) → 精炼图                                   │
│            │   │  3. D(精炼图) → 分数                                    │
│            │   │  4. 计算 17 项指标（见 §1.3, §1.4）                     │
│            │   │  5. 计算 paper_score, distortion_score                  │
│            │   │  6. 存入 → records[epoch]                               │
│            │   └─────────────────────────┬───────────────────────────────┘
│                                          ▼
│                            ┌─────────────────────────┐
│                            │  Oracle 选模：            │
│                            │  best_epoch = argmax     │
│                            │  (records, raw_auc)      │
│                            │  在 epoch 1..20 中选取    │
│                            │  (阈值: auc ≥ 0.60)       │
│                            └──────────┬──────────────┘
│                                        ▼
│  ┌─────────────────────────────────────────────────────┐
│  │  输出:                                               │
│  │  - best_epoch, best_metrics（完整记录）              │
│  │  - best.pth（最优 epoch 检查点的副本）               │
│  │  - summary.json（完整实验记录）                      │
│  │  - figure7_scores.json（逐样本 D(X) vs D(R)）       │
│  │  - triplets/（原始/加噪/生成图像导出）              │
│  └─────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### 2.3 模块清单与接口合同

| 模块 | 文件 | 入口点 | 输入 | 输出 | 被谁调用 |
|------|------|--------|------|------|----------|
| `MNIST` | `MNIST.py` | `MNIST()` 类 | 数字标签, 数量, noise_std | `TensorDataset(imgs, noisy_imgs, labels)` | `build_data()` |
| `build_data` | `mnist_experiment_runner.py:45` | `build_data(args)` | argparse Namespace | `(train_loader, test_loader, [outclass_loader])` | `run_experiment()` |
| `build_model` | `mnist_experiment_runner.py:29` | `build_model(variant, ...)` | 变体字符串, 超参 | `ALOCC / ALOCC_LOSS 实例` | `run_experiment()` |
| `ALOCC._train` | `model.py:263` | `model._train(...)` | train_loader, epoch, r_alpha | int（已训练 epoch 数） | `run_experiment()` |
| `ALOCC_LOSS._train` | `model.py:392` | `model._train(...)` | train + outclass loaders | int（已训练 epoch 数） | `run_experiment()` |
| `evaluate_checkpoints` | `mnist_experiment_runner.py:446` | `evaluate_checkpoints(...)` | model, checkpoint_dir, loader | `(best_epoch, best_metrics, records, selection_info)` | `run_experiment()` |
| `_select_records` | `mnist_experiment_runner.py:296` | `_select_records(...)` | records 列表, 策略参数 | `(best_record, selection_info)` | `evaluate_checkpoints()` |
| `calculate_metrics` | `Metrics.py:89` | `calculate_metrics(model, loader, inner)` | model, dataloader, inner_class | `(F1, ACC, EER, AUC, SSIM_IC, SSIM_OC, VIF_IC, VIF_OC, GMSD_IC, GMSD_OC)` | `evaluate_checkpoints()` |
| `compute_paper_score_stats` | `mnist_experiment_runner.py:121` | `compute_paper_score_stats(...)` | model, loader, inner_class | `{raw_auc, refined_auc, score_gap_gain, auc_gain, ...}` | `evaluate_checkpoints()` |
| `LowRankNoisyBottleneck` | `model.py:76` | `bottleneck.forward(x)` | 特征图 [B,128,4,4] | 特征图 [B,128,4,4]（低秩投影后） | `Generator.forward()` |
| `Generator.forward` | `model.py:163` | `G.forward(x)` | 图像 [B,1,28,28] | 重建图像 [B,1,28,28] | D 和评估 |
| `Discriminator.forward` | `model.py:196` | `D.forward(x)` | 图像 [B,1,28,28] | logit [B,1] | D 训练, 评估 |
| `run_experiment` | `mnist_experiment_runner.py:525` | `run_experiment(args)` | Args 数据类 | 将 summary.json 写入 `args.output_dir` | `run_paper_mnist_figure6_7.py:main()` |

---

## 3. 实验流水线 — 逐步详解

### 步骤 0：随机种子初始化

**文件**: `run_paper_mnist_figure6_7.py:157` → `set_random_seed(args.seed)`  
**文件**: `mnist_experiment_runner.py:526` → `set_random_seed(getattr(args, "seed", None))`

```
输入:  seed（整数或 None；None → 默认 42）
效果: torch.manual_seed(seed) + numpy.random.seed(seed)
输出: 确定性的权重初始化和数据打乱
```

### 步骤 1：数据加载与预处理

**文件**: `mnist_experiment_runner.py:45-81` → `build_data(args)`

```
输入:  args.specific（内类数字）, args.train_count, args.noise_std,
       args.batch_size, args.out_per_class_count（仅 S1D）,
       args.test_outlier_labels（可选过滤器）
流程:
  1. 实例化 MNIST(train=True, specific=d, count=train_count, noise_std=0.31)
     → 返回 (clean_img, noisy_img, label) 三元组的 TensorDataset
     - noisy_img = clean_img + N(0, noise_std)
     - 图像归一化到 [-1, +1]
  2. train_loader = DataLoader(batch=64/128, shuffle=True)
  3. test_dataset = MNIST(train=False, same specific, count=200)
     → 包含所有 10 个数字（内类 + 9 个外类）
  4. 如果指定 test_outlier_labels：将测试集过滤为仅包含那些外类数字
  5. （仅 S1D）outclass_dataset = MNIST(train=True, per_out_class_count=300)
     → 从全部 9 个非内类数字中均衡采样外类样本
  6. outclass_loader = DataLoader(batch=64, shuffle=True)

输出: (train_loader, test_loader, [outclass_loader 或 None])
```

**噪声注入的关键数据流**:
- 训练阶段: `MNIST.__init__` 通过对 `clean_imgs` 添加高斯噪声生成 `noisy_imgs`，然后裁剪到 [-1, 1]。数据加载器返回 `(clean_img, noisy_img, label)`。
- 生成器的输入始终是**加噪**图像；其训练目标（精炼项）是**干净**（内类）图像。

### 步骤 2：模型实例化

**文件**: `mnist_experiment_runner.py:29-42` → `build_model(variant, lr, ...)`

```
输入:  variant ("alocc" 或 "alocc_loss"), lr, bottleneck_rank, bottleneck_dropout, ...
流程:
  1. if variant == "alocc":     返回 ALOCC(in_h=28, out_h=28, ...)
  2. if variant == "alocc_loss": 返回 ALOCC_LOSS(in_h=28, out_h=28, ...)
  3. 两者都继承自 ALOCC 基类（model.py:199）
  4. Generator 和 Discriminator 在 __init__ 中创建
  5. 如果 rank>0 或 dropout>0，则在 Generator 内部构建 S1 瓶颈

输出: PyTorch nn.Module（ALOCC 或 ALOCC_LOSS 子类）
```

### 步骤 3：训练循环

#### 3a. 基线 ALOCC（`alocc` 变体）

**文件**: `model.py:263-339` → `ALOCC._train()`

```
输入:  data_loader（仅内类 → (clean, noisy, label)）,
       epoch=20, step=1, r_alpha=0.2

每个 epoch 循环:
  对 train_loader 中的每个 batch (clean_x, noisy_x, _):
    ┌── D 训练（1 步）─────────────────────────────────────┐
    │  D(clean) → real_logits;  BCE(real_logits, real_label)│
    │  G(noisy).detach() → fake; D(fake) → fake_logits     │
    │  BCE(fake_logits, fake_label)                         │
    │  D_loss = BCE_real + BCE_fake                         │
    └───────────────────────────────────────────────────────┘
    ┌── G 训练（2 步）──────────────────────────────────────┐
    │  G(noisy) → fake_new; D(fake_new) → fake_logits_new   │
    │  L_gan = BCE(fake_logits_new, real_label)             │
    │  L_r = BCE_with_logits(fake_new, noisy_x)  [*verbatim]│
    │  G_loss = L_gan + r_alpha * L_r                       │
    └───────────────────────────────────────────────────────┘
  每个 epoch 保存检查点（step=1）

输出: trained_epochs（整数，通常 == epoch）
```

**关键说明**: 精炼损失 `L_r` 使用 `F.binary_cross_entropy_with_logits(fake_new, noisy_x)` — 这是 verbatim TF1.15 协议中的约定，G 通过预测加噪输入本身来学习去噪。

#### 3b. S1D ALOCC_LOSS（`alocc_loss` 变体）

**文件**: `model.py:391-491` → `ALOCC_LOSS._train()`

```
输入:  data_loader（内类）, outclass_loader（itertools.cycle 循环）,
       epoch=20, step=1, r_alpha=0.2,
       d_outclass_loss_scale=0.1, g_outclass_distortion_scale=0.3,
       g_outclass_distortion_margin=0.6

与基线 ALOCC 的差异:
  1. D 训练增加了外类分支:
     - D(outclass_clean) → BCE(应为 FAKE)
     - D(R(outclass_noisy).detach()) → BCE(应为 FAKE)
     - d_loss_outclass = 0.5 × [BCE(out_clean, fake) + BCE(R(out_noisy), fake)]
     - D_loss = (BCE_real + BCE_fake) × (1−scale) + d_loss_outclass × scale
  2. G 训练增加了外类扭曲分支:
     - R_out = G(out_noisy)
     - out_recon_l1 = |R_out − out_clean|_L1（平均绝对差异）
     - L_distort = ReLU(margin − out_recon_l1)
     - G_loss = L_gan + r_alpha × L_refine + distortion_scale × L_distort
  3. 外类数据来自 itertools.cycle(outclass_loader):
     每个内类 batch 配一个外类 batch（1:1 比例）

输出: trained_epochs（整数，通常 == epoch）
```

### 步骤 4：检查点评估

**文件**: `mnist_experiment_runner.py:446-522` → `evaluate_checkpoints()`

```
输入:  model, checkpoint_dir（20 个已保存 .pth 文件）, test_loader,
       inner_class, selection_strategy="raw_auc"

对每个 epoch 1..20:
  1. 加载 epoch.pth 检查点
  2. 通过 calculate_metrics()（Metrics.py:89）计算 10 项核心指标:
     - F1, ACC, EER, AUC（基于 D(R(x)) 分数，最佳 F1 阈值）
     - SSIM_IC, SSIM_OC, VIF_IC, VIF_OC, GMSD_IC, GMSD_OC
  3. 通过 compute_paper_score_stats()（runner:121）计算 11 项论文/分数统计:
     - raw_auc, raw_eer, raw_acc（基于 D(X) 分数）
     - refined_auc, refined_eer, refined_acc（基于 D(R(X)) 分数）
     - score_gap, score_gap_gain, auc_gain
     - raw_score_gap（D(X) 内类均值 − 外类均值）
     - score_in_mean, score_out_mean, raw_score_in/out_mean
  4. 计算衍生指标:
     - ssim_gap = ssim_ic − ssim_oc
     - vif_gap = vif_ic − vif_oc
     - gmsd_gap = gmsd_oc − gmsd_ic
     - score_gap = refined_score_in_mean − refined_score_out_mean
  5. 全部存入 epoch 记录
  6. 附加 paper_score（复合指标；未用于 Oracle）和 distortion_score

输出: records（20 个字典的列表，每个字典约 40 个键）
```

### 步骤 5：Oracle 选模（"Best of N Epochs"）

**文件**: `mnist_experiment_runner.py:296-443` → `_select_records(strategy="raw_auc")`

```
输入:  records（20 个 epoch 字典）, strategy="raw_auc",
       selection_epoch_start=1, selection_epoch_end=20,
       selection_min_auc=0.60

流程:
  1. 按 epoch 窗口 [start, end] 过滤候选
  2. 按 min_auc 阈值过滤合格候选（如果设置）
  3. 如果无合格 epoch：回退警告，使用全部候选
  4. 选择 best_epoch = argmax(candidates, key="raw_auc")
  5. 复制 epoch_{best}.pth → best.pth

输出: best_record（单个字典）, selection_info（元数据）
```

**关键约束（ADR-011）**：raw_auc 是**唯一**的选模标准。ssim_oc 或其他指标不得用作选模阈值——这样做会构成循环论证，已获项目评审人（师兄）确认。

### 步骤 6：事后分析

**文件**: `run_paper_mnist_figure6_7.py:206-238`

```
Oracle 选模完成后:
  1. Figure-7 风格分数导出:
     - 对 sample_count=40 个内类 + 外类测试图像
     - 计算每张图像的 D(X) 和 D(R(X))
     - 保存到 figure7_scores.json
  2. 三联图导出:
     - 内类和外类的原始 / 加噪 / 生成图像组
     - 保存到 triplets/ 目录
  3. 在输出根目录保存流水线概要 JSON
```

### 步骤 7：跨实验聚合

**不在单个脚本中完成，由报告脚本执行**（`_tmp_b_default_report.py`, `_tmp_per_epoch_table.py`, `b4_per_epoch_plots.py`）：

```
对每个条件（基线或 S1D）:
  对 50 个 runs 中的每一个（10 个数字 × 5 个种子）:
    读取 summary.json → best_metrics, records[]

  计算跨实验统计:
    - best_metrics 的 mean ± std（raw_auc, F1, ACC, ssim_ic, ssim_oc 等）
    - 跨全部 runs 的逐 epoch mean ± std
    - Oracle epoch 频率分布
    - 生成对比图（逐数字 AUC/ACC/F1 折线图）

输出: 用于论文的表格和图表（PROJECT_LOG.md §4）
```

---

## 4. 组件规格

### 4.1 生成器架构

```
Generator (model.py:103-168)
├── 编码器: nn.Sequential
│   ├── Conv2d(1→32, k=5, s=2, p=2), BN, LeakyReLU(0.2)
│   ├── Conv2d(32→64, k=5, s=2, p=2), BN, LeakyReLU(0.2)
│   └── Conv2d(64→128, k=5, s=2, p=2), BN, LeakyReLU(0.2)
│   → 输出: [B, 128, 4, 4]
│
├── [S1] 瓶颈（插在编码器/解码器接缝处）
│   └── LowRankNoisyBottleneck(128, rank=r, dropout=p)
│       ├── Conv1×1(128→r)  [如果 rank > 0]
│       ├── Conv1×1(r→128)  [如果 rank > 0]
│       ├── Dropout2d(p)    [如果 p > 0, noise_type="dropout"]
│       └── （或如果 noise_type="gaussian" 则注入高斯噪声）
│   → 输出: [B, 128, 4, 4]
│   → 当 rank=0 且 dropout=0 时: nn.Identity() → 与基线比特级一致
│
├── 解码器: nn.Sequential
│   ├── ConvTranspose2d(128→32, k=5, s=2, p=2), BN, ReLU
│   ├── ConvTranspose2d(32→16, k=5, s=2, p=2), BN, ReLU
│   └── ConvTranspose2d(16→1, k=5, s=2, p=2), Tanh
│   → 输出: [B, 1, 28, 28] 范围 [-1, +1]
│
└── 可选分类器分支（仅 ALOCC_LOSS_CLS，S1D 中未使用）
```

**权重初始化**: 所有 Conv/Linear 使用 Normal(0, 0.02)；BN 权重使用 Normal(1, 0.02)。

### 4.2 判别器架构

```
Discriminator (model.py:170-197)
├── Conv2d(1→16, k=5, s=2, p=2), LeakyReLU(0.2)
├── Conv2d(16→32, k=5, s=2, p=2), BN, LeakyReLU(0.2)
├── Conv2d(32→64, k=5, s=2, p=2), BN, LeakyReLU(0.2)
├── Conv2d(64→128, k=5, s=2, p=2), BN, LeakyReLU(0.2)
├── Flatten
└── Linear(128 × 2 × 2 → 1)
    → 输出: [B, 1]（logit，forward 中不含 sigmoid）
```

### 4.3 LowRankNoisyBottleneck（S1）

**文件**: `model.py:76-100`

```
输入:  [B, C=128, H=4, W=4]
如果 rank > 0:
  ┌── down: Conv1×1(128→rank), 无偏置
  ┌── up:   Conv1×1(rank→128), 无偏置
  → 低秩投影: rank 个通道瓶颈压缩信息流
如果 dropout > 0:
  ┌── Dropout2d(p=dropout) — 特征图上的空间相关噪声
  或 Gaussian: x = x + randn_like(x) * dropout（仅训练期间）
输出: [B, C=128, H=4, W=4]
```

**正向传播方程**:
```
x' = Up(Down(x))  — rank-r 瓶颈
x'' = Dropout2d(x')  — 如果 p > 0（仅训练时）
```

**理论作用**: 秩瓶颈（通常 r=8）强制将编码器的 128 通道特征表示通过狭窄的 8 通道通道，丢弃信息。解码器必须从压缩后的特征中重建。这阻止了生成器学习近似恒等捷径。对于内类（训练）数字，网络学习到保留类别身份的压缩表示。对于外类数字，压缩会从结构上扭曲重建，打破"复印机效应"。

### 4.4 损失函数

#### 基线 ALOCC

| 损失 | 方程 | 目标 |
|------|------|------|
| D 真实 | `BCE(D(x), 1)` | D 学习识别真实内类图像 |
| D 假 | `BCE(D(G(z)), 0)` | D 学习拒绝 G 的输出 |
| G 对抗 | `BCE(D(G(z)), 1)` | G 学习欺骗 D |
| G 精炼 | `BCE_with_logits(G(z), z)` | G 学习去噪（预测加噪输入） |

**总计**: `L_D = L_real + L_fake`  
**总计**: `L_G = L_adv + r_alpha × L_refine`

#### S1D ALOCC_LOSS

| 损失 | 方程 | 目标 |
|------|------|------|
| D 真实（内） | `BCE(D(x_in), 1)` | 与基线相同 |
| D 假（内） | `BCE(D(G(z_in)), 0)` | 与基线相同 |
| D 外类原始 | `BCE(D(x_out), 0)` | D 学习 x_out 是异常的 |
| D 外类精炼 | `BCE(D(G(z_out)), 0)` | D 学习 R(x_out) 也是异常的 |
| G 对抗 | `BCE(D(G(z_in)), 1)` | 与基线相同 |
| G 精炼 | `MSE(G(z_in), x_in)` | 使用 MSELoss 代替 BCE（注意与基线的区别） |
| G 外类扭曲 | `ReLU(margin − |G(z_out)−x_out|_L1)` | Hinge 损失：保持外类重建质量差 |

**总计**: `L_D = (L_real + L_fake) × (1−d_scale) + 0.5×(L_out_raw + L_out_refined) × d_scale`  
**总计**: `L_G = L_adv + r_alpha × L_refine + g_scale × L_distort`

**与基线的关键区别**:
1. D 显式看到外类图像（原始和精炼版本皆有）
2. G 精炼使用 MSE 而非 BCE
3. G 有一个扭曲分支，惩罚对外类样本的优良重建

---

## 5. 可重复性协议

### 5.1 环境与依赖

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows（开发）, Linux（部署） |
| Python | 3.10+ |
| PyTorch | 2.x 含 CUDA 支持 |
| GPU | 需要（单 GPU 足够） |
| Python 解释器路径 | `D:\Trae_coding\ALLOC\ALOCC-master\.venv\Scripts\python.exe` |
| 关键包 | torch, numpy, scipy, scikit-learn, piq, Pillow, matplotlib, tqdm |

**依赖文件**: `D:\Trae_coding\ALLOC\ALOCC-master\requirements.txt` 或 `pyproject.toml` — 不要手改。

### 5.2 实验启动命令

#### 基线 B（PyTorch 默认） — 50 runs

```powershell
# 对每个数字 c 从 0 到 9，每个种子 s in {42,43,44,45,46}:
python run_paper_mnist_figure6_7.py `
  --output-dir baseline_b_default_20ep/d{c}_s{s} `
  --variant alocc `
  --specific {c} `
  --epochs 20 --train-count 10000 --batch-size 128 `
  --selection-strategy raw_auc `
  --selection-epoch-start 1 --selection-epoch-end 20 `
  --selection-min-auc 0.60 `
  --seed {s}
```

#### S1D — 50 runs

```powershell
# 对每个数字 c 从 0 到 9，每个种子 s in {42,1337,2026,7,123}:
python run_paper_mnist_figure6_7.py `
  --output-dir s1d_20ep_c{c}_seed{s}_oracle `
  --variant alocc_loss `
  --specific {c} `
  --epochs 20 --train-count 4096 --batch-size 64 `
  --bottleneck-rank 8 --bottleneck-dropout 0.3 `
  --d-outclass-loss-scale 0.1 `
  --g-outclass-distortion-scale 0.3 `
  --g-outclass-distortion-margin 0.6 `
  --out-per-class-count 300 `
  --selection-strategy raw_auc `
  --selection-epoch-start 1 --selection-epoch-end 20 `
  --selection-min-auc 0.60 `
  --seed {s}
```

**数字 7 覆盖**（仅 S1D）: `--bottleneck-rank 4 --bottleneck-dropout 0.5`

### 5.3 随机种子策略

| 配置 | 种子 | 说明 |
|------|------|------|
| 基线 B | 42, 43, 44, 45, 46 | 5 个种子 × 10 个数字 = 50 runs |
| S1D | 42, 1337, 2026, 7, 123 | 5 个种子 × 10 个数字 = 50 runs；种子已固化；不要更改 |
| 种子应用 | `set_random_seed()` → `torch.manual_seed(seed)` + `numpy.random.seed(seed)` | 在模型初始化前应用，确保确定性权重初始化 |
| **未控制** | CUDA 卷积非确定性 | 确定性模式**未**启用（性能代价）；接受的方差 |

### 5.4 统计方法

| 方面 | 方法 |
|------|------|
| 集中趋势 | 50 runs 的算术平均值 |
| 离散度 | 标准差（±） |
| 对比指标 | 平均提升：(S1D_mean − BL_mean) / BL_mean × 100% |
| 逐数字分析 | 10 个数字各自的 mean ± std |
| 显著性 | 未显式检验；50-run 均值视为趋势证据 |
| 异常值处理 | 无 — 包含所有 50 个 runs；不剔除任何 run |
| 选模偏差说明 | Oracle 选模对 BL-B 的"美化"效果大于对 S1D（BL-B 逐 epoch 方差更大）；逐 epoch 平均提供制衡 |

### 5.5 输出结构

```
{output-dir}/
├── pipeline_summary.json       # 实验元数据（来自 run_paper_mnist_figure6_7.py）
├── figure7_scores.json         # 逐样本 D(X) 和 D(R(X)) 分数
├── triplets/                   # 原始 / 加噪 / 生成图像导出
│   ├── normal_*.png
│   └── abnormal_*.png
└── experiment/
    ├── summary.json             # 完整实验记录（来自 mnist_experiment_runner.py）
    ├── best.pth                 # Oracle 选中的检查点
    ├── {epoch}.pth              # 逐 epoch 检查点 (1..20)
    └── debug.log                # 训练损失轨迹
```

---

## 6. 损失配置对比

| 超参数 | 基线 ALOCC | S1D ALOCC_LOSS | 影响 |
|--------|------------|----------------|------|
| `r_alpha` | 0.2 | 0.2 | G 中精炼损失的权重 |
| `d_outclass_loss_scale` | 无 | 0.1 | D 中外类分支的权重 |
| `g_outclass_distortion_scale` | 无 | 0.3 | G 中外类扭曲的权重 |
| `g_outclass_distortion_margin` | 无 | 0.6 | Hinge 边界：将 R(外) 的 L1 推至此值以下 |
| `out_per_class_count` | 无 | 300 | 每类外类训练样本数 |
| 精炼损失类型 | BCE-with-logits（基于加噪） | MSE（基于干净） | G 精炼使用不同的损失函数 |

---

## 7. 指标计算详情

### 7.1 分类指标（基于 D(R(X)) 分数）

```
流程:
  1. 对每个测试 batch: 计算 R = G(noisy_X), 分数 = sigmoid(D(R))
  2. 聚合所有分数 (y_score) 和真实标签 (y_true: 内类为 1, 外类为 0)
  3. ROC 曲线: roc_curve(y_true, y_score) → fpr, tpr, thresholds
  4. F1: 在阈值上最大化 2TP/(2TP+FP+FN)
  5. ACC: 在最佳 F1 阈值下的准确率
  6. EER: brentq 求根 (1 − tpr − fpr 插值)
  7. AUC: roc_auc_score(y_true, y_score)
```

**说明**: 项目区分 `auc` / `raw_auc`:
- `auc` / `refined_auc` → **D(R(X))** 分数的 AUC（"精炼后"决策）
- `raw_auc` → **D(X)** 分数的 AUC（"原始"决策，用于 Oracle 选模）

### 7.2 图像质量指标

| 指标 | 工具 | 范围 | 解释 |
|------|------|------|------|
| SSIM | `piq.ssim`（kernel_size=7, downsample=False） | [0, 1] | 1 = 完全相同；逐样本计算 |
| VIF | 自定义多尺度 VIF（4 级, 5×5 高斯窗口） | [0, ∞) | 越高 = 保留信息越多 |
| GMSD | `piq.multi_scale_gmsd` | [0, ∞) | 越低 = 质量越好 |

所有质量指标在 **[0, 1] 图像**上计算（从 [-1, +1] 反归一化）。

### 7.3 衍生指标

```
ssim_gap     = ssim_ic − ssim_oc          （越高 = 分离越好）
vif_gap      = vif_ic − vif_oc
gmsd_gap     = gmsd_oc − gmsd_ic
score_gap    = score_in_mean − score_out_mean  （越高 = 经 R 后分离越好）
raw_score_gap = raw_score_in_mean − raw_score_out_mean  （经 R 前分离度）
score_gap_gain = score_gap − raw_score_gap  （R 带来的改进）
auc_gain     = refined_auc − raw_auc
```

---

## 8. 已知局限与注意事项

1. **数字 1 反转**：S1D 在数字 1 上表现低于基线（S1D AUC 0.964 vs BL 0.992）。最可能的原因：外类扭曲梯度（`distortion_scale=0.3`）压倒内类精炼梯度（`r_alpha=0.2`），对于笔画细的数字 1，将 G 拉离有用的内类重建方向。

2. **GPU 非确定性**：未禁用 CUDA 卷积非确定性。相同种子的 runs 在不同 GPU 架构/驱动下可能产生略有差异的结果。

3. **Oracle 选模偏差**：按 raw_auc 选出 20 个 epoch 中的最佳值，会使得报告 AUC 高于任何单个 epoch 的基线。这影响两种条件，但对方差更大的条件（基线）更有利。

4. **统计推断**：未执行正式的假设检验（t 检验、bootstrap）。所有论断基于 50 runs 的均值比较。

5. **S1D 与 BL-B 的 train_count 不匹配**：S1D 使用 `train_count=4096` 而 BL-B 使用 `train_count=10000`（自动上限为类全量）。这是历史约定（S1D 在 BL-B 切换前就已存在），并未受控。该差异有利于 BL-B（更多训练数据）。

---

## 9. 源代码交叉引用

| 组件 | 主文件 | 行号 |
|------|--------|------|
| 生成器架构 | `model.py` | 103–168 |
| 判别器架构 | `model.py` | 170–197 |
| LowRankNoisyBottleneck | `model.py` | 76–100 |
| ALOCC 基类 | `model.py` | 199–340 |
| ALOCC._train（基线） | `model.py` | 263–339 |
| ALOCC_LOSS._train（S1D） | `model.py` | 392–491 |
| CosinePrototypeClassifier | `model.py` | 43–74 |
| MNIST 数据集 | `MNIST.py` | — |
| build_model 路由 | `mnist_experiment_runner.py` | 29–42 |
| build_data | `mnist_experiment_runner.py` | 45–81 |
| run_experiment | `mnist_experiment_runner.py` | 525–660 |
| evaluate_checkpoints | `mnist_experiment_runner.py` | 446–522 |
| Oracle 选模（_select_records） | `mnist_experiment_runner.py` | 296–443 |
| compute_paper_score_stats | `mnist_experiment_runner.py` | 121–181 |
| 指标计算 | `Metrics.py` | 89–157 |
| Figure 6/7 流水线 | `run_paper_mnist_figure6_7.py` | 105–274 |
| S1 注入补丁 | `_patches/s1_step1_model.py` | — |
| BL-B 启动脚本 | `_patches/run_baseline_b_default_20ep.ps1` | — |
| S1D 启动脚本 | `_patches/run_s1d_20ep_oracle.ps1` | — |
| 跨实验报告（表格） | `_tmp_per_epoch_table.py` | — |
| 跨实验对比 | `_tmp_b_default_report.py` | — |
| 逐 epoch 绘图 | `_patches/b4_per_epoch_plots.py` | — |

---

## 10. ADR 索引（关键决策）

| ADR | 标题 | 文件引用 |
|-----|------|----------|
| ADR-007 | BL-A 的 RMSprop verbatim 默认值 | `model.py:243-244` |
| ADR-008 | S1 瓶颈注入机制 | `model.py:145-149` |
| ADR-011 | 主指标 = raw_auc, Oracle 选模 | `PROJECT_LOG.md §4` |
| ADR-012 | BL-B（PyTorch 默认）为当前基线 | `mnist_experiment_runner.py:243-244` |

---

*文档结束。生成日期 2026-05-01，基于实时源代码审计。*
