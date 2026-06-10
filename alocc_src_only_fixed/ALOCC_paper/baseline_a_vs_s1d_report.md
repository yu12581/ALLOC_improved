# Baseline A (v2026-04-27 Verbatim) vs 主线 S1D (Phase 3) 对比报告

> 生成时间：2026-04-28（§1 同选模口径修订：2026-04-30）· 数据源：`PROJECT_LOG.md` §6.6 / §2.10.7 (Baseline A) + §2.8.25 (S1D Phase 3) + `s1d_final.md` + `_tmp_h2h_baseline_vs_s1d.py`（同选模 H2H 脚本）· 架构基线：ADR-007 隔离原则修复后（§2.10.10）
>
> **核心结论**：在 ADR-006 标准锚点下、对两侧套用**同一份 redline 选模**消除选模混淆变量后，S1D（S1 瓶颈 + L1-hinge outlier distortion）相对忠实复现的 ALOCC 基线（v2026-04-27 verbatim 协议）在 50-run 矩阵上取得 **+0.237 raw_auc** 主效应（0.568 → 0.805），**全 10 类 S1D 严格优于 Baseline A**（最小增益 +0.035 / 最大 +0.423），同时把 outlier 重建 SSIM 从 0.722 压到 0.079（**−89.0%**），跨 seed std 由 0.270 收缩到 0.108（**−60%**）。Verbatim last_epoch 选模下增益更大（+0.307），见 §1.3。
>
> **指标主线声明**：本报告以 `raw_auc`（`sigmoid(D(G(x)))` 单分数 AUC，ALOCC 原版指标）为唯一 headline。`auc`（refined = α·D + (1−α)·SSIM 的混合分数 AUC）已自 ADR-011（2026-04-21，PROJECT_LOG §2.8.17）起降级为辅助诊断指标——主线 S1D 已废除 D-refined 路径，且 refined_auc 在 GAN 极性翻转时被 SSIM 兜底虚高、掩盖 baseline 失败模式。详见 §1.4。
>
> **稳健性声明**：§1.1 的核心结论（S1D vs BL 的 +0.237 raw_auc 增益）对 redline `ssim_oc` 阈值取值不敏感——τ_oc 在 [0.15, 0.90] 范围内 7 点扫描，Δ raw_auc 始终在 [+0.233, +0.237] 区间内（变动 0.004，远小于 raw_auc std=0.108）。两类算法的 ssim_oc 经验分布之间存在 [0.161, 0.466] 的"无 cell 区"，τ_oc 落在此区间的任何取值均得到相同的两侧主路径计数（BL 0/50, S1D 49/50）。完整 7 点扫描表见 §1.5。

---

## 0. 隔离性合规确认（ADR-007）

本次对比严格遵守 §2.10.10 的隔离修复：

| 实验 | 优化器开关 | RMSprop 参数 | 来源 | 备注 |
|:---|:---:|:---:|:---|:---|
| Baseline A (v2026-04-27) | `--tf-verbatim-rmsprop` **ON** | α=0.9, eps=1e-10 | TF1.15 `Sabokrou/ALOCC-CVPR2018/models.py` | verbatim opt-in 插件 |
| 主线 S1D (Phase 3) | flag **OFF**（默认） | α=0.99, eps=1e-8 | PyTorch RMSprop 默认 | 项目历史一致环境 |

**Bit-for-bit 验证**：`_verify_isolation_smoke.py --compare` 对 digit=1 seed=42 重跑 → `best_metrics` 15/15 keys 与 §6.6 既往 d1_s42 完全等同（auc=0.998850, paper_score=0.488859, best_epoch=10）。`_verify_s1d_isolation.py` 验证 `ALOCC_LOSS` 默认实例化拿到 α=0.99 / eps=1e-8。**两边数字均无需重跑，paper 主对比环境一致性已恢复**（§2.10.10 不变量条款）。

---

## 1. 核心指标对比表（50-run 矩阵 · mean ± std）

### 1.0 指标速查（看表前先读这一格）

> **`raw_auc`** = `roc_auc_score(y_true, sigmoid(D(G(X̃))))`——直接拿判别器对"重建样本"的 sigmoid 输出当 OCC 分数，算 ROC-AUC。这是 **ALOCC 论文 Figure 6/7 的原版评估口径**（D 单分数）。
>
> **本项目内的历史定位**：本指标在项目早期（§6.1 / §6.4，2026-04-19 之前）**未被采用**——彼时 headline 是 `auc`（refined = α·D + (1−α)·SSIM 的混合分数 AUC）。**ADR-011（PROJECT_LOG §2.8.17，2026-04-21）将主指标切换为 `raw_auc`**，故本报告中 `raw_auc` 属于"项目内新启用、文献内原版"的指标。完整迁移理由见 §1.4。
>
> | 字段 | 公式 | 范围 | 受极性翻转影响 | 角色 |
> |---|---|:-:|:-:|---|
> | **`raw_auc`** | AUC(`sigmoid(D(G(X̃)))`) | [0, 1] | ✅ 会，0.05 ↔ 0.95 双峰 | 🟢 **本报告 headline** |
> | `auc` (refined) | AUC(`α·D + (1−α)·SSIM`) | [0, 1] | ❌ 被 SSIM 兜底 | 🟡 §1.4 辅助诊断 |
> | `folded_raw_auc` | `max(raw_auc, 1−raw_auc)` | [0.5, 1] | 校正后 | 🟡 §2.3 极性诊断 |
> | `ssim_oc` | mean SSIM(R(X̃), X̃) on outliers | [0, 1] | — | 🟢 redline 选模硬约束 |
>
> **读表约定**：所有 `mean ± std` 形式的数字 = 50-run 矩阵（10 类 × 5 seed）的算术平均 ± 总体标准差（pstdev）。`raw_auc` 越高越好（除非翻转，详见 §2.3）；`ssim_oc` 越低越好（外类被扭得越狠，结构性可信度越高）。

### 1.1 同选模公平对比（main · 同 redline 套用两侧 · headline 口径）

为消除选模差异作为混淆变量，Baseline A 与主线 S1D **套用同一份 redline 选模函数**（earliest epoch with `ssim_oc ≤ 0.15 ∧ raw_auc ≥ 0.60`，子集空时 fallback 至 `distortion_score = max(ssim_gap,0) · max(refined_auc,0)` 排序，tie-break 与 `_select_records()` 一致），数据后处理脚本：`_tmp_h2h_baseline_vs_s1d.py`（直接读两侧 50 个 `summary.json` 的 `records[]`，不重跑训练）。

| 指标 | Baseline A · redline | **主线 S1D · redline** | 绝对增益 | 备注 |
|:---|:---:|:---:|:---:|:---|
| **`raw_auc`** | **0.568 ± 0.270** | **0.805 ± 0.108** | **+0.237** | std −60% |
| **逐类 `raw_auc` 胜率** | 0 / 10 | **10 / 10** | — | 全 10 类 S1D 严格领先 |
| **`ssim_oc`** | 0.722 | **0.079 ± 0.029** | **−0.643** | −89.0% |
| **redline main-path 命中** | **0 / 50** | **48 / 50** | +96 pp | baseline 全程 fallback |
| min_auc=0.60 软回退率 | 24 / 50 | ≈0 | — | baseline 半数 cell `refined_auc<0.6` |

### 1.2 Per-class `raw_auc` 详表（5 seeds · mean ± std · 同 redline）

| digit | Baseline A · redline | **S1D · redline** | Δ raw_auc | Baseline ssim_oc | S1D ssim_oc |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0.693 ± 0.301 | **0.922 ± 0.087** | **+0.229** | 0.774 | 0.078 |
| 1 | 0.479 ± 0.313 | **0.895 ± 0.089** | **+0.416** | 0.585 | 0.054 |
| 2 | 0.552 ± 0.243 | **0.732 ± 0.094** | **+0.180** | 0.826 | 0.077 |
| 3 | 0.644 ± 0.199 | **0.778 ± 0.046** | **+0.134** | 0.728 | 0.093 |
| 4 | 0.401 ± 0.176 | **0.823 ± 0.063** | **+0.423** | 0.732 | 0.072 |
| 5 | 0.579 ± 0.124 | **0.791 ± 0.062** | **+0.212** | 0.783 | 0.101 |
| 6 | 0.573 ± 0.269 | **0.860 ± 0.088** | **+0.287** | 0.736 | 0.055 |
| 7 | 0.523 ± 0.322 | **0.787 ± 0.105** | **+0.264** | 0.711 | 0.096 |
| 8 | 0.624 ± 0.285 | **0.659 ± 0.066** | **+0.035** | 0.628 | 0.076 |
| 9 | 0.616 ± 0.265 | **0.805 ± 0.080** | **+0.189** | 0.711 | 0.084 |
| **AVG** | **0.568** | **0.805** | **+0.237** | 0.722 | 0.079 |

**结论**：在同 redline 选模下，**S1D 在全部 10 类的 `raw_auc` 上严格优于 Baseline A**，最小增益 +0.035（class 8，仍正向），最大 +0.423（class 4）。Baseline 在 50 个 cell 中**没有任何一个**能命中 redline 主路径（`ssim_oc` 全在 0.46–0.93 区间，远超 0.15 阈值），其在 §1.1 中报告的 0.568 全部来自 distortion fallback 的事后选择——baseline 缺少结构性扭曲机制是这一现象的根因（详见 §2.2）。

### 1.3 Verbatim 选模参考（baseline last_epoch · §6.6 既往报告 · 论文复现完整性）

为保留对 ALOCC 论文 D4-C 选模（`last_epoch`）的 verbatim 复现完整性，并存以下数字作为参考（注：选模口径不同于 §1.1）：

| 指标 | Baseline A · last_epoch (§6.6) | S1D · redline (§2.8.25) | 绝对增益 |
|:---|:---:|:---:|:---:|
| `raw_auc` | 0.498 ± 0.297 | 0.805 ± 0.109 | +0.307 |
| `ssim_oc` | 0.923 ± 0.021 | 0.079 ± 0.029 | −0.844 |
| `ssim_gap` | +0.022 ± 0.026 | +0.126 ± 0.044 | +0.104 |
| `ssim_ic` | 0.945 ± 0.011 | ~0.205 | −0.740 |
| clean-pass / 50 | — *(无 redline 门槛)* | 48 / 50 (96%) | — |

§1.3 数字（+0.307）大于 §1.1（+0.237），差额 +0.070 来自 baseline 在 ep10 已退化为通用复印机（详见 §2.1，O2）。**本报告 headline 以 §1.1 的 +0.237（同选模口径）为正文主论断**；§1.3 仅供 verbatim 论文复现章节引用。

### 1.4 辅助指标与方法学声明（不进 paper headline）

| 指标 | Baseline A | S1D | 备注 |
|:---|:---:|:---:|:---|
| `auc` (refined) | 0.502 ± 0.294 | — | S1D 主线无 D-refined 路径（§2.2） |
| `auc_gain` | +0.004 ± 0.012 | — | refinement step 对 baseline 决策无贡献（O2） |
| `paper_score` | 0.433 ± 0.144 | — | redline 与 paper-window 选模不同源，不可对位 |

**指标主线迁移声明**：早期项目（§6.1 / §6.4，2026-04-19 之前）以 `auc`（refined = α·D(G(x)) + (1−α)·SSIM(x, G(x)) 的混合分数 AUC）为主指标。**ADR-011（2026-04-21，PROJECT_LOG §2.8.17）将主指标切换为 `raw_auc`**，三条理由：
1. `raw_auc` 是 ALOCC 论文标准（D 输出单分数 AUC，原作者 Figure 6/7 评估口径）；
2. `refined_auc` 在 GAN 极性翻转时被 SSIM 兜底虚高（baseline `refined_auc` 0.502 vs `raw_auc` 0.498，几乎同源；但折叠后 `raw_auc` 0.751 揭示真实判别力被极性翻转隐藏，refined 无此诊断价值）；
3. redline 选模的硬约束就是 `raw_auc ≥ 0.60`——**主指标必须与选模约束同源**，否则选模和报告构成两层不一致变量。

S1D 主线在 §2.8.23–§2.8.25 重设损失结构后，refinement step 已不再生成 `refined_auc`（保留字段但等于 `raw_auc`），故 §1.1 / §1.2 主表完全在 `raw_auc` 维度对位，无虚高指标污染。

### 1.5 Redline 阈值敏感性分析（稳健性证据）

为回应"`τ_oc = 0.15` 是否对 S1D 友好"这类潜在质疑，对 ssim_oc 上限做 7 点扫描（raw_auc 下限固定 0.60），同 redline 函数同时套用两侧 50 cell。数据后处理脚本：`_tmp_threshold_sensitivity.py`（仓库根，纯读 `summary.json` 后处理，不重训）。

| τ_oc | BL 主路径命中 | BL `raw_auc` (mean ± std) | S1D 主路径命中 | S1D `raw_auc` (mean ± std) | **Δ raw_auc** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **0.15** ← §1.1 现行 | **0 / 50** | 0.568 ± 0.270 | **48 / 50** | 0.805 ± 0.108 | **+0.237** |
| 0.20 | 0 / 50 | 0.568 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.235 |
| 0.30 | 0 / 50 | 0.568 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.235 |
| 0.40 | 0 / 50 | 0.568 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.235 |
| 0.50 | 1 / 50 | 0.568 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.235 |
| 0.70 | 8 / 50 | 0.568 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.235 |
| 0.90 | 25 / 50 | 0.570 ± 0.270 | 49 / 50 | 0.803 ± 0.108 | +0.233 |

**两侧 ssim_oc 经验分布（选模选中 epoch 的实测分位数 · n=50/arm）**：

| 统计 | min | p10 | median | p90 | max |
|:---|:-:|:-:|:-:|:-:|:-:|
| Baseline A | **0.466** | 0.524 | 0.744 | 0.870 | 0.924 |
| S1D | **0.033** | 0.046 | 0.075 | 0.119 | **0.161** |

**结论**：
1. **Δ raw_auc 对阈值近乎不变**：τ_oc 在 [0.15, 0.90] 范围内扫描，Δ ∈ [+0.233, +0.237]，变动幅度 0.004，**远小于 raw_auc std (0.108)**——主对比结论结构性稳健，与阈值具体取值无关。
2. **`τ_oc = 0.15` 不是对 S1D 友好的人为低界**：S1D 50 cell 的 ssim_oc 全分布在 [0.033, 0.161] 内，max 0.161 已超出 0.15——若对 S1D 友好理应取 ≥ 0.20。0.15 阈值的来源是 §2.5「质量红线」（PROJECT_LOG 2026-04-19 立档，**早于全部 S1D 实验**），属先验质量门槛而非事后调参。
3. **Baseline 的 ssim_oc 经验下界是 0.466**：阈值要放宽到 ≥ 0.50 才有 1 个 baseline cell 能进主路径，0.70 才有 8 个，0.90 才有 25 个。**baseline 与 S1D 的 ssim_oc 分布之间存在 [0.161, 0.466] 的"无 cell 区"——τ_oc 落在此区间内任何取值都得到相同主路径计数（BL 0/50, S1D 49/50）**。这印证两类算法在结构可信度上属于不同 regime，不是连续光谱上的渐变。
4. **行业基准对照**：Vanilla 重建型 OCC（无 distortion 机制，包括本报告 baseline）的 ssim_oc 经验范围 0.4–0.9（与本表 baseline 分布一致）；S1D 的 0.03–0.16 区间是 distortion 机制的直接结构产出，**而非阈值人为设置的产物**。换言之，**`τ_oc = 0.15` 反映的是"S1D 实测能稳定达到的水平"，不是"我们想要的水平"**。

---

## 2. 定性分析与解读

### 2.1 为什么 Baseline A `auc≈0.50` 且 `ssim_oc≈0.92`？—— 复印机效应（§2.10.8 O1+O2）

Baseline A 严格执行原作者 TF1.15 源码协议四件套：

- **D1-B**：refinement loss = `BCEWithLogits(R(X̃), X̃_noisy)`（拟合**噪声目标**而非干净 inlier）；
- **D2-B**：RMSprop α=0.9 / eps=1e-10（TF1.15 verbatim）；
- **D3-B**：源码级直译 TF1.15 `train()` 主循环；
- **D4-C**：`last_epoch` 选模（窗口 [10,10] 强制收敛于第 10 epoch）。

D1-B 的几何后果是：R 被训练成**复制噪声目标**的通用复印机，inlier / outlier 都被高保真复制（`ssim_ic=0.945`、`ssim_oc=0.923`，gap 仅 +0.022）。这恰好印证论文 §4.4 的 *overtraining* 警告——R 退化后，分类决策只能由 D 单独完成，而 D 并未受到外类样本的显式压力，于是 `auc≈0.50`、`auc_gain≈0`（refinement 步骤对决策毫无贡献）。

> **§2.10.8 原文**："`ssim_oc≈0.92` 揭示 BCE-on-noisy 的本质——D1-B 把 R 训练成『复制噪声目标』而不是『重建干净 inlier』。这恰恰回应论文 §4.4 的 overtraining 警告——R 退化为通用复印机后，inlier/outlier 都被高保真复制，分类决策只能由 D 单独完成（即便 D 也不强）。"



### 2.2 S1D 如何把 `ssim_oc` 压到 0.079、`raw_auc` 抬到 0.805 —— 结构性扭曲

S1D 在 Baseline A 之上叠加**三个**结构改造（其余锚点对齐 ADR-006）：

1. **S1 瓶颈**（`bottleneck_rank=8`，class 7 用 `r=4`；`bottleneck_dropout=0.3/0.5`）：在 R 的潜在层注入低秩 + dropout 投影，强制 R 只能表达 inlier 子流形上的内容，外类样本被迫沿 inlier 方向被"投影 + 失真"；
2. **L1-hinge distortion 损失**（`g_outclass_distortion_scale=0.3, margin=0.6`）：对外类样本主动施加 `max(0, m − ‖R(X̃) − X̃‖₁)` 推力，让 R 学会"看到 outlier 就扭坏"；
3. **D 端外类项**（`d_outclass_loss_scale=0.1`）：给 outlier 显式打 `target=0` 标签，与 (2) 协同**锁定 GAN 判别极性**。

几何效果：`ssim_oc 0.923 → 0.079`（**外类重建从复印降到结构破坏**），`ssim_ic 0.945 → 0.205`（inlier 也被压低，但仍显著高于 outlier），最终 `ssim_gap +0.022 → +0.126`。这个 gap 直接驱动 `raw_auc 0.498 → 0.805`，并通过 redline 选模（`ssim_oc≤0.15 AND raw_auc≥0.60`）淘汰 2/50 的退化解，得到 clean-pass 48/50 (96%)。

> §2.10.8 O2 原文："**RM-1 / S1D 系列改法的存在意义**就是通过 S1 瓶颈强制 R 学习 inlier 子流形 + 通过 distortion 损失主动扭坏 outlier，把 ALOCC 从一个『几乎不工作的』算法改造成『实用的』OCC 检测器。"

### 2.3 GAN 极性翻转 —— Baseline A 的二态性 vs S1D 的方向锁定（§2.10.8 O3）

vanilla ALOCC（Baseline A）**没有任何机制锁定 D 的判别方向**——D 既可以学成"real=inlier"（auc≈0.95），也可以学成"real=outlier"（auc≈0.05），两种都满足 minimax 均衡。在 5 seed × 10 类 = 50 run 内部，这种二态性体现为：

- **class=1**：`auc 0.476 ± 0.492`（标准差几乎与均值同量级，典型 bimodal 分布）；
- **class=4 / 7**：`raw_auc < 0.50`（均值已经被翻转簇拉到反向）；
- **全局 std=0.297**：远高于任何单个 seed 内部的 epoch 间方差。

S1D 的 distortion 损失通过显式给 outlier 打 0 标签**锁定方向**，于是：

- **全局 std=0.109**（收缩 2.7×）；
- **每类 std≤0.117**（最大 std 出现在 class=7，0.117）；
- **每类均值≥0.66**（最低 class=8，0.660；无任何反向翻转）。

这是 v2026-04-27 baseline → S1D 主对比中的**第二个**强故事点（除了均值差之外的方差收缩故事）。注：本节引用的 std=0.297 / 0.109 来自 §1.3 verbatim 选模协议（last_epoch）；§1.1 同 redline 选模下相应数字为 std=0.270 / 0.108（收缩 2.5×），结论方向一致。

---

## 3. 视觉证据（typical reconstructions）

### 3.1 总览：3×2 对比主图

下图为 §2.8.x 期间生成的 Baseline (no-S1, project-default 环境) vs S1D 三类成对对比主图，**直接展示 R 对外类的复制 vs 扭曲行为差异**：

![MASTER 3x2 BL vs S1D](s1_compare_sheets/MASTER_3x2.png)

> 路径：`ALOCC_paper/s1_compare_sheets/MASTER_3x2.png` · 上 3 行 = Baseline (S1=OFF)；下 3 行 = S1D (S1=ON + distortion)；左列 normal, 右列 abnormal。
>
> 注：本主图来源于 §2.8.7 起的 S1 ablation 配对（`s1_c{0,1,2,6}_off_redline` vs S1D `redline`），是项目历史上最早的 BL vs S1D 视觉对照；其 baseline 端使用项目默认环境（α=0.99/eps=1e-8）而非 v2026-04-27 verbatim。**视觉差异由 S1+distortion 结构差异主导**，与优化器/refinement 协议无关——v2026-04-27 verbatim 端的视觉特征更"复印"（`ssim_oc` 从历史 0.79 进一步推到 0.92），即下方 §3.2 中 BL 端的"高保真复制"现象在 verbatim 协议下只会更显著。

### 3.2 Baseline A 端（"复印机"行为）

`baselines_cuda/A/triplets/`（class 1 inlier · D1-A 历史协议；视觉模式与 v2026-04-27 verbatim 同质，区别仅在 refinement loss 和优化器，`ssim_oc` 同向更高）：

| Normal triplet（inlier） | Abnormal triplet（outlier） |
|:---:|:---:|
| ![BL-A normal](baselines_cuda/A/triplets/normal_triplets.png) | ![BL-A abnormal](baselines_cuda/A/triplets/abnormal_triplets.png) |
| 路径：`ALOCC_paper/baselines_cuda/A/triplets/normal_triplets.png` | 路径：`ALOCC_paper/baselines_cuda/A/triplets/abnormal_triplets.png` |

**视觉特征**：每张三元组依次为 [`X`, `X̃` (noisy), `R(X̃)`]。inlier 与 outlier **均被高保真复制**——R 几乎逐像素还原噪声后输入，证实 §2.10.8 O2 的"通用复印机"判断。outlier (右图) 中其他数字的笔画结构在 R 输出中清晰可见。

### 3.3 S1D 端（"结构性扭曲"行为）

S1D 端典型样本（class=0, seed=42, redline 选模）已在 §3.1 主图下半 3 行直接呈现。原始路径：

- `ALOCC_paper/s1d_final_c0_seed42_redline/triplets/normal_triplets.png`
- `ALOCC_paper/s1d_final_c0_seed42_redline/triplets/abnormal_triplets.png`

**视觉特征**：inlier 仍被 R 大致还原（虽 `ssim_ic` 已从 0.95 降至 0.21，但形状轮廓保留），而 outlier 被 R 明确"扭坏"——笔画断裂、形状坍缩成 inlier 流形上的接近形状（典型表现：把 outlier 数字"硬掰"成 inlier 数字的混淆形）。这正是 distortion loss + S1 瓶颈联合作用的几何外显。

### 3.4 配对对比（class 1 / 2 / 6 · BL vs S1D）

每类 normal + abnormal 各一张，左列 BL（S1=OFF），右列 S1D（S1=ON）：

| Class | Normal (inlier) | Abnormal (outlier) |
|:-:|:-:|:-:|
| 1 | ![c1 normal](s1_compare_sheets/compare_c1_normal.png) | ![c1 abnormal](s1_compare_sheets/compare_c1_abnormal.png) |
| 2 | ![c2 normal](s1_compare_sheets/compare_c2_normal.png) | ![c2 abnormal](s1_compare_sheets/compare_c2_abnormal.png) |
| 6 | ![c6 normal](s1_compare_sheets/compare_c6_normal.png) | ![c6 abnormal](s1_compare_sheets/compare_c6_abnormal.png) |

> 路径：`ALOCC_paper/s1_compare_sheets/compare_c{1,2,6}_{normal,abnormal}.png`

---

## 4. 一句话结论（可直接进 paper 摘要）

> 在 ADR-006 标准锚点（`epochs=10, train_count=4096, batch_size=64, noise_std=0.31, r_alpha=0.2, lr=0.002`）下、对 Baseline A 与主线 S1D **套用同一份 redline 选模函数**（`ssim_oc≤0.15 ∧ raw_auc≥0.60`，fallback 至 distortion-score 排序）消除选模混淆变量后，S1D（S1 瓶颈 r=8/4 + L1-hinge outlier distortion m=0.6/s=0.3 + D 端 outclass scale=0.1）相对于忠实复现的 ALOCC 基线（v2026-04-27 verbatim 协议：BCE-on-noisy refinement + TF1.15 RMSprop α=0.9/eps=1e-10）在 50-run 矩阵上取得 **+0.237 `raw_auc`** 主效应（0.568 → 0.805），**全 10 类 S1D 严格优于 Baseline A**（最小增益 +0.035 / 最大 +0.423），同时把 outlier 重建 SSIM 从 0.722 压到 0.079（**−89.0%**），跨 seed std 由 0.270 收缩到 0.108（**−60%**），redline 主路径命中率从 0/50 跃升至 48/50（**+96 pp**）。Verbatim last_epoch 选模口径（§1.3）下增益更大（+0.307），作为论文 D4-C 复现完整性参考。

---

## 5. 数据溯源与可复现性

| 资产 | 路径 |
|:---|:---|
| Baseline A 50-run summaries | `D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27\d{0..9}_s{42..46}\summary.json` |
| Baseline A 聚合 | `baseline_a_v2026_04_27\_aggregate.{json,md}` |
| Baseline A 调度 | `ALOCC_paper/_patches/run_baseline_a_50.ps1`（含 `--tf-verbatim-rmsprop` opt-in flag） |
| S1D Phase 3 报告 | `ALOCC_paper/s1d_final.md`（per-class + per-cell 详表） |
| S1D triplets / 主图 | `ALOCC_paper/s1d_final_c{0..9}_seed42_redline/triplets/`、`s1_compare_sheets/MASTER_3x2.png` |
| **同选模 H2H 后处理脚本（§1.1 / §1.2 数据源）** | `_tmp_h2h_baseline_vs_s1d.py`（仓库根；纯数据后处理，不重跑训练）|
| **Baseline 单独 redline 重选脚本（§1.1 行级数据）** | `_tmp_redline_select_baseline.py`（仓库根） |
| **Redline 阈值敏感性扫描脚本（§1.5 数据源）** | `_tmp_threshold_sensitivity.py`（仓库根；7 点 τ_oc 扫描 + 双侧 ssim_oc 经验分位数）|
| 隔离验证脚本 | `ALOCC_paper/_patches/_verify_isolation_smoke.py` / `_verify_s1d_isolation.py` |
| ADR-007 修复落地说明 | `PROJECT_LOG.md` §2.10.10 + 2026-04-28 Changelog |
| 指标主线迁移决策（`auc` → `raw_auc`） | ADR-011（PROJECT_LOG §2.8.17，2026-04-21）|

**架构声明**：根据 ADR-007 隔离原则（§2.10.10 落地于 2026-04-28），verbatim 协议现在以"可选插件"形式存在——CLI flag `--tf-verbatim-rmsprop` 显式 opt-in，默认主线行为（α=0.99/eps=1e-8）保持与 §2.8.x 全部历史 PyTorch 数字一致。本对比表中 Baseline A 与 S1D 的环境差异 **仅来自结构改造（S1 + distortion + D-外类）**，优化器/refinement 协议差异作为"verbatim 复现 vs 项目主线"的副产物被显式记录但不构成混淆变量。
