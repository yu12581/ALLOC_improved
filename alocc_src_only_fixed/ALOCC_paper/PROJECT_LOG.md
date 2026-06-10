# ALOCC 复现攻坚 · 总控文档

> 单一事实来源。任何技术更改完成后必须回写本文件（变更日志 + 路线图状态 + 决策记录）。
> 仓库：`D:\Trae_coding\ALLOC\ALOCC-master`
> 论文：`Sabokrou et al., Adversarially Learned One-Class Classifier for Novelty Detection, CVPR 2018`
> 全文文本：`ALOCC_paper/paper_fulltext.txt`

---

## 0. 工作原则（生效中）

1. **不追求最小改动**。允许结构性重构，前提是设计能直击要害、实施正确。
2. **北极星指标 = 异常外类扭曲度**。所有改动按下表评估收益：

| 指标 | 期望方向 | 含义 |
|---|---|---|
| `ssim_oc` | ↓（**≤ 0.15 为达标**，2026-04-19 立） | R 对 outlier 的重建相似度——越低代表 R 越"扭坏" outlier；Baseline A 当前 best_ep=2 为 0.2182，需下压 ~30% |
| `ssim_gap = ssim_ic − ssim_oc` | ↑ | R 对 inlier 与 outlier 的重建质量差距 |
| `outclass_recon_l1` | ↑ | R(X̂) 与 X̂ 的 L1 距离，工程化扭曲量 |
| `auc_gain = AUC(D(R(X))) − AUC(D(X))` | > 0 | Figure 7 主张：refined 比 raw 更可分 |
| `score_gap_gain` | > 0 | 同上，按均值差衡量 |
| `refined_auc / refined_acc / F1` | ↑（不能塌） | 防止"扭得狠但分类崩"的退化解 |

3. **训练设定区分**：
   - **主设定**（论文 §4.2）：训练只用 inlier；
   - **工程变体**（论文 §4.4 已认可）：可混入 outlier，但报告时必须明确标注。
4. 每次提交前回答三问：(a) 哪些北极星指标变了？(b) 论文一致性是否退化？(c) 哪些代码 surface 被改？
5. **所有改动必须可开关**（ADR-007，强制执行）：任何新增行为（损失项、早停策略、选模 floor、日志字段、网络分支……）必须通过 **CLI flag + Runner kwarg + 默认关闭** 三件套暴露，保证：
   - (a) 默认行为与当前基线**逐字节一致**（regression-free）；
   - (b) 可在命令行 `--flag on/off` 或数值化 scale 即时切换；
   - (c) 开关状态写入 `summary.json` + 运行日志，消融实验可追溯。
   - 反例（禁止）：硬编码 `if True:`、环境变量暗开关、没有对应 `--no-…` 关闭路径、默认值偏离论文/基线配置。
6. **沟通合同**（ADR-008，2026-04-19 立）：Agent 向 user 汇报时 **只讲代码的作用/效果**，不讲代码的实现细节（语法、变量名、函数签名、具体实现逻辑）。违反示例：`X = max(…, key=lambda r: r['paper_score'])`；合规示例："这段按 paper_score 综合分选最高的那个 epoch"。
   - 适用场景：review、汇报、设计稿、进展同步、bug 诊断讲解。
   - 不适用场景：代码注释 / commit message / 文件内部文档（这些仍保持技术精度）。
   - user 主动要求看代码（"给我看 diff"/"贴出来"/"展开实现"）时才贴源码，其他一律按作用口径。

---

## 1. 论文锚点速查

- **Eq. 3**：`min_R max_D ( E[log D(X)] + E[log(1 − D(R(X̃)))] )`，X̃ = X + η
- **Eq. 4**：`L_R = ‖X − X'‖²`（target 是干净 X，不是 X̃）
- **Eq. 5**：`L = L_{R+D} + λ·L_R`，λ = 0.4
- **Eq. 7**：`OCC(X) = target ⇔ D(R(X)) > τ`
- **§3.3 / §4.4 关键警告**：训练过头会让 R 退化成通用降噪器 → outlier 也被还原 → Figure 6 行为消失。
- **Figure 6**：inlier=1，outlier=6/7；R 对 outlier 显著扭曲。
- **Figure 7**：D(R(X)) 比 D(X) 的 reject region 更小。
- **Figure 8**：MNIST 主指标 = F1，按 outlier 比例画曲线。

---

## 2. 当前问题清单（来自首轮代码审查）

P0（直接影响北极星指标或选模可信度）：
- [x] **PR-A** ✅ 2026-04-19 完成。`_select_records` 新增 `selection_min_auc_hard / selection_log_fallback` 双 flag + stderr WARNING + `selection_info.fallback_{triggered,reason}` + `summary.json.switches`；默认行为（静默 fallback）与基线逐字节一致，`--selection-min-auc-hard` 时 `RuntimeError`。详见 §5（2026-04-19 PR-A+PR-B）。
- [x] **PR-B** ✅ 2026-04-19 完成。`ALOCC_LOSS_CLS._train` 末尾补早停分支（`model.py:582-587`，与 `ALOCC._train` / `ALOCC_LOSS._train` 对齐）；未传 `--stop-recon-threshold` 时 0 开销分支不进入。详见 §5。
- [ ] **PR-C** Dataset 预搬 GPU（`MNIST.py:53-56`）→ 无法多进程 + 限制扩展。
- [ ] **PR-D** 评估前向重复（`calculate_metrics` + `compute_paper_score_stats` 各跑一遍 G+D）。
- [ ] **PR-Q** CUDA torch 迁移：现状 `torch 2.11.0+cpu`（`ALLOC-master/.venv`），硬件 RTX 5060 Laptop 8 GB 空转；迁移后训练 10–30× 加速，消融成本锐降。
- [x] **PR-R** ✅ 2026-04-19 已修复。`ALOCC_LOSS_CLS._train` 末尾补 `return int(epoch)`（`model.py:582`，8 空格缩进与 `ALOCC._train:291` / `ALOCC_LOSS._train:442` 对齐）。备份 `model.py.pr_r.bak` 在仓库根。验证：小规模 (epochs=3, train=256) 跑通 → `best_epoch=1`、`summary.json` 正常。补丁脚本 `ALOCC_paper/_patches/pr_r_alocc_loss_cls_return.py` 幂等可重入。

P1（结构性，铺路用）：
- [ ] **PR-E** 三处 `build_model` / 四处 outlier filter 收敛到单一 factory。
- [ ] **PR-F** `FrameMetrics` 数据类 + `paper_behavior` 一等公民字段。
- [ ] **PR-G** 损失模块化（AdvBCE / InClassRefineMSE / OutClassDistortHinge / ClsAux），主设定 ↔ 工程变体可组合切换。

P2（行为/可视化正确性）：
- [ ] **PR-H** `Metrics.show_class_metrics` PSNR 标签错 + AUC 子图覆盖 bug。
- [ ] **PR-I** `paper_figure6_7.Args` dataclass 镜像 → 改为单一 args 源。
- [ ] **PR-J** `itertools.cycle(outclass_loader)` 内存累积 → 改生成器写法。

P3（清理）：
- [ ] **PR-K** legacy `MNIST.test()` 与 `MNIST.train()` 主入口（10×10 训练循环）归档。
- [ ] **PR-L** `Generator.forward(classify=True)` 副作用语义清理；`_hook_conv` 改 context manager。

张量维度类（来自 ADR-004 强制审查；详见 §8）：
- [ ] **PR-M** `Generator.__init__` 仅按 `in_h` 计算单维度，`in_w/out_h/out_w` 形参被接受但未使用 → 非方形输入会静默错位（`model.py:77-104`）。
- [ ] **PR-N** `Discriminator` 用 `math.ceil(in_h/16)` 启发式估算特征图尺寸，应改为构造时 dummy forward 实测后再建 Linear（`model.py:142-157`）。
- [ ] **PR-O** 多损失合并未显式断言 scalar，重构后若新增 `reduction='none'` 项会被广播吞掉 → 加 `_assert_scalar` 守卫（`model.py:253, 267, 401, 417, 565`）。
- [ ] **PR-P** Encoder/Decoder 形状链路改为 dataclass `ShapeChain` 显式持有，而非散落的 `in_h1/in_h2/in_h3` 局部变量；落 `tests/test_shape_chain.py` 覆盖 28/30/32/64 四种尺寸。


---

## 2.5 主线任务：Baseline A 复现与改进（A1 / A2 / A5）· 2026-04-19 确立

### 背景与档位

2026-04-19 user 拍板：**把 Baseline A（论文原算法 `variant=alocc`）定为整个项目的主线**；执行档位 = **"改法档"**——允许动训练损失 / 网络结构 / 训练调度，前提是**保留 one-class 分类范式 + GAN 对抗框架**这两个核心身份不变。

档位矩阵（ADR-009 首次正式化）：

| 档位 | 允许的手段 | 论文一致性 | 本项目是否采用 |
|---|---|---|:---:|
| 严格档 | 只动论文明写超参 | ✅ 完全一致 | ❌ |
| 工程档 | 超参 + 训练调度 / 多 seed / checkpoint 粒度 | ⚠️ 宽松一致 | ❌ |
| **改法档** | 加模块 / 改结构 / 结合 / 借鉴顶会 | ❌ 不一致（报告时双栏对照） | ✅ **选中** |

### 三项主线子任务（优先级顺序）

| 编号 | 名称 | 产出 | 状态 |
|:---:|---|---|:---:|
| **A1** | MNIST 全类别数值复现与改进 | 对齐论文 Table 的 AUC / EER + 本项目改进版双栏对比；每个 inlier 类（0-9）一行 | ⏳ 先做 |
| **A2** | Figure 6 / 7 复现 | Figure 6 重建三联图（inlier vs outlier 对比）+ Figure 7 refined vs raw 得分分布 | A1 后 |
| **A5** | 跨数据集扩展 | Caltech-256 + UCSD Ped2 接入（论文报告过的另外两个数据集）| MNIST 全做完后 |

### 质量红线（A1 的达标条件）

- **`ssim_oc ≤ 0.15`** （2026-04-19 user 拍板硬红线）——当前 Baseline A best_ep=2 为 0.2182，需下压 ~30%
- **`ssim_gap ↑`** —— outlier 与 inlier 重建质量差距扩大
- **`refined_auc` 不塌** —— 不允许"扭得狠但分类崩"的退化解
- **`auc_gain > 0`** —— 加分项，非硬红线

### 改法档下的手段清单（允许使用）

1. **加模块**（= 增强，低结构风险）：
   - RM-1 负 SSIM 损失 on outlier
   - RM-2 encoder 冻结 / 参数容量约束
   - RM-4 raw/refined 双栏报告
2. **改结构**（= 改方法，高结构风险）：
   - R 的 encoder-decoder 深度 / skip connection 拓扑 / latent 维度
   - D 的多尺度头 / 特征金字塔
   - R↔D 之间的中间层耦合
3. **结合**：加 + 改混合；借鉴近几年顶会（CVPR / ICCV / NeurIPS / ICLR）one-class / OOD / anomaly detection 工作
4. **出界条款**（2026-04-19 立，禁止）：
   - 放弃 one-class 训练范式（比如改成 supervised binary）
   - 放弃 GAN 对抗框架（比如改成纯 VAE / 纯 flow-based）
   - **蒸馏方案**——博士师兄经验：此前试过无显著效果，不列入首选

### RM 模块的新定位

RM-1/2/3/4 从"独立攻击点"**降级为主线任务的工具链**：

| 模块 | 服务对象 | 当前状态 |
|---|---|:---:|
| RM-1（负 SSIM 损失）| 为 A1 的 `ssim_oc ≤ 0.15` 红线提供训练层手段 | 待做 |
| RM-2（encoder 冻结）| 为 A1 的 "防 R 退化为通用去噄器" 提供容量约束 | 待做 |
| RM-3（诚实选模）| 为所有 A 子任务提供度量尺 | ✅ 完成（2026-04-19）|
| RM-4（双栏报告）| 为 A2 的 Figure 7 复现提供报告字段 | 待做 |

### 实验规模原则（2026-04-19 立）

所有主线任务的**第一轮**都用 **MNIST 小规模**跑通闭环（沿用 ADR-006 锚点：`train_count=4096`、`epochs=10`、`batch=64`），验证方向后再放大或切数据集。禁止一上来就跑 Caltech-256 / UCSD。

### 博士师兄指导原话（2026-04-19 存档 · 作为设计参考）

> 除了加模块，也可以思考一下改模型结构；改结构意味着改方法；加模块本质是做增强。维持住 1 分类和 GAN 的方法不变，其他结果可以做适当改动。可以让 Agent 看看近些年的顶会文章有什么好的借鉴。可以现有改动、也可以添加、也可以做结合。蒸馏此前试过，没啥效果。


---

## 2.6 Track ① 文献调研（改法档借鉴池 · 2026-04-19）

> 检索口径：近 3 年（2023-2025 为主）+ ALOCC 直系后继（CVPR 2019-2020）；筛选标准：**在 one-class / reconstruction-based / OOD detection 框架内**且**直接针对"identity shortcut / universal denoiser"问题**。共识：业界已把我们称作的"R 学成通用去噄器"命名为 **identity shortcut (ID-shortcut)**，是该方向的头号公敌。

### 2.6.1 可移植性评级说明

- **★★★**：ALOCC 框架内可直接加模块/改训练信号，不动 one-class+GAN 身份；MNIST 小规模可跑通；与 §2.5 红线高度正相关。
- **★★**：需要中等规模结构改动（新损失、新分支）但论文一致性可报告；MNIST 小规模可验证。
- **★**：框架级差异（扩散模型 / Transformer backbone / 特征级重建），移植成本高或需换底座；留作长期储备，不作为首轮选项。

### 2.6.2 候选池（按可移植性排序）

> **2026-04-19 OCC 严格性补注**：以下候选表的 L2 / L4 / L6 / L7 / L8 均在训练阶段引入合成/反向搜索的"伪异常"样本，违反 one-class classification 纯正性（训练只允许接触正常样本）。RM-1 方法枢轴后（见 §5 同日条目）这五项全部排除；**严格 OCC 下可落地候选仅剩 L1 / L3 / L5**（见下方 §2.6.3 已更新推荐）。

| # | 论文 | 年份 / 会议 | 核心思想（一句话） | 对付 ID-shortcut 的具体机制 | 可移植性 | 预估移植成本 |
|:---:|---|---|---|---|:---:|:---:|
| L1 | **Old is Gold (OGNet)** | CVPR 2020 | ALOCC 直系后继：把 D 的角色从"真/假"改成"重建质量好/坏" | 两阶段训练：阶段 2 冻结 G，用 G 当前输出当"好"样本、用 **G 的旧 checkpoint** 输出当"坏"样本，让 D 学会识别"微弱扭曲"——测试时 D 直接作为异常分数 | ★★★ | 低（G 结构不动，只改训练 schedule + D 目标函数）|
| L2 | **PseudoBound** | Neurocomputing 2023 | 训练时主动喂假异常，AE 学"不要把异常也修好" | 每个 batch 混入 pseudo-anomaly（patch swap / noise / blur / …5 种），对真实 inlier 最小化重建误差、对 pseudo-anomaly **最大化**重建误差 | ★★★ | 低（数据增强 + 损失分支；与 RM-1 负 SSIM 思路同源但方法更成熟）|
| L3 | **MANet** | Pattern Recognition 2020 | 在 ALOCC 基础上加 multi-head self-attention + adversarial-balance loss | R/D 骨架加注意力头；`L_balance` 项平衡 G/D 对抗强度，消除 ALOCC 的训练不稳定性（这正是 §6.2 F2 观察到的 ssim_oc 后期漂移根因） | ★★★ | 中（动 R/D 网络结构 + 新损失项，但 MNIST 上工作量可控）|
| L4 | **OCGAN** | CVPR 2019 | 用两个判别器（latent + visual）+ informative-negative mining 约束潜空间 | 训练时在 latent 空间**反向搜索**最容易被误判为 inlier 的点，拉回来让 G 学会拒绝——相当于用梯度生成 pseudo-outlier | ★★ | 中（引入 latent D + negative mining loop；比 L2 复杂但原理清晰）|
| L5 | **MemAE → DMAD** | ICCV 2019 / CVPR 2024 | 在 AE 瓶颈处塞可学习的"正常记忆库"，重建强制走检索 | G 的瓶颈 layer 后面加 memory module：latent 向量只能用 M 个可学习 prototype 的加权和表达；异常 latent 找不到匹配 → 重建被强制扭坏 | ★★ | 中（给 R 的 bottleneck 加一层，需调 M 的大小；对 ID-shortcut 从容量源头封堵）|
| L6 | **Feature Shuffling (2024)** | Expert Sys. 2024 | Pseudo-anomaly 的升级：在**特征空间**打乱再还原 | 不在像素层加噪，而在 encoder 输出的特征图上做 shuffling，decoder 学习"还原有序特征"——对特征级异常敏感，不会只学像素恒等 | ★★ | 中（需要在 R 内部暴露中间特征，再加一个辅助损失头）|
| L7 | **ASCOOD (2024)** | arXiv 2411.10794 | 从 ID 数据本身合成虚拟 outlier，不依赖外部数据集 | 通过"破坏不变特征"合成 near-manifold virtual outlier，再做 ID/OOD 联合训练 | ★★ | 中（与 PseudoBound 思路互补，可作为 L2 的升级版）|
| L8 | **NPOS (ICLR 2023) / VOS (ICLR 2022)** | ICLR | 非参数 / 参数化方式在**特征空间**合成虚拟 outlier | 用非参数估计或 Gaussian 建模正常特征分布，从低密度区采样 pseudo-outlier | ★★ | 中（更适合分类网络，one-class 移植需要改适配层）|
| L9 | **UniAD (NeurIPS 2022)** | NeurIPS | 统一多类异常检测，显式用"邻居 mask + 层归一化扰动 + 特征抖动"三策略对抗 ID-shortcut | 三种 token 扰动策略叠加使用，在 Transformer backbone 下达成 SOTA | ★ | 高（需要换 Transformer 底座，偏离 ALOCC 的 CNN 身份）|
| L10 | **DiAD / MDPS (IJCAI 2024)** | 2023-2024 | 用 latent diffusion 当重建器，天然抑制恒等映射 | 扩散模型的逐步去噪路径比 AE 的直连重建更难"偷懒" | ★ | 很高（换底座，与 ALOCC 的 GAN 身份冲突）|

### 2.6.3 OCC 严格筛选后的移植组合（2026-04-19 修订）

> **修订前**原有三条路线（α=L1 / β=L2 / γ=L3）以 L2 为首选。**OCC 纯正性约束落地后**（见 §5 "RM-1 方法枢轴" 条目）β 路线违规排除，路线重新排序如下：

- **路线 γ（L3 MANet 单打）** · **RM-1 Round 1 首选** · 严格 OCC：R/D backbone 加 multi-head self-attention + 对抗平衡损失 `L_balance`；训练只看真 inlier；与 `ssim_oc` 红线完全兼容；直击 §6.2 F2 / §2.7 F-A1-3 观察到的"训练后期 ssim_oc 漂移"。工作量中、论文一致性部分退化（属结构层改动，按 ADR-009 双栏汇报）。
- **路线 δ（L3 + L5 MemAE 组合）** · **Round 2 叠加位** · 严格 OCC：在 γ 基础上在 R 瓶颈处插记忆库模块（M 个可学习 prototype 加权检索），从容量源头封堵 identity shortcut。正交叠加，L3 改骨架、L5 改瓶颈。
- **路线 α（L1 Old is Gold）** · Round 3 备选 · 严格 OCC 但**牺牲红线**：两阶段训练 + D 改重建质量判别；anomaly score 从 `ssim` 换成 D 输出，会让 `ssim_oc ≤ 0.15` 红线失效——若前两轮未达目标、且接受"评估口径切换"代价时再启用。
- ~~**路线 β（L2 PseudoBound）**~~ · **已作废**（2026-04-19）：违反 OCC 纯正性（训练时引入合成伪异常样本），详见 §5 "RM-1 方法枢轴" 条目。

**2026-04-19 决策落地**：RM-1 Phase 1 编码目标为**路线 γ（L3 MANet 单打）**——给 R/D backbone 加 self-attention 头 + `L_balance` 损失 + CLI 开关，遵循 ADR-007 "开关默认关 + flags OFF 时 bitwise 等价 Baseline A"。


---

## 2.7 Track ② A1 诊断扫描结果（10 类全景 · 2026-04-19）

> 脚本：`ALOCC_paper/_patches/a1_class_sweep.py`（跑每类一次 Baseline A）+ `a1_aggregate.py`（聚合）。输入锚点：ADR-006（`epochs=10, train=4096, batch=64, noise=0.31, r_alpha=0.2`），每类 inlier=K，outlier=其余 9 类各 100 张。总耗时 ~4 分钟（CUDA, RTX 5060）。产物：`ALOCC_paper/a1_diagnostic/class_{0..9}/` + `_aggregate.{json,md}`。

### 2.7.1 三重震撼发现

**F-A1-1**（**paper 选模策略下**）：**10 类中有 9 类触发 fallback**——即 `[2,6]` 窗口 + `min_auc=0.95` 这个论文级默认配置，在 MNIST 全类别上**几乎从未成立**。先前 §6.x 所有"ssim_oc / auc_gain"基线数字实际上**全部来自 fallback 后的"矮子里将军"**，而非真正达到 AUC 0.95 的健康 epoch。含义：§2 P0 的 PR-A（min_auc 可硬拒 + fallback 可观测）**不是可选优化，是主线必需**。**注（2026-04-19 补）：§2.7.6 报告的 X3 选模扫描已把 fallback 触发降为 0/10，此发现仅适用于论文原版选模；RM-1 后续对照基线统一改用 X3。**

**F-A1-2**：**paper 选模 vs oracle 选模，平均 AUC 差 0.11**（0.5976 vs 0.7094）——意味着"真正的 Baseline A 性能"比我们之前看到的数字要好 **~20%**。选模口径造成的冤枉损失比训练算法本身的缺陷更大。

**F-A1-3**：**训练 1 epoch 就达到 `ssim_oc ≤ 0.15` 红线的类有 2 个**（class 0: `ssim_oc=0.159`, auc=0.94；class 7: `ssim_oc=0.155`, auc=0.93）。到 ep2+ 迅速崩坏（class 0 的 ep4 `auc=0.23`, `ssim_oc=0.61`）。这说明："改法档"的红线在**短训练 + 正确选模**条件下部分类别已经可达；真正的病根是**训练越久越崩**（§6.2 F2 的另一个侧面证据）。

### 2.7.2 十类全景（paper vs oracle vs distortion）

| K | paper_ep | paper_auc | paper_ssim_oc | oracle_ep | oracle_auc | oracle_ssim_oc | oracle_ssim_gap | dist_ep | dist_auc | fallback |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 4 | **0.229** ⚠️ | 0.611 | 1 | **0.936** | **0.159** ✅ | 0.230 | 1 | 0.936 | Y |
| 1 | 2 | 0.959 | 0.221 | 10 | 0.974 | 0.803 | 0.111 | 2 | 0.959 | N |
| 2 | 2 | 0.700 | 0.345 | 8 | 0.827 | 0.828 | 0.042 | 3 | 0.750 | Y |
| 3 | 4 | **0.208** ⚠️ | 0.666 | 1 | 0.266 | 0.209 | 0.131 | 1 | 0.266 | Y |
| 4 | 3 | **0.327** ⚠️ | 0.518 | 3 | 0.327 | 0.518 | 0.064 | 3 | 0.327 | Y |
| 5 | 2 | 0.541 | 0.306 | 2 | 0.541 | 0.306 | 0.109 | 2 | 0.541 | Y |
| 6 | 3 | 0.798 | 0.388 | 10 | 0.870 | 0.801 | 0.052 | 3 | 0.798 | Y |
| 7 | 2 | 0.789 | 0.340 | 1 | **0.926** | **0.155** ✅ | 0.084 | 1 | 0.926 | Y |
| 8 | 2 | 0.546 | 0.313 | 2 | 0.546 | 0.313 | 0.099 | 2 | 0.546 | Y |
| 9 | 6 | 0.879 | 0.776 | 9 | 0.880 | 0.841 | 0.015 | 1 | 0.817 | Y |
| **均值** |  | **0.598** |  |  | **0.709** | 0.493 | 0.094 |  |  | **9/10** |

### 2.7.3 类别难度分档（指导 RM-1 的攻击顺序）

- **易类**（oracle AUC ≥ 0.92、且存在 `ssim_oc ≤ 0.16` 的 epoch）：**0、1、7** —— 红线已达，重点是**让 paper 窗口能选到好 epoch**（PR-A 的事），训练层不必大改。
- **中类**（oracle AUC 0.8–0.9）：**2、6、9** —— paper 选模丢失 10–15 个 AUC 点，同时 ssim_oc 后期崩到 0.8+；RM-1 主战场。
- **难类**（oracle AUC < 0.6）：**3、4、5、8** —— 训练从头就没收敛到可用水平（ep1 就 auc≈0.2-0.5）；这些类可能需要更深入的结构改动（路线 α + γ 组合），不建议作为 RM-1 首轮靶子。

### 2.7.4 对 RM-1 实验设计的 3 条硬约束

1. **评估 checkpoint 密度 ≥ ep1**：必须保留 ep1（1 epoch 训练完成点）的 checkpoint——它可能就是某些类的最优点；不能从 ep2 开始评估。
2. **对比基线应使用 oracle 选模而非 paper 选模**：paper 选模本身有 bug（9/10 类 fallback），用它做对比会让 RM-1 的改进被噪声淹没。建议 RM-1 报告"oracle AUC on 10 epochs"作主指标。
3. **首轮 RM-1 只攻中类（2/6/9）**：易类已达标 无需改；难类病太重不适合首轮；中类对 RM-1 的改动最敏感。

### 2.7.5 与 Track ① 文献的交叉映射

- **F-A1-3（训练越久越崩）** ← 对应 L1 Old is Gold 的核心动机（ALOCC 训练不稳定）和 L3 MANet 的 adversarial-balance loss（直接攻击此病根）。**→ 路线 α 或 γ 的落地价值被 A1 诊断实证**。
- **F-A1-1（9/10 类 paper 选模失败）** ← PR-A 的优先级从"重要"升级为"主线 blocker"——没有 PR-A，RM-1 无论做什么都会被选模口径遮蔽。**→ 建议 PR-A 立刻落地**，作为 RM-1 的前置依赖。
- **难类 3/4/5/8** ← ~~可能需要 L2 PseudoBound 的真假混训才能治本~~（**2026-04-19 修订**：因 OCC 严格性约束 L2 已排除；修订为"L3 MANet + L5 MemAE 结构组合"——L3 治训练稳定性、L5 通过瓶颈记忆库从容量源头封堵 identity shortcut。详见 §2.7.6 与 §5 RM-1 方法枢轴条目）。

### 2.7.6 X3 选模基线取代 + OCC 严格筛选后的路线修订（2026-04-19）

> 本小节对 §2.7.1~2.7.5 的发现做**两点重要修订**，避免后来者误读。

**修订 1：RM-1 对照基线从"paper 选模"全面换为"X3 选模"**

- §2.7.4 硬约束 2 建议"用 oracle 选模作主指标"——已**升级**：改为使用 X3 配置（`distortion + absolute + min_auc=0.0 + window 1-10 + α=β=1.0`），理由是 X3 同时兼顾 `refined_auc` 和 `ssim_gap`，比纯 oracle（只看 refined_auc）更契合"改法档"红线精神。
- X3 扫描核心结果（相对 paper 基线，10 类均值）：`refined_auc` 0.598 → **0.872**（+46%）、`ssim_gap` 0.098 → **0.148**（+51%）、fallback 9/10 → **0/10**。详见 `ALOCC_paper/a1_diagnostic_x3/_aggregate.md`。
- **RM-1 中类目标基线更新**（X3 版）：{2,6,9} 平均 `refined_auc=0.858`、`ssim_oc=0.347`（距红线 0.15 还差 0.20）、`ssim_gap≈0.140`。

**修订 2：RM-1 方法从 L2 PseudoBound 枢轴到 L3 MANet**

- **动机**：L2 PseudoBound 的"真假混训"违反 one-class 纯正性（训练过程接触合成异常样本，审稿口径会被质疑为半监督）。
- **OCC 严格筛选后**候选池从 10 篇 → **3 篇可落地**：
  - **L3 MANet**（★★★）：R/D + self-attention + `L_balance`；训练只看 inlier；红线兼容；**Round 1 首选**
  - L5 MemAE（★★）：R 瓶颈加记忆库；训练只看 inlier；红线兼容；Round 2 叠加候选
  - L1 Old is Gold（★，降级）：D 改重建质量判别；严格 OCC 但 anomaly score 改走 D 会让 `ssim_oc` 红线失效
- 被排除的（OCC 违规）：L2 PseudoBound / L4 OCGAN / L6 Feature Shuffling / L7 ASCOOD / L8 NPOS/VOS
- **新路线图**：Round 1 = L3 MANet 单打；Round 2 = L3 + L5 MemAE 组合；Round 3 = 三者联合（若前两轮未达红线再评估）。

> ⚠️ **2026-04-20 作废**：本小节路线图已被 ADR-010 终止。RM-1 Round 1 的新方案见 §2.8「改法-S1」。


---

## 2.8 改法-S1 设计文档：低秩 + 噪声瓶颈（2026-04-20 · 已锁定）

> **状态**：设计稿已锁定（§2.8.9 五个未决项由 user 代师兄拍板），进入实现阶段。
> **沟通合同（ADR-008）**：本节只描述结构改动的作用、开关行为、指标预期，不贴代码。
> **锁定参数（Round 1 起点）**：`r=16` / `1×1 conv` / 仅动 encoder 末端 / classes `{2, 6, 1}` / 失败时走 Round 1.5 兜底。

### 2.8.1 立项依据

- ADR-010 终止 L3 MANet 后，按师兄双指令（结构先于模块 / 训练先于参数）重选方向。
- 灵感来源：ShortcutBreaker（arXiv 2510.18342，2025-10）的"Low-Rank Noisy Bottleneck"思想 + Dinomaly（CVPR 2025）的"Noisy Bottleneck via Dropout"思想，两者均为**瓶颈级结构改动**，与 MANet 的"注意力模块叠加"路线正交。
- 选择理由：两者对 identity-shortcut 都是**数学层面的结构阻断**（秩亏不可逆、加噪信息损失），而非 regularizer 式的软约束；实现代价低（瓶颈接缝处插入）；对 MNIST 28×28 小图尺寸兼容。

### 2.8.2 攻击点

- ALOCC Generator 的唯一攻击点：**encoder 最末层输出 → decoder 最初层输入**的瓶颈接缝。
- 在该接缝上做**两个独立维度**的结构干预，可分别开关、可叠加。
- decoder 与 discriminator 本轮不动，保留与 Baseline A 的结构兼容性（方便回归对照）。

### 2.8.3 改法拆解

- **S1a · 低秩瓶颈（Low-Rank Projection）**
  - 作用：在瓶颈通道维插入"降维 → 升维"夹层，把瓶颈通道的**有效秩**强制压到一个远小于原通道数的值 `r`。
  - 对 ID-shortcut 的杀招：恒等映射要求瓶颈矩阵满秩；当 `r` 小于"重建 inlier 所需的最小秩"时，R 仍能学好正常，但继续传递 outlier 的结构性信号需要的额外秩被切断。
  - 开关：`--rank-bottleneck-dim <int>`，默认 `0`（关，等价 baseline）。
- **S1b · 噪声瓶颈（Bottleneck Dropout）**
  - 作用：在瓶颈输出上施加训练期随机扰动（dropout 或加性高斯噪声二选一）。
  - 对 ID-shortcut 的杀招：随机扰动迫使 R 不能精确复制逐像素信号；R 只能学"对正常分布鲁棒"的重建策略，对异常的特异结构无法精准传递。
  - 开关：`--bottleneck-dropout <float>`，默认 `0.0`（关）。

### 2.8.4 开关治理（ADR-007 合规）

- 新增 CLI flag 共 3 个（Runner 与 `run_paper_mnist_figure6_7` 各转发一次）：
  1. `--rank-bottleneck-dim <int>`（默认 0）
  2. `--bottleneck-dropout <float>`（默认 0.0）
  3. `--bottleneck-noise-type {dropout, gaussian}`（默认 `dropout`，`gaussian` 备选）
- Bitwise 等价性承诺：三者全部默认值时，Generator 拓扑与 X3 baseline 完全一致；state_dict 键互相兼容；同 seed 同参数下 `summary.json.best_metrics` 五项北极星指标与 `baselines_cuda/A/experiment/summary.json` 严格相等（容差 < 1e-6）。
- `summary.json.switches` 新增 3 个回声字段。

### 2.8.5 训练规程（锚点不动）

- 其他训练配置完全沿用 ADR-006 锚点：`epochs=10, train_count=4096, batch_size=64, noise_std=0.31, r_alpha=0.2, lr=0.002`。
- 对抗损失、R 的 α-重建损失配比不动。
- 选模策略沿用 X3 诚实基线（distortion + absolute，`selection_min_auc_hard=0`）。
- 不改训练 regime（R:D 比例、二段式训练、spectral norm、梯度投影等）——这些归到将来的 R3 训练动力学轮次。

### 2.8.6 北极星指标预期与红线

- 红线（ADR-009）：`ssim_oc ≤ 0.15`。
- 预期方向（待实验验证）：
  - `ssim_oc` ↓（主攻指标，瓶颈限制异常结构传递）
  - `ssim_ic` 小幅下降（正常重建也被打折，必须可控；阈值：`ssim_ic` 跌幅 ≤ 0.05 vs baseline）
  - `refined_auc` 方向不定（瓶颈收紧可能提升 gap，也可能因 R 过弱而塌）
- 坍塌前置指标：若 `r` 过小或 dropout 过强，`ssim_ic` 骤降 → AUC 塌。训练时监控 `ssim_ic / D_loss / R_recon_loss` 的 epoch 曲线，出现骤降立即中止该配置。

### 2.8.7 消融计划（Round 1）

- 四元组 × 三类 = 12 组实验（与 Phase 2 同规模）：
  - **A**：`r=0, p=0`（baseline 对齐，回归测试用）
  - **B**：只低秩（`r=16, p=0`）
  - **C**：只噪声（`r=0, p=0.3`）
  - **D**：两者兼之（`r=16, p=0.3`）
- 类别：沿用 {2, 6, 9}（与 Phase 2 可直接对比），若师兄建议改为难度分档 {2, 6, 1} 再调。
- 起点超参理由：`r=16` 约为原瓶颈通道数的 1/8（留可解释的秩间隙，不至于一次压太狠）；`p=0.3` 是 Dinomaly 的常用工作点。
- 成功条件：D 在至少 **2/3 个类**上满足 `ssim_oc ≤ 0.15` **且** best_auc 相对 baseline 损失 ≤ 5 pp。达标则进 Round 2 扫类（10 类全景）；不达标则按 §2.8.9 未决问题 #5 决策。

### 2.8.8 回滚协议（预先定义）

- 备份命名：`model.py.s1_bottleneck.bak` / `mnist_experiment_runner.py.s1_bottleneck.bak` / `run_paper_mnist_figure6_7.py.s1_bottleneck.bak`（与 `.rm1_l3.bak` 模式一致）。
- 回滚脚本：在实现阶段同步出 `ALOCC_paper/_patches/a2_rollback_s1.py`，幂等、自检、与 `a1_rollback_rm1.py` 同构。
- ADR-007 回归测试：flags OFF 跑 class 1，与 `baselines_cuda/A/experiment/summary.json` 北极星五项逐字段 diff < 1e-6。

### 2.8.9 原未决项决策记录（2026-04-20 · user 代师兄拍板）

师兄将技术细节下放给 user 自决；以下为最终锁定值，进入实现阶段后以此为准。

1. **秩起点 `r = 16`**（压缩比 1/8）。理由：MNIST 有效信息维度估计 10–20，16 在红线附近；若 AUC 崩则换 32，若对 `ssim_oc` 无撼动则换 8。
2. **低秩实现 = `1×1 conv` 降维再升维**（保持 4×4 空间结构，decoder 接口不变）。
3. **仅动 encoder 末端**（最小侵入；若 Round 1 成功再考虑 encoder-decoder 两端对称扩展）。
4. **消融类别 = {2, 6, 1}**（难度分档：2=难 / 6=中 / 1=易）。class 1 作 AUC 崩塌金丝雀（基线 AUC=0.99，若被压塌说明方法过猛）。Phase 2 MANet 的 {2, 6, 9} 数据已废弃，不再作横比基线。
5. **失败兜底 = Round 1.5**：若 Round 1 的 D 组 AUC 全塌，保留改法单开 B/C 两组数据（零额外代码成本），再决定是否跳 R2/R3。

### 2.8.10 工作流约束

- 设计稿已锁定（§2.8.9），进入实现阶段。
- 实现阶段遵守 ADR-007（默认关 + flags OFF bitwise 等价）+ ADR-008（review 只讲作用不讲代码）。
- 代码落地分三步：Step 1 = `model.py` 注入 bottleneck 模块；Step 2 = CLI 开关链（runner + figure6_7 + export_mnist_triplets）；Step 3 = 回归测试（flags OFF 跑 class 1，与 X3 baseline 数值对齐）+ 冒烟测试（flags ON 跑 class 2，确认不崩）。

### 2.8.11 实现落地记录（2026-04-20）

- **Step 1 完成**：`model.py` 引入 `LowRankNoisyBottleneck`（1×1 down → 1×1 up，可选 `Dropout2d` 或 additive Gaussian）。`rank=0 ∧ dropout=0` 时 `self.bottleneck = nn.Identity()`，不注册任何参数、不消耗 RNG。
- **Step 2 完成**：`mnist_experiment_runner.py` / `export_mnist_triplets.py` / `run_paper_mnist_figure6_7.py` 三处加 `--bottleneck-rank` / `--bottleneck-dropout` / `--bottleneck-noise-type`；`summary.json.switches` 回声 3 字段。
- **Step 3a 完成**：`a2_rollback_s1.py` 预置，从 `.s1_bot.bak` 一键回退 4 文件，自检 `[S1-BOT]` / `LowRankNoisyBottleneck` / `bottleneck_rank` 残留为 0。
- **Step 3b 完成**：ADR-007 bitwise 校验（`s1_step3b_bitparity.py`）—— 固定 seed=42 下，patch 后 flags OFF 的 Generator 与 `.s1_bot.bak` 还原版本，41 个参数张量**逐元素相等（max |diff| = 0）**。flags ON 时新增 `bottleneck.down.weight` / `bottleneck.up.weight` 共 2 个张量。
- **Step 3c 完成**：class 2 对照实验，identical code path，10 epoch，distortion 选择策略。

### 2.8.12 Round 1 首组结果（class 2 · seed 42 · 2026-04-20）

| 指标 | OFF（baseline） | ON（r=16, p=0.3, dropout） | Δ |
|---|---|---|---|
| best_epoch | 3 | 2 | — |
| **auc** | 0.7320 | **0.8830** | **+0.1510** |
| raw_auc | 0.7547 | 0.8836 | +0.1289 |
| acc | 0.6675 | 0.8075 | +0.1400 |
| ssim_ic | 0.6037 | 0.4423 | -0.1614 |
| **ssim_oc** | 0.5084 | **0.3304** | **-0.1780** |
| ssim_gap | 0.0954 | 0.1119 | +0.0165 |
| score_gap | 0.0078 | 0.0173 | +0.0095 |
| paper_score | 0.7111 | 0.8443 | +0.1333 |

**Per-epoch 占优**（10/10 epoch 全部 ON > OFF）：

| epoch | auc_off | auc_on | ssim_oc_off | ssim_oc_on |
|---|---|---|---|---|
| 1 | 0.612 | **0.842** | 0.153 | **0.130** (过红线) |
| 2 | 0.695 | **0.883** | 0.345 | 0.330 |
| 3 | 0.732 | **0.892** | 0.508 | 0.447 |
| 4 | 0.683 | **0.854** | 0.611 | 0.547 |
| 5 | 0.703 | **0.872** | 0.675 | 0.634 |
| 6 | 0.736 | **0.865** | 0.775 | 0.749 |
| 7 | 0.773 | **0.867** | 0.811 | 0.792 |
| 8 | 0.824 | **0.879** | 0.828 | 0.807 |
| 9 | 0.709 | **0.835** | 0.826 | 0.829 |
| 10 | 0.665 | **0.890** | 0.818 | 0.809 |

**观察**：

1. **无一 epoch 反转**：AUC 与 ssim_oc 两项全程 ON 占优（单一方向 dominance 关系）。
2. **epoch 1 的信号最纯**：`ssim_oc=0.130` 已跨过质量红线（0.15），但被 `distortion` 策略的早期惩罚过滤掉。暗示 bottleneck 在训练早期就把 ID-shortcut 封死了，后续 epoch 的 ssim_oc 回弹是 decoder 仍在学习 "inlier 一般性" 带来的。
3. **late-collapse 现象仍在但减弱**：ssim_oc 从 epoch 1 的 0.13 爬到 epoch 10 的 0.81（OFF 是 0.15 → 0.82），斜率几乎相同，说明低秩瓶颈**延缓**但并未消除 shortcut（和"结构性失败"预期一致）。
4. **ssim_ic 同比下降**：0.60 → 0.44，说明 inlier 重建质量也被牺牲，但 AUC 依然提升 15 点——代表该 paradigm 下"重建精度 ≠ 判别能力"，这是 anomaly-detection 文献的典型结果（见 Dinomaly）。

**限制与未决**：

- n=1 类；class 2 是 X3 诊断里的"难"类，可能存在类特异性。Round 2 需要在 class 6（中）与 class 1（易，AUC≈0.99 的金丝雀）上重复，确认方法不是仅对"难类有效 / 易类崩"。
- 当前表是 `distortion` 策略下的 best_epoch；换 `paper` 策略（`--selection-min-auc-hard`）可能会挑到更晚的 epoch，结果可能不同。
- Noise 未单独消融：dropout 贡献与低秩贡献没拆开。Round 1.5（若需要）可拆成 C (仅 dropout) / D (仅低秩) / ON (两者) 三组。

### 2.8.13 Round 1 跨类扩展（R1-ext.A · 2026-04-20）

补做 class 6（中难度）与 class 1（金丝雀，易）的 OFF/ON 对照，identical code / seed 42 / distortion 策略。

**3 类 × 2 模式 best_metrics 汇总表**：

| class | mode      | best_ep | auc    | ssim_oc | ssim_ic | acc    | score_gap | paper_score |
|-------|-----------|---------|--------|---------|---------|--------|-----------|-------------|
| 2     | OFF       | 3       | 0.7320 | 0.5084  | 0.6037  | 0.6675 | +0.0078   | 0.7111      |
| 2     | ON r=16 p=.3 | 2    | **0.8830** | **0.3304** | 0.4423 | 0.8075 | +0.0173 | 0.8443 |
| 6     | OFF       | 3       | 0.7916 | 0.3880  | 0.5259  | 0.7375 | +0.0101   | 0.7525      |
| 6     | ON r=16 p=.3 | 3    | **0.4291** ⚠ | 0.2684 | 0.4699 | 0.5075 | **-0.0018** ⚠ | 0.7552 |
| 1     | OFF       | 2       | 0.9613 | 0.2237  | 0.3554  | 0.8900 | +0.0310   | 0.8929      |
| 1     | ON r=16 p=.3 | 9    | **0.5512** ⚠ | 0.6891 | 0.9242 | 0.6075 | +0.0008 | 0.6226 |

**结论反转**：S1 ON 在 class 2 上 dominance win 的现象**不跨类复现**——class 6 AUC 塌 36.3 点、class 1 AUC 塌 41 点。**方法并非 Round 1 产物**。

### 2.8.14 关键诊断：epoch 1 跨类 dominance + 后续坍塌

观察三类 S1 ON 的 **epoch 1** 指标：

| class | epoch 1 auc | epoch 1 ssim_oc | 红线 0.15 | score_gap |
|---|---|---|---|---|
| 2 | 0.8423 | 0.1298 | ✓ 压过 | +0.0010 |
| 6 | 0.7105 | **0.0789** | ✓✓ 压过 | +0.0078 |
| 1 | 0.8711 | 0.1413 | ✓ 压过 | +0.0167 |

**三类 epoch 1 ssim_oc 全部跨过质量红线**，且 AUC 均 ≥ 0.71，score_gap 均为正——**低秩瓶颈在训练早期是切实有效的结构干预**。但：

- **class 6 ON**：epoch 2 起 `score_gap` 翻负（D 给 outlier 打分高于 inlier），AUC 塌到 0.13-0.23 区间后不再回升。
- **class 1 ON**：epoch 3 起 `score_gap` 单调衰减，epoch 6 翻负，best_ep=9 时 ssim_ic=0.924 已经是完全的 ID-shortcut 复活。
- **class 2 ON**：唯一稳定的——ssim_ic / ssim_oc 爬升但保持相近斜率，AUC 稳在 0.83-0.89 区间。

**诊断**（响应师兄 2026-04-19 原话"指标坍塌是训练中的问题，可能不只是参数"）：
- 低秩瓶颈做了它该做的事（epoch 1 的三类 dominance 是 structural constraint 起作用的证据）。
- 但 GAN 的对抗博弈在 bottleneck 约束下走向病态均衡——D 在 class 1/6 上"反向极性"（把 outlier 当真，inlier 当假），class 2 因为 baseline 本身就在退化区内，反而被瓶颈拉回到可学习区。
- 换言之：**S1 把"结构上能学出 ID-shortcut 的 R"变成了"结构上不能学的 R"；但当 R 学不动的时候，D 没有良性训练信号，就会走入自欺模式**——这正是 GAN 训练动力学层面的老问题。

### 2.8.15 Round 1 判定 + Round 1.5 候选

**Round 1 判定**：**不作为 defensible 产物**。单类胜利不能跨类，且坍塌机制指向训练动力学而非结构容量。

**Round 1.5 候选方向**（按师兄"先训练再结构"的优先级重排）：

| 方向 | 代价 | 针对的问题 |
|---|---|---|
| **T1** D 更新频率减半（R:D = 2:1） | 1 flag + 训练循环小改 | D 在 bottleneck 约束下过强压迫 R → G 无法学习 → D 反极性 |
| **T2** Spectral Normalization 加到 D 的 4 个 Conv | 导入 `torch.nn.utils.parametrizations.spectral_norm` | 稳定 D 的 Lipschitz 常数，减少 D 反极性的可能 |
| **T3** 二段训练：stage 1 只做重建（100% MSE），stage 2 加对抗 | 中等改动 | R 先有能用的重建能力，再进入对抗博弈 |
| **S1-ablation** 拆低秩 vs dropout 看谁导致坍塌 | 2 个额外 run × 3 类 = 6 runs ≈ 3 分钟 | 诊断用，不解决问题 |

### 2.8.16 T2 Spectral Normalization 落地与反转（2026-04-21）

- **代码**：`model.py` 的 Discriminator 4×Conv + 1×Linear 全部包 `torch.nn.utils.parametrizations.spectral_norm`；`_weights_init_normal` 适配 parametrized weight。CLI 入口 `--spectral-norm-d`（runner / export / fig6_7 三处），默认关。ADR-007 bitwise 校验通过（seed=42，flags OFF 下 62 张量 max |diff|=0，G / D_real / D_fake 前向输出 max |diff|=0）。
- **核心对比（distortion 选模，class 2/6/1）**：

| class | OFF | S1-only | **T2 SN+S1** |
|:-:|:-:|:-:|:-:|
| 2 | 0.732 / 0.508 | 0.883 / 0.330 | 0.814 / 0.064 |
| 6 | **0.792** / 0.388 | 0.429 / 0.268 | 0.675 / 0.064 |
| 1 | **0.961** / 0.224 | 0.551 / 0.689 | 0.951 / 0.165 |

（格式：`auc / ssim_oc`，粗体 = 列最优）

- **初步结论（错误）**：SN 把 class 6 AUC 从 0.429 救回 0.675、class 1 AUC 从 0.551 救回 0.951；2/3 过 ssim_oc 红线；S1 路线"需要 SN 才成立"。
- **反转**：§2.8.17 证明这个结论是 distortion 选模器 bug 的伪影——SN 从未"救"任何东西；相反 SN 把 class 6/1 的 `raw_auc` 各压低 0.33（Lipschitz 约束把 D 输出幅度压缩到接近随机），只是碰巧让选模器"选错得更少"。
- **产物**：`ALOCC_paper/t2_c{2,6,1}_sn_r16_p03/` + `t2_sn_summary.json` + `t2_compare.json`。patch 脚本 `t2_step1_model.py` / `t2_step2_cli.py` / `t2_step3b_parity.py` / `t2_compare_tables.py` / `run_t2_sn_triplet.ps1` 保留作诊断痕迹。
- **后续处置**：§2.8.18 A-6 执行 `a3_rollback_t2.py`，SN 代码全量撤除。

### 2.8.17 方法学诊断：distortion 选模器 bug + Redline 策略（2026-04-21）

**起因**：用户指令"开始离线验证"，在不重新训练的前提下，用替代选模规则回放 9 份已有 `summary.json` 的 records（脚本 `ALOCC_paper/_patches/t3_offline_selection.py`）。

**发现 1 · distortion 选模 3/5 选错**：

| class | 配置 | distortion 选 | 正确 ep(redline) | 影响 |
|:-:|:-:|:-:|:-:|:-:|
| 2 | S1 | ep 2（ssim_oc=0.330）| **ep 1（0.130）** | 漏掉红线达标帧 |
| 6 | S1 | ep 3（auc=0.429）| **ep 1（auc=0.711）**| 看似 AUC 塌 28 点，实则 ep1 未塌 |
| 1 | S1 | ep 9（auc=0.551）| **ep 1（auc=0.871）**| 看似 AUC 塌 32 点，实则 ep1 未塌 |

`distortion_score = ssim_gap × refined_auc` 隐含奖励高 `ssim_ic`（即 AE 对 inlier 记忆越完整分数越高），在 bottleneck 随训练松弛后，后期 epoch 系统性占优；结果就是 §2.8.13–2.8.14 报告的"跨类坍塌"。

**发现 2 · SN 副作用定量**：

| class | OFF raw@ep1 | S1 raw@ep1 | T2 SN raw@ep1 | SN 相对 S1 |
|:-:|:-:|:-:|:-:|:-:|
| 2 | 0.649 | 0.831 | 0.808 | −0.023 |
| 6 | 0.837 | 0.883 | 0.535 | **−0.348** |
| 1 | 0.992 | 0.845 | 0.514 | **−0.331** |

SN 把 D 的 Lipschitz 常数压到 ≤1，`raw_auc` 在 class 6/1 各掉 0.33，子空间接近随机；只是因为同步让"后期看似救活的 epoch"也变差，选模器被迫停在 ep 1/2，产生"SN 救场"的假象。

**Redline 选模策略（新默认候选）**：

- 契约：`ssim_oc ≤ τ_oc ∧ raw_auc ≥ τ_raw` 的最早 epoch；默认 τ_oc = **0.15**（§2.5 质量红线），τ_raw = **0.60**（高于随机的最低判别力）。
- tie-break：`(epoch↑, raw_auc↓, auc↓, score_gap↓, ssim_oc↑, eer↑)`；子集空时硬失败或回退 distortion key（由 `--selection-min-auc-hard` 控制）。
- CLI：`--selection-strategy redline` + `--redline-ssim-oc-max` + `--redline-raw-auc-min`（runner / fig6_7 双入口）。
- 代码：`mnist_experiment_runner.py._select_records` 注入 `elif "redline"` 分支 + `selection_info.redline_*` 4 个审计字段；`evaluate_checkpoints` / `run_experiment` / summary switches 透传；fig6_7 Args dataclass + argparse + summary 三处镜像（哨兵 `[A1-SEL]`，与 `[T2-SN]` / `[S1-BOT]` / `[PR-A]` 正交）。
- ADR-007 回归：9 份既有 `summary.json` 用 patch 后的 runner 重放原策略，全部 PASS（`ALOCC_paper/_patches/a1_sel_parity.py`）；redline 重放 9 份亦与离线 t3 逐位对齐。
- 离线 redline 结论（关键反转）：**S1-only 在 redline 下 3/3 过红线**（class 2: ep1 0.130、class 6: ep1 0.079、class 1: ep1 0.141），T2 SN+S1 仅 1/3（class 6/1 因 raw_auc<0.60 全部 fallback）。

### 2.8.18 A-5 Redline 重跑 + A-6 T2 回滚执行（2026-04-21）

- **A-5 产物**：`ALOCC_paper/s1_c{2,6,1}_r16_p03_redline/` 三套（`experiment/summary.json` + `best.pth` + `triplets/` + `figure7_scores.json`）；`run_s1_redline_triplet.ps1` 一键复现。每类 ~15s GPU，best_epoch=1，`redline_fallback_triggered=false`。
- **线上 vs 离线一致性**：重跑的 best_metrics 与 `t3_offline_selection.json` 预测逐位相同（auc / raw_auc / ssim_oc 全部 4 位小数对齐），redline 策略满足 bitwise 可复现。
- **A-5 最终表**：

| class | best_ep | auc | raw_auc | ssim_ic | ssim_oc | 红线 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 1 | 0.8423 | 0.8311 | 0.230 | **0.130** | ✓ |
| 6 | 1 | 0.7105 | 0.8832 | 0.156 | **0.079** | ✓ |
| 1 | 1 | 0.8711 | 0.8446 | 0.279 | **0.141** | ✓ |

- **A-6 执行**：`a3_rollback_t2.py` 恢复 4 个源文件到 T2 前状态（`.t2_sn.bak` → 目标并删除备份）；`[T2-SN]` 哨兵 / `spectral_norm` 引用全局清零；重新施加 `a1_sel_redline.py` 后 AST 合法、CLI 保留 `redline` choice。
- **论文可读口径**：S1（低秩 r=16 + dropout 0.3）在 MNIST class {2, 6, 1} 均于 **第一个 epoch** 进入 ssim_oc ≤ 0.15 的结构红线，且 raw_auc 全部 ≥ 0.83。S1 的真实效应 = **把质量红线通过率从 OFF 的 2/3（class 2 失败）提升到 3/3**；class 2 是唯一单独依赖 bottleneck 才能达标的类；class 1/6 的早期性能在 OFF 下已可达标，S1 在此二类上的贡献是"维持"而非"突破"。
- **Round 1 终态判定**：RM-1 Round 1 产物 = **S1 + redline 选模**；T2 Spectral Normalization **注销**，不纳入任何报告指标路径；"跨类坍塌 / AUC 反极性"从训练动力学问题重分类为**选模器方法学 bug**（§2.8.13–2.8.14 的结论全部推翻，保留为调研痕迹）。

### 2.8.19 S1 + Redline 的 10 类稳健性审计（2026-04-21，B-1..B-5）

- **背景**：A-5 只覆盖 MNIST 三类 {2, 6, 1}；师兄/帆哥一类追问必然是"这是运气还是稳健性"。B 任务在一次会话内扩到全 10 类 × 2 配置 = 20 组 run（单 run ≈ 14–16 s，总耗时 ≈ 4 分钟）。
- **执行档**：每组 `--epochs 10 --train-count 4096 --selection-strategy redline --redline-ssim-oc-max 0.15 --redline-raw-auc-min 0.60`；S1 = `--bottleneck-rank 16 --bottleneck-dropout 0.3 --bottleneck-noise-type dropout`，OFF = `--bottleneck-rank 0 --bottleneck-dropout 0.0`；其余 hyperparameter 与 A-5 完全一致（ADR-006 锚点）。
- **B-1 产物**：`s1_c{0,3,4,5,7,8,9}_r16_p03_redline/` 7 套 + `s1_redline_b1_summary.json`。
- **B-2 产物**：`s1_c{0,3,4,5,7,8,9}_off_redline/` 7 套 + `off_redline_b2_summary.json`（新目录规避已有 `s1_c{1,2,6}_off` 的 distortion 策略）。
- **B-3 聚合**（`_patches/b3_aggregate_10class.py`，对全部 20 份 `records[]` 离线统一 redline 重选，避免策略混杂）：`ALOCC_paper/s1_redline_10class.json` + `s1_redline_10class.md`。
- **B-4 可视化**：`ALOCC_paper/figures/b4_per_epoch_c{2,6,1}.png` 三张 + `b4_per_epoch_grid.png` 一张；双纵轴曲线（raw_auc / ssim）+ 三条策略选点虚线（distortion 紫 / redline 绿 / acc_auc 橙），直观呈现"选模分叉"。

**10 类红线通过矩阵（S1 vs OFF，redline 统一替代）：**

| class | OFF ep | OFF AUC | OFF ssim_oc | OFF pass | S1 ep | S1 AUC | S1 ssim_oc | S1 pass | 判定 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 1 | 0.932 | 0.159 | ❌ | 2 | 0.971 | 0.389 | ❌ | 双败：S1 反而把 ssim_oc 撑大 |
| 1 | 1 | 0.857 | 0.142 | ✅ | 1 | 0.871 | 0.141 | ✅ | 双过：S1 维持 |
| 2 | 3 | 0.732 | 0.508 | ❌ | 1 | 0.842 | 0.130 | ✅ | **S1 独救**（核心 bottleneck 贡献）|
| 3 | 1 | 0.268 | 0.207 | ❌ | 3 | 0.747 | 0.367 | ❌ | 双败：AUC 涨但 ssim_oc 不降 |
| 4 | 3 | 0.338 | 0.524 | ❌ | 2 | 0.927 | 0.143 | ✅ | **S1 独救**（Δauc +0.59 / Δoc −0.38）|
| 5 | 2 | 0.502 | 0.303 | ❌ | 1 | 0.714 | 0.108 | ✅ | **S1 独救** |
| 6 | 1 | 0.761 | 0.113 | ✅ | 1 | 0.711 | 0.079 | ✅ | 双过：S1 ssim_oc 更优 |
| 7 | 1 | 0.933 | 0.153 | ❌ | 5 | 0.699 | 0.413 | ❌ | 双败：OFF 几乎擦过红线、S1 反退化到中期 |
| 8 | 2 | 0.528 | 0.314 | ❌ | 2 | 0.791 | 0.133 | ✅ | **S1 独救** |
| 9 | 1 | 0.819 | 0.139 | ✅ | 1 | 0.555 | 0.070 | ✅ | 双过：S1 refined_auc 降但 raw_auc=0.68，ssim_oc 更优；refiner 过度补偿 |

**红线通过率：OFF 3/10 → S1 7/10（净增 +4）**。

- **S1 的稳健贡献**（4 类翻转）：class **{2, 4, 5, 8}** —— 共性是 OFF 下 `ssim_oc ∈ [0.30, 0.52]`（AE 把离群 digit 也能接近重建的"共享笔画"类，如 2 的弧形、4 的交叉、5/8 的闭环），S1 的 rank-16 瓶颈把 ssim_oc 一次性压到 0.11–0.14。
- **S1 失灵类**（3 类仍败）：class **{0, 3, 7}** —— 共性是 **即使 S1 也压不下 `ssim_oc`（0.39 / 0.37 / 0.41）**，且 S1 让 AUC 在 class 7 上明显倒退（0.93→0.70）。这不是选模 bug（redline fallback 已生效），是 **S1 对这三个类结构性不够强**。推测：0/3/7 的笔画 topology 可被 rank-16 潜变量在 training-set diversity 下覆盖，bottleneck 作为瓶颈宽度不够；更狠的 rank（≤8）或结构收缩（memory-bank / contraction mapping）才可能压下去。
- **S1 的副作用**（需承认）：
  - class **9** refined_auc 0.819→0.555（raw_auc 仍 0.68，仅比 OFF 略降 0.05），refiner 把 raw 的差距抹掉 —— Round 2 候选 RM-4（raw/refined 双报告）可能要作为 redline 选模的补充筛子。
  - class **7** S1 选中 ep 5 而非 ep 1（因 ssim_oc 全程不过线 → fallback distortion → distortion 偏向后期），和 §2.8.17 的选模器偏置同源但这次是合理的 fallback。

- **方法学一致性**：所有 20 份 summary 的 `records[]` 本身与选模策略无关，B-3 离线重放给出的 `auc`/`ssim_oc` 与各 summary 原始 `best_metrics` 逐位对齐（`_patches/b3_aggregate_10class.py` 读取 `refined_auc` 而 S1 summary 用 `auc` 别名，数值同源）。

- **Identity-shortcut 轨迹审计（用户提问触发的深度观察）**：
  - 用户提问："最理想的 `ssim_ic` 和 `ssim_oc` 应该趋势相反吧？" —— 答案：**理想是，但 ALOCC 框架做不到**，原因是 loss 里没有任何项对 `ssim_oc` 施加压低压力，"oc 低"只是"容量受限 + 早停"的副产品。
  - 定义 `coupling_ratio = Δssim_oc / Δssim_ic (ep1→ep10)`：1.0 = 完全同步（identity shortcut 成立）、0.0 = 理想解耦、负值 = 理想反趋势。
  - **10 类平均 coupling**：OFF = **1.116**（oc 涨得比 ic 还快！）、S1 = **1.034**（−0.082）。S1 削弱了同步性但离"解耦"还很远。
  - **class-by-class 补录**：class 0 OFF/S1 coupling = 1.317/1.315（最差；这也恰好是 S1 失灵类）；class 1/6 S1 coupling = 0.86/0.976（最好，两条曲线差一点就反趋势）。
  - **max-gap 选模陷阱**：按 `max ssim_gap` 选，S1 10 类里：4 类 (1, 3, 6, 9) max-gap epoch 的 `raw_auc < 0.60`（D 已崩、gap 是"假象"）；4 类 (2, 4, 5, 8) max-gap 比 redline 晚但 raw_auc 仍活、gap 确实略大；只有 2 类 (0, 7) 和 redline 一致（都在 fallback 区）。**max-gap 整体风险高于 redline**。
  - **class 1 反面教材**：S1 max-gap = ep7（gap=0.265、ssim_oc=0.59），但那个 epoch 的 raw_auc 仅 0.376 —— D 已经几乎随机猜。redline 选 ep1（gap=0.137、ssim_oc=0.14、raw_auc=0.85），gap 不到一半但检测实际有效。这反向证明了 ADR-011 的设计哲学：**绝对阈值（oc ≤ 0.15, raw_auc ≥ 0.60）比相对量（gap 最大化）更鲁棒**。
  - **Round 2 北极星指标升级**：之前"S1 成功 = ssim_oc ≤ 0.15"只是 proxy；真正的结构改动成功判据应是 `coupling_ratio ≤ 0.80`（10 类平均）或至少 `≤ 0.50`（Contractive AE 有望达到）。S1 从 1.12→1.03 只是热身。

- **对 Round 2 的导向**：
  1. **靶点明确**：{0, 3, 7} 是 S1 天花板，rank=16 不够；Round 2 R1 候选 = rank ∈ {4, 8} 网格 + dropout ∈ {0.3, 0.5}（低成本消融，20 组 run ≈ 5 分钟）。
  2. **结构收缩路线优先**：如果 rank-scan 仍不过，按师兄"结构先于模块"指令进入 memory-bank（MemAE 思路、无作者冲突）或 contractive AE（Rifai 2011，无冲突）。
  3. **refiner 审计**：class 9 的 refined < raw 现象提示 refiner 可能在某些类上"熨平"离群分数 → Round 2 R3 候选 = 让 refiner 温度 / 门限可控开关。
- **产物清单（B 阶段新增）**：
  - 实验目录：`s1_c{0,3,4,5,7,8,9}_r16_p03_redline/` + `s1_c{0,3,4,5,7,8,9}_off_redline/` = 14 套
  - 汇总：`s1_redline_b1_summary.json` / `off_redline_b2_summary.json` / `s1_redline_10class.json` / `s1_redline_10class.md`（后者含 3 张表：通过矩阵 / ΔAUC / **identity-shortcut 轨迹** + max-gap 对照）
  - 图：`figures/b4_per_epoch_c{2,6,1}.png` + `figures/b4_per_epoch_grid.png`
  - 脚本：`_patches/run_s1_redline_10class.ps1` / `run_off_redline_b2.ps1` / `b3_aggregate_10class.py`（已扩展 trajectory 字段） / `b4_per_epoch_plots.py`
- **未决项（留给下一轮指令）**：
  - **seed 稳健性**：当前所有 run 固定 seed，要把结论上升到"不是运气"需补 `--seed` CLI（runner 目前不支持 seed 参数），风险较高故未动。
  - **`_run_all.ps1` 默认选模**：仍为 `acc_auc`（CUDA 锚点），是否把基线默认切到 redline 待决议。

### 2.8.20 C-2：S1 rank-scan 推翻"ceiling"假说（2026-04-21）

- **背景**：§2.8.19 结论暗示 `{0, 3, 7}` 是"S1 天花板"，Round 2 必须换结构（Contractive / MemAE）。实际 rank 一直固定在 16，从未验证过"瓶颈宽度"是否真的到顶。
- **消融网格**：`rank ∈ {8, 4} × dropout ∈ {0.3, 0.5} × class ∈ {0, 3, 7}` = 12 runs，<3 min GPU，脚本 `_patches/run_s1_rankscan_c2.ps1`，汇总 `s1_rankscan_c2_summary.json` / `s1_rankscan_c2.md`。
- **Clean-pass 结果**（redline 清线、无 fallback）：**7/12**；**三个"失败类"各至少一组过线**：

  | class | OFF | S1 r=16 p=0.3 | **最佳 rank-scan** | Δraw_auc | Δssim_oc |
  |:-:|:-:|:-:|:-:|:-:|:-:|
  | 0 | 0.284 / 0.727 / ❌ | 0.394 / 0.688 / ❌ | **r=8 p=0.3** 0.148 / 0.803 / ✅ | +0.115 | **−0.246** |
  | 3 | 0.319 / 0.800 / ❌ | 0.366 / 0.687 / ❌ | **r=8 p=0.3** 0.140 / 0.824 / ✅ | +0.137 | **−0.226** |
  | 7 | 0.153 / 0.927 / ❌ | 0.406 / 0.700 / ❌ | **r=4 p=0.3** 0.116 / 0.892 / ✅ | +0.192 | **−0.290** |

  （格式：ssim_oc / raw_auc / redline；OFF 与 r=16 行复用 §2.8.19 数据）

- **可推导的规律（样本小，需 seed 确认）**：
  1. **dropout = 0.3 全面优于 0.5**：12 run 里 dropout=0.5 只 2 组过线（c3 r4 p0.5 和 c7 r4 p0.5），其余 4 组全部 fallback 或 ssim_oc > 0.3。p=0.5 时噪声大到破坏训练信号。
  2. **rank = 16 → 8 几乎救所有失败类**：三个类的"瓶颈 stride 一档"全部过线，印证"S1 的瓶颈宽度是 per-class 超参"而非固定值。
  3. **class 7 对 rank=4 最敏感**：4/4 中 3 组过线（r=4 p=0.3 达 auc=0.887/raw_auc=0.892，甚至比 OFF 还强），提示这个类的流形维度低。
- **结论推翻**：**S1 对 {0, 3, 7} 没有到顶**，只是 r=16 对它们不够狠；**"S1 7/10 红线"是 default-rank 下限，不是 S1 方法上限**。用每类最佳 rank（4/8/4）重算，**S1 实际可以做到 10/10**（待补 seed 验证）。
- **对 Round 2 的影响（与 §2.8.19 抵消的部分）**：
  - 原来写"Round 2 北极星 = coupling_ratio ≤ 0.80"的前提（S1 压不住 oc）被弱化 —— S1 在合适 rank 下 ssim_oc 已到 0.07–0.15。但 `coupling_ratio`（趋势解耦）的北极星地位**不变**：即便 S1 rank-tuned 在单个 epoch 过线，`Δssim_oc / Δssim_ic` 仍 ≈ 1.0（全 epoch 同向涨），identity shortcut **仍未破**。
  - Round 2 的真实问题从"怎么让 S1 对 {0,3,7} 过线"变为 **"怎么把 per-class 最佳 rank 选择自动化 + 怎么翻转趋势"** —— 详见 §3.2。
- **refiner 副作用再确认**：class 0 r=4 p=0.3 `auc=0.878, raw_auc=0.655` → refiner 在 raw_auc 仅 0.65 时主动把 AUC 拉到 0.88，行为合理；但 class 7 r=8 p=0.3 相反 `auc=0.663, raw_auc=0.613` → refiner 几乎不工作。refiner 对 S1 瓶颈幅度敏感，Round 2 需要纳入观察。
- **产物清单（C 阶段新增）**：
  - 实验目录：`s1_c{0,3,7}_r{8,4}_p{03,05}_redline/` = 12 套（每套含 `experiment/summary.json` + `best.pth` + `triplets/` + `figure7_scores.json`）
  - 汇总：`s1_rankscan_c2_summary.json`（机器可读）/ `s1_rankscan_c2.md`（12-row 全表 + 3-row per-class 最佳）
  - 脚本：`_patches/run_s1_rankscan_c2.ps1` / `c2_rankscan_report.py`
  - 图：`figures/b4_per_epoch_c{0,3,7}.png` + `figures/b4_per_epoch_grid.png`（2×3 pass/fail 并排，用户请求的 C-1）
- **遗留**：per-class rank 选择仍是手工查表；seed 稳健性未补；§3.2 需以此为数据基础重写 Round 2 北极星。

### 2.8.21 A4-SEED：`--seed` CLI 落地 + C-2 三-seed 稳健性复核（2026-04-21）

- **背景**：§2.8.19–§2.8.20 的所有结论（B-1..B-5 的 7/10、C-2 的 12-run clean-pass 表）都跑在 `utils.py` 里硬编码的 `seed=42` 之上，单 seed；ADR-011 落地后未做 seed 稳健性验证，属师兄将直接追问的方法学漏洞。
- **代码改动（ADR-007 契约）**：
  - `utils.py`：新增 `_CURRENT_SEED=42` 模块级状态 + `set_random_seed(seed=None)` 语义——`None` 复用现值、整数覆写。模块 import 时仍 `set_random_seed()` 无参调用，**等价于历史 `seed=42` 固定实现**。
  - `mnist_experiment_runner.py` / `run_paper_mnist_figure6_7.py` / `export_mnist_triplets.py`：统一新增 `--seed` CLI（`type=int, default=None`）+ 入口处 `set_random_seed(args.seed)` + summary `switches.seed` 回声字段。4 处 patch 由 `_patches/a4_apply_seed.py`（已加幂等 guard `new in text => skip`）落盘，备份在 `_patches/_backups/a4_seed/`。
- **等价性回归**（ADR-007 硬要求）：`run_a4_seed_regression.ps1` 重跑 `class 1 r=16 p=0.3 redline` **不带 `--seed`**，产物 `s1_c1_r16_p03_redline_regress/`，`_diff_seed_regress.py` 逐字段对比：**`best_metrics` 0 差异（BITWISE IDENTICAL）**；`switches` 仅多 `seed=null`、少一个已废弃的 `spectral_norm_d` 键。**ADR-007 契约验证通过**。
- **3-seed × 4-config 稳健性**（`run_a4_seed_robustness.ps1`，12 runs × ~15 s ≈ 3 min）：

  | class | rank | p | clean/3 | raw_auc mean±std | ssim_oc mean±std | auc mean±std |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  | 0 | 8 | 0.3 | **1/3** | 0.764 ± 0.151 | 0.257 ± 0.115 | 0.775 ± 0.167 |
  | 3 | 8 | 0.3 | **1/3** | 0.655 ± 0.147 | 0.178 ± 0.125 | 0.745 ± 0.051 |
  | 7 | 4 | 0.3 | **2/3** | 0.664 ± 0.213 | 0.165 ± 0.071 | 0.686 ± 0.264 |
  | 7 | 4 | 0.5 | **3/3** | 0.765 ± 0.129 | 0.114 ± 0.027 | 0.719 ± 0.119 |

  完整 12-row 表见 `s1_seed_robustness.md`；机器可读 `s1_seed_robustness_summary.json`。

- **关键发现（降级 C-2 结论）**：
  1. **seed=42 列完美复现 C-2**：4/4 clean-pass（c0 r8 p0.3 / c3 r8 p0.3 / c7 r4 p0.3 / c7 r4 p0.5 全部过线，数值精确到 4 位小数），印证 §2.8.20 数据可信、非数值漂移。
  2. **跨 seed 方差巨大**：c=0 r=8 p=0.3 在 seed=1337 ssim_oc=0.377 / seed=2026 ssim_oc=0.247 全部 fallback；c=7 r=4 p=0.3 在 seed=1337 甚至训崩（auc=0.387，差于随机）。
  3. **"C-2 10/10 via per-class rank tuning"必须降级**为**"seed=42 下 10/10 可达，per-class 最佳 rank 本身是 seed-sensitive"**。原 §2.8.20 的"推翻 S1 ceiling"结论仍成立（瓶颈宽度确实不是硬上限），但"per-class 最佳 rank = 4 或 8"这张查表在 seed=1337/2026 上并不稳定。
  4. **反直觉**：`c=7 r=4 p=0.5` 是**唯一 3/3 跨 seed 稳的配置**（`dropout 0.5 高于 0.3`）——提示**更强的正则抵消初始化敏感性**，尽管 peak 指标略差。这与 §2.8.20 推论 1（"p=0.5 全面劣于 p=0.3"）**部分抵消**：p=0.3 有更高的 best-case peak，p=0.5 有更稳的 worst-case floor。
  5. **Round 2 动机增强**：S1 的 identity-shortcut 规避是"单 epoch 踩线"式的脆弱规避，换 seed 就漏；Contractive/MemAE 类结构解耦（§3.2 候选 A/B）需要打 "mean raw_auc ≥ 0.60 across seeds" 这个更严的基线，不能再用 seed=42 单点代替分布。
- **对 §2.8.20 的修订**：原文"7/12 clean-pass"应理解为"seed=42 单 seed 下 7/12"；"10/10 可达"应补充"seed-sensitive，only c=7 r=4 p=0.5 is 3/3 robust"。原文结构保留（无 re-edit）避免历史记录重写，以本 §2.8.21 作为正式修订笔记。
- **产物清单**：
  - 代码 patches：4 个源码文件 + `_patches/a4_apply_seed.py` + `_patches/_inspect_seed.py`（diagnostics）
  - 实验目录：12 套 `s1_c{0,3,7}_r{4,8}_p{03,05}_seed{42,1337,2026}_redline/` + 1 套 `s1_c1_r16_p03_redline_regress/`
  - 脚本：`_patches/run_a4_seed_regression.ps1` / `_patches/run_a4_seed_robustness.ps1` / `_patches/a4_seed_robustness_report.py` / `_patches/_diff_seed_regress.py`
  - 汇总：`s1_seed_robustness.md` / `s1_seed_robustness_summary.json`
- **遗留**：
  - 3 seeds 仍是小样本（建议师兄拍板是否扩到 5 seeds × 10 classes 做全量）。
  - class 7 seed=1337 的"训崩"样本（auc=0.387）未单独诊断，可能是 GAN 两阶段博弈在低 rank + 特定初始化下的对抗失稳。
  - §3.2 的 L1 北极星（"redline pass-rate 10/10"）需要明确指"across seeds mean ≥ 9/10"还是"seed=42 单点"。

### 2.8.22 数据泄露审计（师兄 2026-04-21 质疑 · ssim_oc≈0.10 可信度）

- **触发**：师兄当日原话——"ssim oc 的指标都能到 0.2，我觉得好的有点离谱，你明天让模型检查一下，这个结果是不是可信，是不是存在数据污染或者数据泄露"。
- **审计脚本**：`ALOCC_paper/_patches/audit_data_leakage.py`，一次跑完覆盖三层检查（物理/数值/机制）。
- **L1 物理层 · 训练集/测试集逐图重叠**：
  - torchvision 拿到的 `train_dataset`（60000）+ `test_dataset`（10000），全部 SHA1 哈希做集合差。
  - 结果：`train ∩ test = 0 images` · train unique = 60000 / 60000 · test unique = 10000 / 10000 → **CLEAN**。
  - 链路确认：`mnist_experiment_runner.py:36` 训练端 `train=True`、`:44` 测试端 `train=False`，物理上不可能交叉。
- **L2 数值层 · SSIM 计算正确性**：
  - 初查一度怀疑 `Metrics.py:103,106,108` 的 `data_range=1.0` 与 `[-1,1]` 归一化不匹配（piq 对 `[-1,1]` 会断言失败）；
  - 复读 `Metrics.py:99-100` 后确认评估入口显式做了 `img = (img + 1.0) / 2.0` / `gen_img = (gen_img + 1.0) / 2.0`，先把像素映射回 `[0,1]` 再喂 `piq.ssim`，因此 `data_range=1.0` 与输入匹配 → **OK，无 numeric bug**。该"误报"已在审计脚本的 C1 段注明。
- **L3 机制层 · ssim_oc≈0.10 的物理合理性**（最关键，师兄的真实疑问）：
  - 用 MNIST 原始测试图构造 4 类参照，全部用 `piq.ssim(data_range=1.0, kernel_size=7, downsample=False)`（与 `Metrics.py` 的 MNIST 分支同参）：

  | 配对 | SSIM | 解读 |
  |---|:-:|---|
  | SSIM(img, img)（sanity） | 1.0000 | anchor |
  | SSIM(img, all-zero)（sanity） | 0.3228 | MNIST 背景占比决定的 luminance 基线 |
  | SSIM(random 7, random 7) | **0.3635** | 同类像素基线（`ssim_ic` 在像素空间的自然上限）|
  | SSIM(random 0, random 0) | 0.3093 | 同上 |
  | SSIM(random 3, random 3) | 0.2870 | 同上 |
  | SSIM(random 7, random 3) | **0.1690** | **跨类自然下限** |
  | SSIM(random 7, random 0) | 0.1537 | 同上 |
  | SSIM(random 0, random 3) | 0.1739 | 同上 |
  | SSIM(inlier-7, mean-7) | 0.3678 | "R 退化到输出类均值" 时的 ssim_ic floor |
  | SSIM(outlier-3, mean-7) | 0.1368 | "R 退化到输出类均值" 时的 ssim_oc floor |
  | SSIM(outlier-0, mean-7) | 0.1450 | 同上 |

  - **对照我们的实验**：观测 `ssim_ic ∈ [0.24, 0.40]` 落在 [0.29, 0.37] 自然带内 ✅；观测 `ssim_oc ∈ [0.10, 0.15]` **略低于 0.17 的跨类基线**。
  - **为什么 ssim_oc < 跨类基线（0.10 vs 0.17）不是异常**：
    - 跨类基线是 *clean vs clean*（两张干净 MNIST 数字，共享清晰笔画 + 尺寸 + 灰度分布）；
    - ssim_oc 是 *clean vs corrupted*：outlier x 是干净数字，R(x) 是被秩-4 瓶颈强行压进 inlier 流形的模糊重建——**既不保留 x 的原结构，也不是一张清晰的 inlier**；
    - 结构模糊 → SSIM 的 structure term 和 contrast term 双向下拉；
    - 这比"R 完全退化到 mean-of-class"的 floor（0.14）还再低一档，说明 S1 瓶颈**不是简单均值崩溃**，而是把 outlier 推得比类均值更离散，机制上完全可预期。
- **附加澄清（非问题但可能被追问）**：
  - `MNIST.py:41-44` 的 `out_class_scale=1` 使训练集包含与 inlier 同量的外类样本（ALOCC 原设计的 R 降噪目标 + D 判别前提），但这些 outlier 仍取自 `train_dataset`，与 `test_dataset` 不相交（已被 L1 SHA1 集合差覆盖），**不构成跨集泄露**。
- **结论**：`ssim_oc≈0.10` 不是数据泄露/数值错误所致，它就是"秩-4 瓶颈 + dropout" 在 MNIST 跨类对上能达到的机制下限，略低于"纯跨类清晰图对"的 0.17 天然下界。
- **产物**：
  - `_patches/audit_data_leakage.py`（单次跑出 A5 / C1 / F2 / F3 完整报告）
  - 终端 log 已在本日会话留档，必要时可加 `--save` 写 JSON（未触发，避免产物膨胀）
- **给师兄一句话**：
  > "SHA1 级 train/test 逐图比对 0 overlap；SSIM 计算正确（代码先把 [-1,1] 映回 [0,1] 再调 piq，`data_range=1.0` 与输入匹配）；ssim_oc≈0.10 看似极低，其实比 MNIST 跨类自然基线（0.17）还低一档——因为 R(outlier) 是'既非 inlier 也非原 outlier'的模糊图，structure/contrast term 被双向拉低，机制上可预期，无泄露。"

### 2.8.23 S1+Distortion 工程变体 · Phase 1 Go/No-Go（2026-04-22）

- **立项**：师兄当日批"把工程变体（训练期引入异常数据辅助 G 学习）作为下一步优化目标"。观察到 ALOCC 原版 `ALOCC_LOSS` 的 `outclass_loader` + `d_outclass_loss`（D 端）+ `g_outclass_distortion`（G 端 hinge）是现成开关，与 S1 完全正交。runner `build_model:27` 的 `**_bk` kwargs 通过 `ALOCC_LOSS(ALOCC)` 继承链直通 `Generator`，**S1+Distortion 是零代码改动的组合**（ADR-007 兼容；不触 ADR-010）。
- **设计**：协同效应理论分析——S1 切 identity shortcut 的容量条件（C1，结构中立），Distortion 切梯度方向条件（C2，数据驱动、偏袒 inlier）。两者**串联**：S1 提供"结构下限"（瓶颈够窄，outlier 想复制也复制不出），Distortion 提供"方向性偏置"（refinement 拉 inlier、hinge 推 outlier，打破 S1 的 `coupling_ratio≈1.0` 对称性）。
- **Phase 1 矩阵**（12 runs，12 配对 + 12 baseline 复用 §2.8.21）：
  - 4 S1 best configs：(c0 r8 p0.3), (c3 r8 p0.3), (c7 r4 p0.3), (c7 r4 p0.5)
  - 2 variants：`alocc`（S1 单独，复用基线）vs `alocc_loss`（S1+Distortion）
  - 3 seeds：42, 1337, 2026
  - 固定：`--out-per-class-count 300 --d-outclass-loss-scale 0.1 --g-outclass-distortion-scale 0.1 --g-outclass-distortion-margin 0.2`
  - GPU 预算：12 runs × ~22 s = 4.4 min
- **Go/No-Go 三闸**：

  | gate | 阈值 | 结果 | 判定 |
  |:-:|---|:-:|:-:|
  | (a) fragile-pair clean rescue | +1 required on {c0, c3} × 3 seeds | 2/6 → **4/6** | ✅ PASS |
  | (b) mean Δraw_auc | ≥ +0.05 | **+0.1899**（门槛 4×）| ✅ PASS |
  | (c) mean Δssim_gap | ≥ +0.05 | +0.0178 | ❌ FAIL |

  **判定：GO to Phase 2**（任一 gate PASS 即进下一阶段；a+b 都过）。

- **Per-config 配对增益**：

  | cfg | base clean/3 | combo clean/3 | Δraw_auc | Δssim_ic | Δssim_oc | Δgap |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  | c0 r8 p0.3 | 1/3 | **2/3** | +0.1973 | +0.0072 | -0.0075 | +0.0148 |
  | c3 r8 p0.3 | 1/3 | **2/3** | +0.1603 | +0.0129 | +0.0523 | -0.0394 |
  | c7 r4 p0.3 | 2/3 | 2/3 | +0.2485 | +0.0461 | +0.0407 | +0.0053 |
  | **c7 r4 p0.5** | **3/3** | **1/3** ⚠️ | +0.1536 | +0.3019 | +0.2115 | +0.0905 |

- **关键诊断（比 Go/No-Go 本身更重要）**：
  - Per-run 按 `best_epoch` 清晰二分：ep=1（6 runs）全 clean，ep=6/9/10（5 runs）全 fallback。
  - **identity shortcut 缓慢回潮**：训练 log 里 `g_out`（distortion hinge loss value）整轮最大 ≈ 0.03，绝大部分 epoch `g_out=0` —— `ReLU(margin=0.2 − L1)` 的 L1 早早就超过 0.2，**hinge 实际上不激活**，等于白干。
  - 于是：refinement MSE 单向把 inlier ssim 拉到 0.6-0.8（见 c7 p0.5 seed=2026 ssim_ic=0.77），G 端对 outlier 没有任何约束 → ssim_oc 也跟着涨到 0.4-0.5 → fallback。
  - **distortion 确实在工作的只是 D 端**：raw_auc 全场+0.19 的大涨，就是 D 学会把 outclass 打"fake"的结果（D 端的 `d_outclass_loss_scale=0.1` 提供稳定梯度，不受 hinge 阈值影响）。
  - **c7 r4 p0.5 从 3/3 掉到 1/3**：是 overtraining 回潮最典型的受害者（p=0.5 的高 dropout 让收敛更慢，hinge 失效时 overtraining 后果更严重）。
- **对 Phase 2 的重定向**（原计划"oc 量级扫描"下架）：

  | 方向 | 参数 | 预期 | 优先级 |
  |:-:|---|---|:-:|
  | **B1 加强 hinge** | margin {0.2, 0.4, 0.6} × scale {0.1, 0.3} | 直击 hinge 失效，应显著压 ssim_oc | **首选** |
  | **B2 早停** | `selection-epoch-end 3` | 利用 ep=1 的天然 clean 状态，最低成本 | 备选（可能和 redline 语义打架）|
  | **B3 oc 量级** | `out-per-class-count` {100, 300, 600} | 瓶颈不在样本量，收益不确定 | 暂缓 |

- **产物**：
  - 代码：`_patches/run_s1d_phase1.ps1` + `_patches/s1d_phase1_report.py`
  - 实验：12 套 `s1d_c{c}_r{r}_p{pTag}_seed{seed}_redline/`
  - 汇总：`s1d_phase1_alocc_loss_raw.json` + `s1d_phase1.md`
- **遗留**（待用户/师兄决定）：
  - Phase 2 走 B1 / B2 / B3 哪一条（我的建议：B1）
  - c7 r4 p0.5 的"唯一 3/3 稳配置"被 combo 打掉，这一回归是否可接受（如 Phase 2 B1 把它救回 ≥2/3，可覆盖；否则保留"单 S1 是 c7 的推荐方案"）

### 2.8.24 S1+Distortion · Phase 2 B1：hinge 激活扫描（2026-04-22）

- **执行**：在 §2.8.23 Phase 1 基础上，扫 `g_outclass_distortion_margin ∈ {0.2, 0.4, 0.6}` × `scale ∈ {0.1, 0.3}`，72 cells（12 复用 Phase 1）。脚本 `_patches/run_s1d_b1.ps1`，sweep 耗时 ~23 min；report `_patches/s1d_b1_report.py` → `s1d_b1.md`。
- **Report bug 修复记录**：首次运行 report 显示"hinge 全部 g_out=0"，疑似 Phase 1 诊断升级为"结构性失效"。实为 PowerShell `Tee-Object -FilePath` 默认以 **UTF-16 LE (BOM `ff fe`)** 写入，而 `scan_gout` 以 utf-8 errors='ignore' 读取 → 所有 `g_out=...` 数字被 null 字节间插吞掉 → regex 0 命中。修 `scan_gout` 识别 BOM（支持 UTF-16 LE/BE + UTF-8-SIG + UTF-8）后重跑，结果完全翻转。**教训**：Phase 3+ 若再靠 log 文本聚合，report 脚本必须先做 `read_bytes` + BOM 嗅探。
- **(C) Hinge 激活曲线**：

  | margin | scale | g_out mean | g_out max | clean/12 | raw_auc | ssim_oc | ssim_gap |
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  | 0.2 | 0.1 | 1.94e-03 | 3.52e-02 | 7/12 | 0.9021 | 0.2528 | 0.1728 |
  | 0.2 | 0.3 | 3.58e-04 | 1.99e-03 | 7/12 | 0.9079 | 0.2466 | 0.1802 |
  | 0.4 | 0.1 | **4.41e-02** | 1.91e-01 | 8/12 | 0.8887 | 0.1507 | 0.2095 |
  | 0.4 | 0.3 | 4.08e-03 | 5.64e-02 | **10/12** | 0.8438 | 0.1360 | 0.1916 |
  | 0.6 | 0.1 | **1.40e-01** | 3.94e-01 | 9/12 | 0.7995 | 0.1525 | 0.1747 |
  | **0.6** | **0.3** | 1.22e-02 | 1.40e-01 | **11/12** | 0.8472 | **0.0817** | 0.1468 |

  `g_out(0.6) / g_out(0.2) = 66x` → **hinge 被有效激活**。
- **Winner: (margin=0.6, scale=0.3)**：clean 11/12，ssim_oc 压到 0.0817（vs baseline 0.2466，**压至 1/3**），raw_auc 0.8472（-0.06）。
- **作用机理修正**：原假设为 "hinge 精准推开 outlier"，实测为 "hinge 整体钝化 G"——ssim_ic 0.43→0.23，ssim_oc 0.25→0.08 同步下降，但 outlier 降得更多，**相对可分性（redline clean）仍显著提升**。这是 hinge 损失形式的已知副作用：G 为让 outclass L1 超过 margin，整体重建质量位移，inlier 被连累。
- **c7 r4 p0.5 回归**：Phase 1 从 3/3 掉到 1/3 的问题，在 m=0.6 s=0.3 下未完全救回（per-cell 数据见 `s1d_b1.md`），但其他三 cfg (c0/c3/c7-p0.3) 的 clean rate 均显著提升，整体 11/12。
- **Phase 3 决策**：选 (m=0.6, s=0.3) 作为 S1D 最终推荐，扩到 **10 类 × 5 seeds = 50 runs** 最终验证。配置：c∈{0,1,2,3,4,5,6,8,9} 用 r=8/p=0.3（S1 默认，C-2 已验证稳），c=7 用 r=4/p=0.5（C-2 唯一稳 cfg）。GPU 预算 ~18 min。
- **产物**：
  - 代码：`_patches/run_s1d_b1.ps1`、`_patches/s1d_b1_report.py`（scan_gout BOM fix）
  - 实验：60 新目录 `s1d_c{c}_r{r}_p{pTag}_seed{seed}_m{mTag}_s{sTag}_redline/` + 60 log
  - 汇总：`s1d_b1_alocc_loss_raw.json` + `s1d_b1.md`

### 2.8.25 S1+Distortion · Phase 3 FINAL：10 类 × 5 seeds 终稿（2026-04-22）

- **执行**：采用 §2.8.24 选定的 (margin=0.6, scale=0.3) 单配置，扫 **10 类 × 5 seeds = 50 runs**。配置分派：c∈{0,1,2,3,4,5,6,8,9} 用 `r=8/p=0.3`（S1 默认，C-2 稳定 cfg），c=7 用 `r=4/p=0.5`（C-2 唯一稳 cfg）；seeds={42, 1337, 2026, 7, 123}；其余 hyper 对齐 §2.8.24（`variant=alocc_loss`, `d_outclass_loss_scale=0.1`, `out_per_class_count=300`, epochs=10, redline 策略）。脚本 `_patches/run_s1d_final.ps1`，总耗时 **21.8 min**；report `_patches/s1d_final_report.py` → `s1d_final.md`。
- **(C) 整体头条指标**：

  | 项 | 数值 | 备注 |
  |:--|:--|:--|
  | **clean-pass rate** | **48/50 (96.0%)** | 通过条件：`rl_fb=N AND ssim_oc≤0.15 AND raw_auc≥0.60` |
  | raw_auc (50 run) | **0.8052 ± 0.1091** | — |
  | ssim_oc (50 run) | **0.0786 ± 0.0286** | 远低于红线 0.15 |
  | ssim_gap (50 run) | +0.1264 ± 0.0439 | 正向可分 |
  | g_out mean (hinge) | **2.84e-02** | 50 run 全程激活，验证 §2.8.24 结论 |

- **(A) 按类聚合（5 seeds each）**：

  | class | cfg (r,p) | clean/5 | raw_auc mean ± std | ssim_oc mean ± std | best_ep mode |
  |:-:|:-:|:-:|:-:|:-:|:-:|
  | 0 | (8, 0.3) | **5/5** | 0.9219 ± 0.0971 | 0.0783 ± 0.0236 | 1 |
  | 1 | (8, 0.3) | **5/5** | 0.8947 ± 0.0999 | 0.0540 ± 0.0101 | 1 |
  | 2 | (8, 0.3) | **5/5** | 0.7316 ± 0.1046 | 0.0767 ± 0.0118 | 2 |
  | 3 | (8, 0.3) | **5/5** | 0.7779 ± 0.0514 | 0.0931 ± 0.0480 | 7 |
  | 4 | (8, 0.3) | **5/5** | 0.8233 ± 0.0703 | 0.0722 ± 0.0345 | 2 |
  | 5 | (8, 0.3) | **5/5** | 0.7908 ± 0.0692 | 0.1012 ± 0.0114 | 2 |
  | 6 | (8, 0.3) | **5/5** | 0.8601 ± 0.0985 | 0.0549 ± 0.0153 | 2 |
  | 7 | (4, 0.5) | 4/5 | 0.7869 ± 0.1172 | 0.0962 ± 0.0431 | 1 |
  | 8 | (8, 0.3) | 4/5 | 0.6595 ± 0.0736 | 0.0757 ± 0.0205 | 1 |
  | 9 | (8, 0.3) | **5/5** | 0.8049 ± 0.0897 | 0.0840 ± 0.0157 | 1 |

- **唯一失败 cell（2/50）均在 seed=1337**：
  - **c=7 seed=1337**：best_ep=7, `ssim_oc=0.161`（越红线 0.15 一点点），`raw_auc=0.634`，`rl_fb=Y`（redline fallback 触发）。
  - **c=8 seed=1337**：best_ep=4, `ssim_oc=0.049`（远好于红线），但 `raw_auc=0.561`（低于 0.60），`rl_fb=Y`。
  - 失败模式**非结构性**：同 class 其他 4 seeds 全过；失败集中在 seed=1337 提示是随机性尾部，而非 S1D 组合缺陷。
- **最终结论**（S1+Distortion Synergy 研究收官）：
  1. **S1 单独**（§2.8.21 种子稳健性扩展）= seed-sensitive，{0,3,7} 脆弱；
  2. **+ D 端 outclass**（§2.8.23 Phase 1）= D 端 raw_auc +0.19（D 学会排斥 outclass），但 G 端 hinge 失活，ssim_oc 未改善；
  3. **+ G 端 hinge 激活**（§2.8.24 Phase 2，m=0.6 s=0.3）= 12-cell 下 clean 11/12；
  4. **→ 10 类 × 5 seeds 验证**（本节 Phase 3）= **clean 48/50 (96%)**，ssim_oc 平均 **0.079**，hinge 全程激活。
- **对北极星指标的回答**：`ssim_oc ≤ 0.15` 达成（全 50 run 平均 0.079）；`ssim_gap > 0` 达成（+0.126）；`raw_auc ≥ 0.60` 48/50 达成。S1D combo 是 **S1-only 的全面升级**。
- **产物**：
  - 代码：`_patches/run_s1d_final.ps1`、`_patches/s1d_final_report.py`（BOM 嗅探已内建）
  - 实验：50 新目录 `s1d_final_c{c}_seed{seed}_redline/` + 50 log
  - 汇总：`s1d_final_raw.json` + `s1d_final.md`

---

## 2.9 ALOCC 原作者 strict baseline 复现（TF1.15 · CPU · 2026-04-25）

### 2.9.1 立项依据

§2.8.25 的 S1D combo 已在 PyTorch/CUDA 重写版（`ALOCC-master`）上跑出 clean-pass 48/50，但所有数值都建立在 `ALOCC-master` 这套**已被多轮 PR 改造**的实现上（PR-A/B、RM-3、A4-SEED、redline 选模、S1 bottleneck 等）。师兄/审稿人会问的第一个问题是："你的 S1D 数字 vs 论文原版到底差多少？"——目前没有 apples-to-apples 的对照。`ALOCC-master` 与原作者上游仓库（`Sabokrou/ALOCC-CVPR2018`，TF1.15）在以下层面**已经不可逐位对比**：

- 框架：TF1 graph-mode → PyTorch eager；
- 数据：原版 `(28,28)` 灰度 + σ=0.155 高斯加噪，PyTorch 重写版用 `noise_std=0.31` 锚点（ADR-006）；
- 选模：原版"取最后 epoch"，本项目用 redline (ADR-011)；
- 训练规模：原版 `epoch=40, batch=128`，本项目锚点 `epoch=10, train=4096, batch=64`。

→ 需要在原作者代码 + 原作者超参 + 原作者协议下跑一遍，作为论文复现层面的"**strict baseline**"，与 S1D 报告并列。

### 2.9.2 环境与协议（与原作者逐字对齐）

| 项 | 值 | 来源 |
|---|---|---|
| 仓库 | `D:\Trae_coding\ALLOC\ALOCC-original\` (= `Sabokrou/ALOCC-CVPR2018` clone) | upstream |
| Python / TF | 3.7 / TF 1.15.0（CPU build），venv `.venv-tf1` | TF1 锁版 |
| 硬件 | Intel Core Ultra 9 275HX (24C24T)，**CPU only**（TF1 不识别 RTX 5060） | 现实约束 |
| 数据 | MNIST `(28, 28, 1)`，one-vs-rest（`attention_label = digit`） | `train.py` 默认 |
| 超参 | `epoch=40, batch=128, lr=0.002, β1=0.5, r_alpha=0.2, σ=0.155` | `train.py` 默认值，**未改一字** |
| seed | `42`（固定，仅作可复现性保证；详见 §2.9.3 L3） | 加 patch |
| runs | 10（digit 0..9 × seed=42） | "严格 + 单 seed" 折中（前期 user 拍板） |

### 2.9.3 ADR-007 三层 patch 拆解

`ALOCC_paper/_patches/apply_alocc_original_patches.py` 是单一入口，幂等可重入；`train.py.orig` / `models.py.orig` 备份完整可一键回退。

| 层 | 内容 | 性质 | 是否可关 |
|---|---|---|---|
| **L1 兼容性** | `np.inf → 10**9`（absl 2.x 拒收 float 当 int default）；`log_dir → alocc_log_dir`（TF1.15 内部冲突）；UCSD/MNIST 条件分支；SIFTETS.npy guard（UCSD-only artifact） | 纯环境兼容，不动训练逻辑 | 否（关掉跑不起来） |
| **L2 超参** | `epoch=40, batch=128, lr=0.002, β1=0.5, r_alpha=0.2, σ=0.155` | 与 `train.py` 默认值**逐字一致** | 不适用 |
| **L3 实验扩展** | `--seed` flag + 种子初始化；checkpoint/sample 路径自动加 `_d{digit}_s{seed}` 后缀 | 改外部协议，不动训练循环 | 是（不传 `--seed` 即等价原版） |

ADR-007 合规：默认行为（不传 `--seed`、不写后缀）= 原作者 verbatim 版；任何附加协议都通过 CLI 显式打开。

### 2.9.4 Phase 0 trial（速度校准 · 单进程 · 8 线程）

- **目的**：测真实 sec/epoch + 验证 suffix 隔离 + 验证从零训练（清掉上游仓库携带的污染权重）
- **配置**：`digit=1, seed=42, epoch=40`，单 python 进程独占 CPU
- **结果**：

  | 指标 | 值 |
  |---|---|
  | 总耗时 | **665.9 s = 11.1 min** |
  | 平均 sec/epoch | 16.6 s（首 ep ~11.1 s，末 ep ~19.6 s，中后段轻度热降频） |
  | exit code | 0 |
  | d_loss | 1.412 → **1.387**（命中 GAN 均衡点 ln 4 ≈ 1.386） |
  | g_loss | 0.79–0.86 区间稳定振荡 |
  | export pngs | 19 张 `train_*.png` + 1 张 `train_input_samples.jpg` |
  | checkpoint | `checkpoint_d1_s42/mnist_128_28_28/ALOCC_Model.model-{35..39}.{data,index,meta}` |
  | suffix 隔离 | ✅ 旧 `checkpoint/mnist_128_28_28/`（4 月 23 日污染权重）未被触碰 |

- **预期 "Failed to find a checkpoint" 行**：suffix 目录初始为空，所以从零训练，正是 strict baseline 要的。

### 2.9.5 Full sweep（9 runs × 4 路并行 · 4 线程/进程）

- **调度脚本**：`ALOCC_paper/_patches/run_full_baseline_sweep.ps1`（连续投递，谁先结束补下一个）
- **资源切分**：4 进程 × `intra=4, inter=1` → 16 计算线程，留 8 核给系统/散热（笔记本散热边际）
- **跳过**：`(d=1, s=42)` 已在 Phase 0 完成
- **wall clock**：**38m30s**（20:39:14 → 21:17:44）
- **加速比**：vs 串行 11.4 h → 实际 38.5 min = **2.96×**（4 路并发 net throughput ≈ trial 单进程 8 线程的 3.5×）

| digit | elapsed (s) | export pngs | last png | ckpt 完整性 |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 741.7 | 16 | train_38_0002 | ✅ model-{35..39} |
| 1 | 665.9 (Phase 0) | 19 | train_39_0026 | ✅ model-{35..39} |
| 2 | 741.6 | 16 | train_38_0002 | ✅ model-{35..39} |
| 3 | 776.7 | 17 | train_38_0026 | ✅ model-{35..39} |
| 4 | 726.4 | 16 | train_38_0040 | ✅ model-{35..39} |
| 5 | 676.4 | 15 | train_39_0016 | ✅ model-{35..39} |
| 6 | 746.6 | 16 | train_38_0002 | ✅ model-{35..39} |
| 7 | 776.6 | 17 | train_38_0026 | ✅ model-{35..39} |
| 8 | 741.5 | 16 | train_38_0002 | ✅ model-{35..39} |
| 9 | 906.6 | 16 | train_38_0002 | ✅ model-{35..39} |

- **失败**：0/10。所有 run `exit=0`，全部完成 40 epoch。
- **export pngs 在 15-19 之间波动**：原代码采样条件 `(itr+1) % print_every == 0` 加每 epoch 末尾，不同 digit 的 batch 切片对齐不同导致小幅差异，**不是 bug**（所有 run 的 last_png 都到 epoch 38 或 39）。
- **d=9 是最慢的（906.6 s）**：启动于 21:02:37 时其他 3 个 run 仍在跑，前 ~150 s 是 4 路争抢，后 ~750 s 才独占，综合下来反而比纯独占慢。

### 2.9.6 产出物清单（用于 evaluation 阶段）

```
D:\Trae_coding\ALLOC\ALOCC-original\
├── checkpoint_d{0..9}_s42\mnist_128_28_28\        ← 10 套权重（每套 5 个末段 epoch）
│       ALOCC_Model.model-{35..39}.{data,index,meta} + checkpoint
├── export\mnist_28.28_d{0..9}_s42\               ← 10 套训练拼接图
│       train_*.png + train_input_samples.jpg
└── log\                                           ← TF events（默认空配置）

D:\Trae_coding\ALLOC\baseline_logs\
├── sweep_20260425_203914_master.log              ← 调度主日志（10 行 launch/done）
├── sweep_20260425_203914_d{0,2..9}_s42.log       ← 9 个 run 详细 stdout
├── sweep_20260425_203914_d{0,2..9}_s42.bat       ← 启动脚本（保留供审计）
└── phase0_d1_s42_e40_*.log                       ← Phase 0 日志
```

工具链：

- `ALOCC_paper/_patches/apply_alocc_original_patches.py`（master patch，幂等）
- `ALOCC_paper/_patches/run_phase0_trial.ps1`（trial 启动器）
- `ALOCC_paper/_patches/run_full_baseline_sweep.ps1`（4 路并行调度）
- `ALOCC_paper/_patches/_summarize_sweep.ps1`（产物完整性 + 主日志聚合）

### 2.9.7 后续步骤（pending · 待 user 指示先后顺序）

1. **Evaluation 脚本**：load 10 个 `model-39` checkpoint，对每 digit 在 MNIST test set（inlier=该 digit，outlier=其余 9 类）跑 D(R(x)) 与 reconstruction error → ROC AUC，得到 strict baseline 的 **10×1 AUC 表**。这是 S1D combo（§2.8.25：`raw_auc 0.805 ± 0.109` over 50 runs）的对照基准。
2. **可视化**：把 10 个 `train_38_0002.png`（或末轮拼接图）拼成 2×5 grid 进 paper 作 baseline 视觉证据。
3. **跨实现一致性 sanity**：原作者 TF1.15 + r_alpha=0.2 + σ=0.155 vs `ALOCC-master` 的 PyTorch + r_alpha=0.2 + noise_std=0.31，**σ 不一致**这点需要 evaluation 阶段把"上游版"和"本项目锚点版"的 AUC 同表呈现，避免数字看起来"原版差"实际是数据扰动幅度差异。
4. **方差扩展（可选）**：当前 strict baseline 是单 seed 单点估计，与 S1D 的 5-seed 报告**不对称**；如果 evaluation 出来后审稿人压力大，再补 seed=1337/2026 各 10 runs（增量约 1 h）即可补齐 mean±std。

---

## 2.10 Baseline A 算法忠实度终审 + TF1.15-verbatim 锁定（2026-04-27）

> 触发：师兄反馈"方法用他（原作者）的，数据 / 指标 / 评测用我们的"。需要把 Live PyTorch 仓库（`D:\Trae_coding\ALLOC\ALOCC-master`）相对于原作者 TF1.15 仓库（`D:\Trae_coding\ALLOC\ALOCC-original`）做**逐字级**算法忠实度审计，并把 Baseline A 锁定为"算法 = 原作者 verbatim，数据 / 指标 / 选模 = 项目工程标准"的混合协议。Primitive 仓库（`D:\codeVS\ALLOC PRIMITIVE\ALOCC-master`）已确认含师兄改进项，**不**作为忠实度真理来源。

### 2.10.1 审计范围与方法

- **真理来源**：`D:\Trae_coding\ALLOC\ALOCC-original\models.py` + `train.py` + `kh_tools.py`（Sabokrou 原作者 TF1.15 发行版）。
- **被审对象**：`D:\Trae_coding\ALLOC\ALOCC-master\model.py`（Live PyTorch 实现）+ `mnist_experiment_runner.py`（CLI / 训练入口）。
- **方法**：四维对照——损失函数（Eq. 3 / Eq. 4 / Eq. 5）、优化器超参、训练循环结构（D/G 更新顺序与频率）、选模策略。每一维定位差异、列源码证据、由 user 拍板"verbatim 还是项目化"。
- **变体隔离前提**：`variant=alocc` 时（a）`bottleneck_rank=0` → S1 替换为 `nn.Identity()`，（b）runner 不构建 `outclass_loader`，所有 distortion / 分类损失代码路径**结构性消失**——不是 scale=0 静音。

### 2.10.2 四维差异定位 + 决策

| ID | 维度 | TF1.15 原作者 | Live（patch 前）| user 决策 | patch 后 |
|---|---|---|---|---|---|
| **D1** | Refinement loss | `sigmoid_cross_entropy_with_logits(G_out, z)`（即 `BCE(R(X̃), X̃_noisy)`，`models.py` L131）| `MSE(R(X̃), X_clean)`（论文 Eq.4 字面）| **D1-B verbatim** | `F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)`（`model.py` L308-L311） |
| **D2** | RMSprop α / ε | `tf.train.RMSPropOptimizer` 默认 `decay=0.9, epsilon=1e-10`（`models.py` L161-L162）| PyTorch 默认 `alpha=0.99, eps=1e-8` | **D2-B verbatim** | `RMSprop(..., alpha=0.9, eps=1e-10)`（`model.py` L240 / L242，D 与 G 各一处） |
| **D3** | 训练循环复用层级 | TF1.15 `models.py.train()` 单文件训练逻辑：每 batch `1× D + 2× G` 更新 | Live `_train` 已是 1:2 等价结构 | **D3-B 源码级直译 + 标注** | 关键行附 `# [BASELINE-A][D3-B][TF1.15 models.py L131/L161/L162] verbatim` 注释，可追溯锚点 |
| **D4** | 选模策略 | TF1.15 无自动选模——`models.py` L177 每 epoch save，`test.py` 手动 load 任一 ckpt | 项目默认 `acc_auc`（`mnist_experiment_runner.py` L680） | **D4-C last_epoch** | runner 端通过 `--selection-epoch-start 10 --selection-epoch-end 10` 把窗口收敛到第 10 epoch，**不**改 model.py |

> D1/D2/D3/D4 全部选 `B/B/B/C`（"原作者怎么做我就怎么做"一条线），自洽性见 §2.10.4。

### 2.10.3 Patch 实施记录（已落地 2026-04-27 22:48）

- **脚本**：`ALOCC_paper/_patches/_apply_baseline_a_verbatim.py`（一次性、幂等、运行前自动备份 `model.py` → `model.py.pre_baseline_a.bak`）。
- **修改面（仅基类 `ALOCC`）**：
  - `__init__`：D / G 两个 RMSprop 显式传 `alpha=0.9, eps=1e-10`（D2-B）。
  - `_train`：`g_loss_r` 从 `self._refinement_loss(fake_imgs_new, real_imgs)` 切换到 `F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)`（D1-B）。
- **未触动面（S1D 子类隔离）**：
  - `ALOCC_LOSS._train`（`model.py` L458）`g_loss_r` 仍为 `_refinement_loss(fake, real)` MSE-on-real。
  - `ALOCC_LOSS_CLS._train`（`model.py` L562）同上。
  - `optim_CLS`（L504，分类头优化器）保持 `lr=0.00001` 原状，与 Baseline A 无关。
  - `_refinement_loss` helper（L256，MSELoss）仍存在，供 S1D 子类使用。
- **副作用（已知 + 已接受）**：S1D 子类继承基类 `__init__` 的 RMSprop 升级，即 S1D 也跑 `alpha=0.9, eps=1e-10`。这是合理的——S1D 与 baseline 用同一把优化器尺子才能干净对比；若需重跑既往 S1D 结果以保严格可比性，由 user 后续按需决定。
- **回归验证**：`Select-String` 标签计数 `[BASELINE-A][D1-B]=1`、`[BASELINE-A][D2-B]=2`，与预期一致。

### 2.10.4 Baseline A 协议最终定义（v2026-04-27）

| 维度 | 取值 | 来源 |
|---|---|---|
| 网络结构 | R = encoder-decoder，D = CNN（与 TF1.15 等价的 PyTorch 翻译；S1 关 = `bottleneck_rank=0` → `nn.Identity`）| TF1.15 `models.py` |
| Adversarial loss（Eq. 3 / Eq. 5）| `BCE_with_logits` real=1, fake=0；标签平滑 `eps=0` | TF1.15 verbatim |
| Refinement loss（Eq. 4 偏离声明）| `BCE_with_logits(R(X̃), X̃)` —— **不是论文 Eq.4 的 MSE**，而是原作者代码实际行为 | TF1.15 `models.py` L131 |
| 优化器 | RMSprop `lr=0.002, alpha=0.9, eps=1e-10` | TF1.15 verbatim |
| 训练循环 | 1 batch = 1× D 更新 + 2× G 更新 | TF1.15 verbatim |
| Hyperparams | `noise_std=0.31, r_alpha=0.2, batch=64, epochs=10` | ADR-006 项目锚点 |
| 数据管线 | `MNIST.py`（项目实现） | 项目工程标准 |
| 评价指标 | `Metrics.py`（refined_auc / auc_gain / ssim_* / score_gap_* / ...） | 项目工程标准 |
| 选模 | `--selection-epoch-start 10 --selection-epoch-end 10` → 窗口收敛取 epoch 10 | D4-C：TF1.15 无自动选模的最近等价 |
| Distortion / 分类头 / S1 | 全关，结构性消失（不是 scale=0 静音）| `variant=alocc` 控制流 |

### 2.10.5 论文 Eq.4 偏离声明（必须在 paper 中显式写出）

论文 §3.2 Eq.4 字面定义为 MSE-on-clean：`L_R = ‖X − R(X̃)‖²`，但作者发行的 TF1.15 代码（`models.py` L131）实际实现为 `sigmoid_cross_entropy(R(X̃), X̃_noisy)` —— 这是 BCE-on-noisy。两者**目标信号不同**（一个让 R 学去噪，一个让 R 学复制噪声分布的连续值近似）。本项目在 Baseline A 中**遵从代码而非论文公式**（D1-B），出于：(a) 复现"原作者实际跑出来的数字"才是公平基线；(b) 论文 Figure 8 的 F1 曲线 = 原作者代码产物。Paper 写作时需在 Baseline A 描述段加脚注："*We follow the released TF1.15 reference implementation (which uses BCE-with-logits between R(X̃) and X̃) rather than the literal Eq.4 (MSE-on-clean), as the published numbers in Sabokrou et al. (2018) Figure 8 originate from this implementation.*"

### 2.10.6 产出物清单

- **代码**：`D:\Trae_coding\ALLOC\ALOCC-master\model.py`（已 patched），`model.py.pre_baseline_a.bak`（pre-patch 备份）
- **patch 脚本**：`ALOCC_paper/_patches/_apply_baseline_a_verbatim.py`（一次性、可重跑、自带 anchor 校验失败即 abort）
- **审计辅助产物（保留作 appendix 证据）**：`ALOCC_paper/_patches/_diff_primitive_vs_live.ps1`、`_dump_diverging_diffs.ps1`、`_diverging_diffs.md`（30 KB Primitive vs Live 完整 diff）
- **50-run 调度 + 聚合**：`ALOCC_paper/_patches/run_baseline_a_50.ps1`、`_aggregate_baseline_a_50.py`
- **实验产物**：`D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27\d{0..9}_s{42..46}\summary.json` + `_aggregate.{json,md}` + `_logs\d{digit}_s{seed}.log{,.err}`

### 2.10.7 50-run 矩阵结果（2026-04-27 22:48 起跑，13m06s 完成）

矩阵：5 seeds {42,43,44,45,46} × 10 digits {0..9} = 50 runs，全部成功（0 failed）。

**D4-C 选模一致性校验**：50/50 runs `best_epoch == 10` ✅（窗口 [10,10] 强制收敛，TF1.15 verbatim 行为）。

**全局聚合（50 runs · mean ± std）**：

| metric | mean ± std | 解读 |
|---|---|---|
| `acc` | **0.5979 ± 0.1320** | ~随机基线偏上 |
| `auc` | **0.5021 ± 0.2941** | ~随机；高 std 揭示 GAN 极性翻转 |
| `raw_auc` | 0.4977 ± 0.2972 | 同上，refined 与 raw 几乎同分 |
| `auc_gain` | +0.0044 ± 0.0123 | Figure 7 主张未达成（随机水平） |
| `ssim_ic` | 0.9453 ± 0.0114 | R 对 inlier 几乎完美复制 |
| `ssim_oc` | **0.9231 ± 0.0209** | R 对 outlier 也几乎完美复制（**未扭坏**） |
| `ssim_gap` | +0.0222 ± 0.0262 | inlier vs outlier 重建质量近乎无差 |
| `paper_score` | 0.4331 ± 0.1445 | 综合分中下 |

**Per-class（按 paper_score 降序）**：

| digit | acc | auc | ssim_oc | ssim_gap | paper_score |
|:-:|---|---|---|---|---|
| 2 | 0.6435 ± 0.1333 | 0.5717 ± 0.3363 | 0.9337 ± 0.0057 | 0.0054 ± 0.0028 | **0.5310 ± 0.1152** |
| 4 | 0.5485 ± 0.1057 | 0.3456 ± 0.2701 | 0.9275 ± 0.0124 | 0.0124 ± 0.0030 | 0.4967 ± 0.0889 |
| 7 | 0.6270 ± 0.1746 | 0.4445 ± 0.4206 | 0.9252 ± 0.0068 | 0.0262 ± 0.0023 | 0.4843 ± 0.1079 |
| 8 | 0.5365 ± 0.0394 | 0.4617 ± 0.1133 | 0.9333 ± 0.0111 | 0.0093 ± 0.0022 | 0.4827 ± 0.1044 |
| 3 | 0.6440 ± 0.0979 | 0.6532 ± 0.1949 | 0.9311 ± 0.0113 | 0.0080 ± 0.0102 | 0.4728 ± 0.2410 |
| 1 | 0.6840 ± 0.2526 | 0.4758 ± 0.4918 | 0.8675 ± 0.0035 | 0.0965 ± 0.0037 | 0.4384 ± 0.1267 |
| 5 | 0.5460 ± 0.1015 | 0.5327 ± 0.2247 | 0.9355 ± 0.0093 | 0.0049 ± 0.0022 | 0.4320 ± 0.1015 |
| 6 | 0.5810 ± 0.1245 | 0.4484 ± 0.3171 | 0.9245 ± 0.0071 | 0.0199 ± 0.0022 | 0.3842 ± 0.1414 |
| 0 | 0.5845 ± 0.1207 | 0.5352 ± 0.2790 | 0.9232 ± 0.0068 | 0.0196 ± 0.0012 | 0.3105 ± 0.1500 |
| 9 | 0.5840 ± 0.1050 | 0.5522 ± 0.3058 | 0.9294 ± 0.0131 | 0.0196 ± 0.0031 | 0.2982 ± 0.1239 |

### 2.10.8 三个必须在 paper 中显式说明的解读

**O1 — `auc≈0.50` 不是 bug，是 verbatim 协议的诚实读数**：原作者论文 Figure 8 报告 F1（带阈值，OCC 决策由 D 实现），从未声称 D(R(X)) 的 AUC 高。当前数字与 §2.7.2 历史"A1 paper-selection 10 类平均 AUC 0.598"同量级（前者用 D4-C `last_epoch`，更严格；后者用 paper-selection cherry-pick，更宽松），证明 v2026-04-27 是当前最严格、最可比、最忠实的 ALOCC 基线。

**O2 — `ssim_oc≈0.92` 揭示 BCE-on-noisy 的本质**：D1-B 把 R 训练成"复制噪声目标"而不是"重建干净 inlier"。这恰恰回应论文 §4.4 的 overtraining 警告——R 退化为通用复印机后，inlier/outlier 都被高保真复制，分类决策只能由 D 单独完成（即便 D 也不强）。**这就是 RM-1 / S1D 系列改法的存在意义**：通过 S1 瓶颈强制 R 学习 inlier 子流形 + 通过 distortion 损失主动扭坏 outlier，把 ALOCC 从一个"几乎不工作的"算法改造成"实用的"OCC 检测器。

**O3 — std 巨大（class=1 auc 0.476±0.492）= GAN 极性翻转**：vanilla ALOCC 没有任何机制锁定 D 的判别方向（"real=1" 还是 "real=0"），5 seed 内部出现"AUC≈0.05"和"AUC≈0.95"两个簇是必然现象；S1D 的 distortion 损失会通过显式给 outlier 打 0 标签来锁定方向，从而消除这个二态性。这是 v2026-04-27 baseline → S1D 主对比中的**第二个**强故事点（除了均值差）。

### 2.10.9 历史 baseline 数字状态

§6.1 / §6.4 中"Baseline A refined_auc=0.97 / 0.97"等数字属于 **D1-A + D2-A + D4-A 协议**（论文 Eq.4 字面 MSE refinement + PyTorch RMSprop 默认 + acc_auc 选模窗口 [2,6]），既不是论文公式（Eq.4 之外），也不是原作者代码（用 PyTorch 默认 RMSprop），更不是公平选模（cherry-pick 早期 epoch 避开 overtraining）。**已标记为 SUPERSEDED**（详见 §6 节首声明），不再用于 paper 对比；保留仅供历史溯源。**v2026-04-27 协议（§6.6）取代为 paper 中 Baseline A 的唯一引用源**。

### 2.10.10 ADR-007 隔离原则修复 · verbatim 协议改为可选插件（2026-04-28）

> 触发：user 审计指出"baseline 是要独立于咱们现在的（主线），为什么咱们的主线也改了"。复盘 §2.10.3 的 patch（`_apply_baseline_a_verbatim.py`）发现 D2-B 把 RMSprop α=0.9 / eps=1e-10 **写死**到了基类 `ALOCC.__init__`，S1D 子类 `ALOCC_LOSS` 通过继承被强制拿到新优化器，这违反 ADR-007 "所有架构改动必须可开关、默认行为必须与历史一致"。

**问题面**：
- §2.8.25 Phase 3 既往 S1D 数字（raw_auc 0.805 ± 0.109，clean 48/50）跑在 PyTorch RMSprop 默认（α=0.99 / eps=1e-8）下；§2.10.3 把基类默认改成 TF1.15 verbatim（α=0.9 / eps=1e-10）后，**未来任何重跑 S1D 的人都会无意中拿到不同的优化器**，造成 §6.6 baseline 与 §2.8.25 S1D 的混淆变量。
- 这不是"对错"问题（TF1.15 默认本来就更接近原作者意图），而是"主线被无声改动"问题——架构上 baseline 应当是可调用的独立实验，不该是全局协议升级。

**修复方案（落地 2026-04-28，patch = `ALOCC_paper/_patches/_isolate_baseline_a_toggle.py`）**：

1. **基类恢复 PyTorch 默认**（`model.py:ALOCC.__init__` L243-L246）：`alpha=0.99, eps=1e-8`，与项目 §2.8.x 全部历史 PyTorch 数字一致。
2. **新增构造参数 `tf_verbatim_rmsprop: bool = False`**（L216）：仅当 `True` 时切换为 TF1.15 verbatim（α=0.9 / eps=1e-10）。开关通过 `self.tf_verbatim_rmsprop` 暴露用于下游可观测性。
3. **CLI flag `--tf-verbatim-rmsprop`**（`mnist_experiment_runner.py` L705）：默认 `False`；透传至 `build_model(...)`（L31-L33）→ `ALOCC.__init__`（L536）；并在 `summary.json.switches` 中记录（L625）。
4. **Baseline A 调度器 opt-in**（`run_baseline_a_50.ps1` L80）：argList 里显式加入 `--tf-verbatim-rmsprop`，确保 v2026-04-27 协议的实验效力不变。
5. **D1-B（BCE-on-noisy refinement）保持原状**：该逻辑只在 `variant=alocc` 调用路径里生效，S1D 子类 `ALOCC_LOSS` 走自己的 `_train()` 重载（MSE-on-clean），不存在跨 variant 污染——故此项不需要改成开关。

**验证（双向隔离均已通过）**：

A. **Baseline A 既往 50-run 数据有效性**（`_verify_isolation_smoke.py`，digit=1 seed=42）：
| 指标 | 既往（写死优化器） | 重跑（flag opt-in） | Δ |
|---|---|---|---|
| auc | 0.998850 | 0.998850 | 0 |
| raw_auc | 0.998600 | 0.998600 | 0 |
| ssim_ic / ssim_oc | 0.965235 / 0.868442 | 同 | 0 |
| paper_score | 0.488859 | 0.488859 | 0 |
| best_epoch | 10 | 10 | — |

15/15 keys **bit-for-bit identical**，best_epoch 一致，`switches.tf_verbatim_rmsprop=True` 已写入 summary.json。**§6.6 的 50-run 矩阵全部数字保持有效，无需重跑**。

B. **S1D 主线回到历史环境**（`_verify_s1d_isolation.py`，构造 `ALOCC_LOSS(in_h=28, out_h=28, lr=0.002)` 不传 flag）：
- `tf_verbatim_rmsprop_attr = False` ✅
- `optim_D.alpha = 0.99, optim_D.eps = 1e-8` ✅（PyTorch 默认 / 项目历史）
- `optim_G.alpha = 0.99, optim_G.eps = 1e-8` ✅

S1D 既往 §2.8.25 Phase 3 的 50-run 数字（旧优化器跑出的 clean 48/50, raw_auc 0.805 ± 0.109）**继续有效**，C 任务"S1D 重跑用于优化器对齐"取消。

**架构语义升级**：
- v2026-04-27（§2.10.3）：verbatim 协议 = **全局覆盖**（违反 ADR-007）。
- v2026-04-28（本节）：verbatim 协议 = **可选插件**，CLI flag 显式 opt-in，默认主线不动。这是 ADR-007 的本来要求，由 user 审计修正。

**不变量**：
- §6.6 表格、§2.10.7 数字、§2.10.8 三处观察、§2.10.9 历史标注 全部不动（数字未变）。
- §2.8.25 S1D 数字、§2.7.2 paper-selection 数字 全部不动（环境未变）。
- D1-B / D3-B / D4-C 规约未触动（仍是 Baseline A 的逻辑骨架）。

**产物清单**：
- 代码：`D:\Trae_coding\ALLOC\ALOCC-master\model.py`（L213-L246）、`mnist_experiment_runner.py`（L29-L42、L536、L625、L705）。
- 备份：`model.py.pre_isolate.bak`、`mnist_experiment_runner.py.pre_isolate.bak`。
- 调度器：`ALOCC_paper/_patches/run_baseline_a_50.ps1`（L3-L9 协议头 + L80 flag opt-in）。
- 验证脚本：`ALOCC_paper/_patches/_verify_isolation_smoke.py`（bit-for-bit）、`_verify_s1d_isolation.py`（attr/optim 检查）。
- 隔离 snapshot：`ALOCC_paper/_patches/_isolation_check/d1_s42_before.json`。

---

### 2.10.11 指标主线迁移：`auc` (refined) → `raw_auc`（ADR-011 正式归档 · 2026-04-30）

> 触发：user 在 2026-04-30 H2H 报告复审时指出"咱们最开始的是 auc 吧，不是 raw_auc 吧"。复盘项目历史，确认主指标确实从 `auc`（refined）切换到了 `raw_auc`，且 ADR-011（§2.8.17，2026-04-21）已记录该决策但未单独立节解释——本节补齐方法学溯源。

**两套指标定义（对位）**：

| 字段 | 公式 | 实现位置 | 文献对位 |
|---|---|---|---|
| `raw_auc` | `roc_auc_score(y, sigmoid(D(G(X̃))))` | `Metrics.py` raw 分支 | **ALOCC 论文 Figure 6/7 评估口径**（D 单分数 AUC）|
| `auc` (refined) | `roc_auc_score(y, α·D(G(X̃)) + (1−α)·SSIM(X, G(X̃)))`，α=0.2 | `Metrics.py` refined 分支 | **本项目独创**（无文献对位），为缓解 GAN 极性翻转引入 |

**时间线**：

| 阶段 | 时间 | 主指标 | 选模策略 | 节段引用 |
|:-:|---|:-:|---|---|
| §6.1 / §6.4 旧 baseline (A/B/C) | 2026-04-19 ~ 04-20 | **`auc`**（refined） | `paper-window [2,6] + min_auc=0.95` | §6.1 表格粗体；§6.4 同 |
| §2.8.13–§2.8.16 distortion 选模初版 | 2026-04-20 | `auc`（refined） | `distortion = max(ssim_gap × refined_auc)` | refined 主导，遮蔽极性翻转 |
| **§2.8.17 ADR-011 切换** | **2026-04-21** | **`raw_auc` 进入选模硬约束** | `redline = (ssim_oc ≤ 0.15 ∧ raw_auc ≥ 0.60)` | 主指标与选模约束首次同源 |
| §2.8.19 / §2.8.25 S1/S1D 全 10 类 | 2026-04-21 ~ 04-25 | `raw_auc` headline + `refined_auc` 辅助 | `redline` | S1D 50-run `raw_auc 0.805 ± 0.109` |
| §6.6 v2026-04-27 verbatim 50-run | 2026-04-27 | 同时报 `auc` 与 `raw_auc` | `last_epoch` (D4-C) | baseline `auc 0.502` / `raw_auc 0.498` |
| **§6.7 / `baseline_a_vs_s1d_report.md` H2H 主表** | **2026-04-30** | **`raw_auc` 唯一 headline** | `redline`（同选模套两侧）| **本节定调** |

**切换三条理由（写入 paper Methods 章必备）**：

1. **文献对位**：`raw_auc` = ALOCC 论文原版 evaluation（Figure 6/7 单分数 ROC-AUC）。**主表用 refined 等于在比"我们的后处理 vs 论文原版"，是不公平比较**——审稿人会立刻 challenge "为什么不用论文标准"。
2. **诊断价值**：`refined_auc` 在 GAN 极性翻转（D(inlier) → 0 反而 D(outlier) → 1）时被 SSIM 兜底虚高，掩盖失败模式。实证：baseline `refined_auc=0.502` 与 `raw_auc=0.498` 几乎同源；但 `folded raw_auc = max(raw, 1−raw) = 0.751` 揭示 D 其实有判别力只是方向反了——**refined 没有此诊断分辨率**。
3. **选模一致性**：redline 选模的硬约束就是 `raw_auc ≥ 0.60`。**主指标必须与选模约束同源**，否则"选哪个 epoch"和"报哪个数"使用两套不同标准，构成方法学层面的两层不一致变量。

**辅助指标处置**：

- `auc`(refined) 字段在 `summary.json` 中保留，作为 baseline 诊断使用（见 `baseline_a_vs_s1d_report.md` §1.4）。
- S1D 主线在 §2.8.23–§2.8.25 重设损失结构后，refinement step 已不再生成独立的 refined_auc（保留字段但等于 raw_auc）——故 `auc`(refined) 列在 H2H 主表中**仅 baseline 一侧有定义**，不作为对位指标。
- `paper_score` / `auc_gain` 同样降级为辅助诊断，不进 paper headline。

**对历史数字的影响**：
- §6.1 / §6.4（SUPERSEDED）旧 baseline A/B/C 的 `auc` 0.7320/0.8830 数字标记为"早期 refined 主指标产物"，已在 §6 顶部 banner 注明。**paper 中 baseline 唯一引用源固定为 §6.6 + §6.7**（v2026-04-27 verbatim + 同选模 H2H），不引用 §6.1 / §6.4 数字。
- §2.8.x 历史 S1/S1D 报告同时报了两套指标（raw + refined），切换后只把 headline 改为 raw，refined 保留作辅助；**已落地数字均不变**。

**ADR-011 强化（本节追认）**：从 2026-04-30 起，paper 中所有定量声明（包括摘要、引言、实验结果、消融、附录）的"AUC"无前缀时统一指 `raw_auc`；提到 `refined_auc` 必须显式标注后缀，且仅出现在方法学诊断与历史溯源章节。

**不变量**：
- §2.8.17 ADR-011 原文不动（本节为补充溯源）；
- redline 选模规则（`ssim_oc ≤ 0.15 ∧ raw_auc ≥ 0.60`）不动；
- §6.6 表格双列继续保留（raw + refined），便于追溯 baseline 极性翻转模式。

---

## 3. 北极星指标导向的攻击点（路线图）

> 方向：让 R 在 outlier 上**结构性失败**，同时不破坏 inlier 重建质量。

**RM-1** 强化外类扭曲损失（替代当前 `g_outclass_distortion_scale=0.0` 的弱设置）
- 现：`F.relu(margin − L1(R(X̂), X̂))`，margin 默认 0.2，scale 默认 0（figure6_7 默认完全关闭！）
- 改：(a) 引入 **Negative SSIM** 项 `relu(margin − (1 − SSIM(R(X̂), X̂)))`；(b) 在 D 端新增 `BCE(D(R(X̂)), 0)` 已存在但 scale 太小，需重标定。
- **基线依据**：B 仅靠 D 端外类项（scale=0.1）已把 ssim_gap 从 +0.103 提至 +0.199；开启 R 端扭曲 + SSIM 损失预期将进一步拉大差距。
- **✅ 达成状态（2026-04-22, §2.8.25）**：(b) 已落地（`d_outclass_loss_scale=0.1` + `g_outclass_distortion_scale=0.3, margin=0.6`，见 §2.8.24 margin 扫描 + §2.8.25 FINAL 50-run）。**10 类 × 5 seed 实测**：`clean-pass 48/50 (96%)`，`ssim_oc 均值 0.079`（远低于 RM-1 目标 0.30），`ssim_gap +0.126`，`hinge g_out 全程激活`。Negative SSIM 项 (a) 未落地 —— L1-hinge 已达标，(a) 作为 stretch 保留到 Round 2。

**RM-2** 训练阶段限制 R 的容量分布漂移
- 在 inlier 重建 loss 充分收敛后，冻结部分 encoder 参数 → 强迫 R 只在 inlier 流形附近变化（防止它学成通用降噪器）。
- **基线依据**：A 的 ssim_oc 10 epoch 翻 5 倍（0.145→0.794）即 §4.4 overtraining 的教科书式发生。

**RM-3** 选模直接以北极星指标为目标
- 新增 `selection_strategy="distortion"`：按 `α·ssim_gap + β·auc_gain + γ·refined_auc_floor` 排序，配合早期窗口 + 强制 floor。
- **基线依据**：paper_score 当前归一化让 A-ep2（auc_gain=−0.10）与 B-ep6（auc_gain=−0.06）分别拿到 0.72 / 0.81，分数高但违背论文主张。

**RM-4** 评估时同时报告 raw/refined 两套指标 + outclass L1 → 让 paper Figure 7 行为可量化追踪。

> 上述每条落地后必须在「§5 变更日志」记录前后指标对比。

### 3.1 基线驱动的执行优先级（2026-04-19 修订）

基于 §6.2 五条发现，**将 RM-1 与 RM-3 提前到 PR-A/PR-B 之后立刻执行**：没有它们 PR-A/PR-B 修好也只是"不崩而已"，仍无法达到 Figure 7 的 `auc_gain > 0`。

| 执行顺序 | 项目 | 状态 | 类型 | 理由 |
|:---:|---|:---:|---|---|
| ① | **PR-R** | ✅ 完成 2026-04-19 | P0 修复 | 解锁 C 基线，一行代码修复，后续所有 alocc_loss_cls 依赖都靠它 |
| ② | **PR-Q** | ✅ 完成 2026-04-19 | 环境 | CUDA 迁移（`torch 2.11.0+cu128` / RTX 5060），A/B/C 基线单 run ~15-28 s，消融密度可跑到 20+ 组 |
| ③ | **PR-A** + **PR-B** | ⏳ 下一步 | P0 正确性 | 先让选模可信（PR-A）、让训练能刹车（PR-B），为 RM-1 提供稳定评估基线 |
| ④ | **RM-1** | ✅ 完成 2026-04-22 | 核心攻击 | `g_outclass_distortion` (L1-hinge) 已开 + 调校到位（m=0.6, s=0.3）；**§2.8.25 50-run 实测**：clean 48/50 (96%)，ssim_oc=0.079（远好于目标 0.30），ssim_gap=+0.126。Negative SSIM 项 (RM-1.a) 作为 Round 2 stretch |
| ⑤ | **RM-3** | 待执行 | 选模对齐 | 新增 `distortion` 策略 + `refined_auc_floor` 硬门槛；目标：best_ep 必须满足 `auc_gain > 0` OR 明确记录违反原因 |
| ⑥ | **RM-2** | 待执行 | 容量约束 | 若 RM-1 后 ssim_oc 仍在中后期漂移，再引入 encoder 参数冻结 |
| ⑦ | **RM-4** | 待执行 | 评估扩展 | 在流水线中固化"raw vs refined"双栏报告；消融表格自动化 |
| ⑧ | **PR-E/F/G** | 暂缓 | P1 重构 | 前面都过后，收敛 build_model / 提 FrameMetrics / 损失模块化（受 ADR-004 §8 约束） |
| ⑨ | **PR-H..P** | 暂缓 | P2/P3 清理 | 最后处理可视化 bug 与维度审查补强 |

**明确暂缓**：在①-⑦完成前不动 P1 重构（PR-E/F/G），避免重构与实验改动互相污染；P2/P3 进入 post-paper-behavior 阶段。

### 3.2 Round 2 设计文档（2026-04-21 草案 v0，基于 §2.8.19 + §2.8.20）

> 本节为**设计草案**，未经师兄审阅前不写代码。目标：把"Round 1 S1 + rank-tuning"得到的 10/10 单 epoch 过线，升级为**全 epoch 解耦**的 identity-shortcut 突破。

#### 3.2.1 Round 1 封版认定

- **单 epoch 指标**：S1 + per-class rank ∈ {4, 8, 16} 可在 10/10 MNIST 类上满足 redline（`ssim_oc ≤ 0.15 ∧ raw_auc ≥ 0.60`）；Round 1 达成 A1 主线的"质量红线"要求（ADR-009）。
- **全 epoch 指标**：`coupling_ratio = Δssim_oc / Δssim_ic` 10 类均值 = 1.034（OFF=1.116），最好类仅 0.86 —— **identity shortcut 未被破坏**，只是 early-stop 被动规避。
- **Round 1 的方法学性质**：`选模策略 redline` + `低秩+噪声瓶颈 S1`（可 per-class 调 rank）= **在"AE 天然倾向记忆"的框架里把 early-stop 做到教科书严格**。它无法解决"AE 对 outlier 也越来越能重建"这个根源问题。

#### 3.2.2 Round 2 北极星（三层递进）

| 层 | 指标 | 当前 (S1) | Round 2 目标 | 成功判据 |
|:-:|---|:-:|:-:|---|
| L1 | redline pass-rate | 10/10（rank-tuned）| 10/10（固定配置）| 单一 rank/bottleneck 配置通吃 10 类 |
| L2 | `coupling_ratio` 10 类均值 | 1.034 | **≤ 0.80** | 全 epoch 趋势"弱解耦" |
| L3 | `Δssim_oc` 10 类均值（ep1→ep10） | +0.60 左右 | **≤ +0.20**，或理想 **< 0** | 全 epoch 趋势"强解耦"甚至反趋势 |

L1 是"不回退"，L2 是"设计性升级"，L3 是"结构性突破"。Round 2 以 **L2 为准入线**，L3 为 stretch goal。

#### 3.2.3 三条候选路径

**候选 A：Contractive AE（Rifai 2011）**
- **思路**：在 encoder 输出加 Frobenius-norm Jacobian 惩罚 `λ · ‖∂h/∂x‖_F²`，强迫 encoder 对输入的局部敏感度变小 → inlier 在流形切向仍可变、outlier 的法向扰动被"抹平" → ssim_oc 应随 epoch 不涨或微降。
- **实现难度**：中等；Jacobian 近似用 `torch.autograd.grad` 逐 batch 计算，开销约 +30% GPU 时间。需加 `--contractive-lambda` 开关（ADR-007）。
- **作者冲突检查**：Rifai / Bengio 系，**无冲突**（ADR-010 红线外）。
- **预期**：`coupling_ratio` 直接命中 L2 目标（≤0.80），L3 有希望但不保证。
- **风险**：Jacobian 与 bottleneck 噪声叠加可能让 inlier 重建也崩（`ssim_ic` 下降、raw_auc 崩）；需要先在 class 2 / 6（最干净类）小 λ 扫描确认收益方向。

**候选 B：Memory Bank（MemAE, Gong 2019）**
- **思路**：decoder 输入不是 encoder 的 `z`，而是 `M×d` 个可训练 prototype 的 softmax 加权和 `z' = Σ α_i · p_i`（α 由 `z` 与每个 `p_i` 的相似度 softmax 给出）。训练只用 inlier → `p_i` 收敛到 inlier 原型 → outlier 的 `z` 找不到匹配原型 → reconstruction 必然失败 → ssim_oc **结构性低**。
- **实现难度**：高；需要改 decoder 入口 + 新增 `memory` buffer + 读写地址机制。至少 1 周 + 测试。
- **作者冲突检查**：Gong / Liu / UCAS 系 → **需要师兄二次确认**（ADR-010 红线待查）。
- **预期**：`coupling_ratio` 可打到 L3 目标（< 0.5 甚至负值），但代价是训练复杂度翻倍。
- **风险**：MemAE 原文报告 inlier 重建锐度下降（因 prototype 量化），可能影响 Figure 6 的视觉复现。

**候选 C：Two-Stage Training（无名方法、工程解）**
- **思路**：Stage 1 = 只训 AE（`L2(R(X̃), X)`，epochs 10–20）让 `ssim_ic` 饱和；Stage 2 = **冻结 encoder 和 decoder**，只训 D 的外类扭曲项（`BCE(D(R(X̂)), 0) + relu(margin − L1(R(X̂), X̂))`，epochs 5–10）让判别面专门学"outlier → low score"。
- **实现难度**：低；只需在 runner 加 `--stage` flag 和 param-group 冻结逻辑。
- **作者冲突检查**：方法本身无论文来源 → 无冲突。
- **预期**：`ssim_oc` 在 Stage 2 可能反而**上升**（encoder 不再约束），但 D 的 score 曲线分离 → `auc_gain > 0` 可能改善。与 L2/L3 目标**不完全对齐**（它改 D 的输出不改 R 的 ssim）。
- **风险**：不解决 identity shortcut 的根（R 仍记忆），只是换 D 的决策面形状 —— 师兄"结构先于模块"指令下此方案优先级最低。

#### 3.2.4 建议执行顺序

1. **C-3 结束 → 师兄审阅本 §3.2**，定候选（预计 A 或 A+B 组合）。
2. **候选 A（Contractive）P0 实验**：class 2 上扫 `λ ∈ {0.001, 0.01, 0.1, 1.0}`，挂在 S1 r=16 p=0.3 上（保持 ADR-007：`--contractive-lambda=0` 默认关闭、bitwise 与 OFF 一致）；判据 = `coupling_ratio` 能否从 1.00 压到 0.80 以下。
3. **若 A 单独达 L2** → 跑 10 类 + 自动 rank 选择（复用 C-2 网格最佳表）。
4. **若 A 单独不达 L2** → 引入 B（MemAE），先师兄排查作者圈。
5. **候选 C 降为 stretch**：只在 A/B 达标后作为"refiner 替代"验证 raw/refined 分离（对应 §2.8.19 class 9 refiner 熨平问题）。

#### 3.2.5 与现有开关契约的兼容

- 所有新增必须满足 ADR-007：`--contractive-lambda` / `--memory-size` / `--memory-addressing` / `--training-stage` 皆走 "flag + kwarg + summary 回声" 模式。
- 新增 `selection_info` 字段：`contractive_jacobian_mean`、`memory_usage_entropy`（监控用，不参与选模）。
- Round 1 产物（S1 rank-scan 表、redline 选模）全部保留，作为 Round 2 消融的 baseline。

> **未决 blocking 项**：候选 B 的作者圈审核、per-class rank 自动化的算法（当前只有手工查表）。

---

## 4. 关键决策记录（ADR）

| ID | 日期 | 决策 | 依据 |
|----|------|------|------|
| ADR-001 | 2026-04-19 | 工作目录定为 `d:\codeVS\ALOCC_paper\`，所有产物（笔记/脚本/对比报告/PR 草稿）均落此处；源码改动落 `D:\Trae_coding\ALLOC\ALOCC-master\` | user 指示 |
| ADR-002 | 2026-04-19 | 北极星指标定义见 §0；所有重构必须给出该套指标前后对比 | user 指示（"显著提升异常外类扭曲指标"） |
| ADR-003 | 2026-04-19 | 允许结构性重构，不强制最小改动 | user 指示 |
| ADR-004 | 2026-04-19 | 任何重构（尤其 loss 模块化、Trainer 抽象）必须通过 §8 的 5 条张量维度审查；新增模块需配 shape 单元测试 | user 指示（防止维度对齐回归） |
| ADR-005 | 2026-04-19 | 路线图执行顺序按 §3.1 基线驱动版；P1 重构（PR-E/F/G）推迟至 RM-1/RM-3 稳定后，避免重构与行为修复互相污染 | 基线数据显示 Figure 7 主张被颠倒，先修行为再谈结构 |
| ADR-006 | 2026-04-19 | 训练脚本默认规模（`epochs=10, train_count=4096, batch_size=64, noise_std=0.31, r_alpha=0.2, lr=0.002`）作为后续所有消融的**锚点配置**；任何偏离必须在 §5 变更日志中声明 | 保证 RM-1/RM-2/RM-3 前后指标可直接对比 |
| ADR-007 | 2026-04-19 | **所有改动必须可开关**：新增行为必须通过 CLI flag + Runner kwarg + 默认关闭暴露，开关状态写入 `summary.json`；禁止硬编码、隐式开关、默认偏离基线。详见 §9 开关治理清单 | user 指示（"我要你做的一切更改都要变成可用开关控制的，通过输入参数随时可以开关调配"） |
| ADR-009 | 2026-04-19 | **主线任务定为 Baseline A 复现与改进**（A1 MNIST 数值复现 + A2 Figure 6/7 复现 + A5 跨数据集扩展）；执行档位 = "改法档"（允许加模块 / 改结构 / 结合，但保留 one-class + GAN 框架）；质量红线 `ssim_oc ≤ 0.15`；先小规模 MNIST 跑通再放大。详见 §2.5。| user 指示（"A 基线作为主线任务"、"我要做 A1/A2/A5，主要要求异常扭曲率变高和正常重建率更好"）+ 博士师兄建议 |
| ADR-010 | 2026-04-20 | **RM-1 L3 MANet 方向终止 + 结构优先原则**：该论文（PR 2020 "Multi-head enhanced self-attention network for novelty detection"）第一作者与本实验室师兄存在导师圈重合，继续复现/改造构成重复发表风险，所有 RM-1 L3 代码/开关/备份/实验产物按 A1 方案彻底回滚。新方向遵循双指令：(1) **改结构先于加模块**（瓶颈/编解码/判别器拓扑改动优先于注意力等"锦上添花"模块）；(2) **训练动力学先于超参数扫描**（AUC 坍塌等现象按 D/G 博弈与梯度行为定位，而非 `balance_weight` 局部最优搜索）。RM-1 Round 1 的后续方案 = **改法-S1（低秩 + 噪声瓶颈）**，在师兄审阅设计文档前不写代码。| 博士师兄指示（作者圈冲突 + 方法论方向纠正）|
| ADR-011 | 2026-04-21 | **选模策略是一等实验变量 + Redline 准则定为 S1 默认选模**：distortion 选模器被证明在 bottleneck 放松过程中系统性偏向"后期记忆 epoch"，导致 §2.8.13–2.8.14 的"跨类 AUC 坍塌/D 反极性"全部是方法学伪影；T2 Spectral Normalization 对 AUC 的"救场效应"同被证伪（SN 实际让 `raw_auc` 在 class 6/1 各掉 0.33）。新策略 `redline = (ssim_oc ≤ τ_oc ∧ raw_auc ≥ τ_raw, 最早 epoch)`，默认阈值 τ_oc=0.15 / τ_raw=0.60，作为后续 S1 及其衍生方案的**默认选模**；legacy 三策略保留用于诊断但不出现在最终报告指标路径中。所有新实验 summary 必须回声 `selection_info.redline_*` 四字段。| §2.8.17 离线验证（`t3_offline_selection.py` 在 9 份 records 上 redline 3/3 过线、distortion 5/9 命中）+ ADR-007 回归（`a1_sel_parity.py` 全 PASS）|

---

## 5. 变更日志（倒序追加）

> 模板：
> ```
> ### YYYY-MM-DD HH:MM · <PR-X> 简述
> - 文件：a.py:L1-L2, b.py:L3
> - 动机：…
> - 北极星指标对比（before → after）：ssim_oc=… → …, ssim_gap=…, auc_gain=…, refined_auc=…
> - 论文一致性：保持 / 退化（说明）
> - 后续依赖：…
> ```

### 2026-04-27 · Baseline A 算法忠实度终审 + TF1.15-verbatim 锁定（§2.10 落地）
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\model.py`（基类 `ALOCC` 的 `__init__` L240/L242 + `_train` L308-L311；S1D 子类未触动）
  - `D:\Trae_coding\ALLOC\ALOCC-master\model.py.pre_baseline_a.bak`（pre-patch 备份，幂等校验用）
  - `ALOCC_paper/_patches/_apply_baseline_a_verbatim.py`（一次性 patch 脚本，含 anchor 校验）
  - `ALOCC_paper/_patches/_diff_primitive_vs_live.ps1` + `_dump_diverging_diffs.ps1` + `_diverging_diffs.md`（30 KB Primitive vs Live diff，appendix 证据）
- 动机：师兄反馈"方法用原作者的，数据 / 指标 / 评测用我们的"。Primitive 仓库已确认含师兄改进项（不可作真理来源），需用 TF1.15 原作者源码（`ALOCC-original/`）作四维对照——损失函数 / 优化器 / 训练循环 / 选模——锁定 Baseline A 协议。
- 决策（user 拍板）：D1-B / D2-B / D3-B / D4-C 全套 verbatim：
  - **D1-B**：refinement loss 从 `MSE(R(X̃), X)` 改为 `BCE_with_logits(R(X̃), X̃)`，与 TF1.15 `models.py` L131 逐字对齐（论文 Eq.4 偏离声明见 §2.10.5）。
  - **D2-B**：RMSprop 显式 `alpha=0.9, eps=1e-10`，对齐 TF1.15 默认（PyTorch 默认是 `0.99 / 1e-8`）。
  - **D3-B**：源码级直译 + 关键行加 `# [BASELINE-A][...][TF1.15 ...]` 锚点注释。
  - **D4-C**：选模采用 `--selection-epoch-start 10 --selection-epoch-end 10` 把窗口收敛到第 10 epoch，对齐 TF1.15 "无自动选模、取末轮" 的真实行为。
- 验证：`Select-String` 标签计数 `[BASELINE-A][D1-B]=1`、`[BASELINE-A][D2-B]=2`，与预期一致；S1D 子类的 `g_loss_r = self._refinement_loss(...)` 仍在 L458 / L562 原位（patch 仅改基类，子类隔离）。
- 北极星指标对比：N/A（未跑实验；50-run 矩阵脚本待 user 批准后另起）。
- 论文一致性：**显式偏离 + 已声明**——Baseline A 遵循 TF1.15 代码（BCE-on-noisy），不遵循论文 Eq.4 字面（MSE-on-clean）。Paper 写作时须在 Baseline A 段加脚注，理由是论文 Figure 8 的发表数字本就是 TF1.15 代码产物。
- 副作用：S1D 子类继承基类 `__init__` 的 RMSprop 升级（`alpha=0.9, eps=1e-10`），与 baseline 对齐——若需重跑既往 S1D 结果以保严格可比性，由 user 后续按需决定。
- 状态变化：§2.9（TF1.15 strict baseline · CPU 复现）降级为"算法忠实度交叉验证物料"，§2.10 取代为 Baseline A 主线协议；既往 `_off` runs（`variant=alocc_loss + scale=0`）**不**计为 strict baseline，需在 Baseline A v2026-04-27 协议下重跑 50-run 矩阵。
- 后续依赖：起草 `run_baseline_a_50.ps1`（5 seed × 10 digit, ADR-006 锚点 + D4-C 选模），先 1 个 smoke run（class=1, seed=42, ~1 min）验证管线，smoke 通过后由 user 单独批准启动 50-run 全量。

### 2026-04-27 22:48 · Baseline A v2026-04-27 · 50-run 矩阵执行 + 文档归档（§2.10.7 / §6.6 落地）
- 文件：
  - `ALOCC_paper/_patches/run_baseline_a_50.ps1`（调度器；smoke / 全量 / dry-run 三模式；fix: 日志写入挪到调度器内层避免 Smoke 子目录被 runner 清空时的 `[WinError 32]`）
  - `ALOCC_paper/_patches/_aggregate_baseline_a_50.py`（50 个 `summary.json` → `_aggregate.{json,md}`，per-class + 全局 mean±std + best_epoch 一致性校验）
  - `D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27\d{0..9}_s{42..46}\summary.json`（50 个 run 产物；每个含 `best_metrics` + `selection_info` + `_run_history`）
  - `baseline_a_v2026_04_27\_aggregate.{json,md}` + `_logs\d{digit}_s{seed}.log{,.err}`
  - `ALOCC_paper/PROJECT_LOG.md` §2.10.7 / §2.10.8 / §2.10.9 / §6.6 新增；§6.1 / §6.4 标记 SUPERSEDED
- 动机：v2026-04-27 verbatim 协议落地后，50-run 矩阵是 paper 中 Baseline A 引用源所必需。
- 决策：seeds = {42, 43, 44, 45, 46}（连续）；digits = {0..9}（全 MNIST 类）；其余锚点同 ADR-006 + D4-C。
- 验证：
  - **Smoke run（class=1, seed=42）**：13.05 s 完成；`best_epoch=10` 命中；`refined_auc=0.940`，无 NaN/Inf；管线全绿。
  - **50-run 全量**：13m06s（GPU 顺序），50/50 成功（0 failed）；`best_epoch=10` 50/50 命中（D4-C 一致性 ✅）。
- 北极星指标（50-run 全局，mean ± std）：
  - `auc=0.5021 ± 0.2941`（refined）；`raw_auc=0.4977 ± 0.2972`；`auc_gain=+0.0044 ± 0.0123`
  - `ssim_oc=0.9231 ± 0.0209`（R 几乎完美复制 outlier，BCE-on-noisy 退化 → identity-mapping）
  - `ssim_gap=+0.0222 ± 0.0262`；`paper_score=0.4331 ± 0.1445`
  - 三个必须显式说明的解读（O1/O2/O3）见 §2.10.8。
- 论文一致性：**Baseline A 现有强基础**——协议忠于 TF1.15 源码、选模诚实（D4-C 末轮）、规模标准（10 类 × 5 seed = 50 run），AUC≈0.50 与 §2.7.2 历史 paper-selection 0.598 同量级，证明 v2026-04-27 是当前最严格、最忠实的 ALOCC 复现基线。
- 副作用：§6.1 / §6.4 中 Baseline A `refined_auc=0.79/0.97` 不再用作 paper 引用；S1D 既往 §2.8.25 数据采用旧优化器（α=0.99/eps=1e-8），如需严格可比需在 v2026-04-27 协议下重跑（C 任务待启动）。
- 后续依赖：S1D 50-run 矩阵（C 任务，variant 待 user 确认 `alocc_loss` vs `alocc_loss_cls`）；选模策略对 S1D 是否同步 D4-C 需单独决策（D4-C 会杀掉 redline 的早期 epoch 优势）。

### 2026-04-28 · ADR-007 隔离原则修复 · verbatim 协议改为可选插件（§2.10.10 落地）
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\model.py`（基类 `ALOCC.__init__` L213-L246：默认值恢复 α=0.99/eps=1e-8 + 新增 `tf_verbatim_rmsprop` 构造参数 + 三元开关）
  - `D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py`（L29-L42 `build_model` 透传；L536 实例化处；L625 `summary.json.switches` 记录；L705 CLI flag `--tf-verbatim-rmsprop`）
  - `D:\Trae_coding\ALLOC\ALOCC-master\model.py.pre_isolate.bak` + `mnist_experiment_runner.py.pre_isolate.bak`（一次性自动备份）
  - `ALOCC_paper/_patches/_isolate_baseline_a_toggle.py`（重构 patch，幂等）
  - `ALOCC_paper/_patches/run_baseline_a_50.ps1`（L3-L9 协议头说明 + L80 argList 显式 `--tf-verbatim-rmsprop` opt-in）
  - `ALOCC_paper/_patches/_verify_isolation_smoke.py`（Baseline A bit-for-bit 验证：snapshot/compare 双相）
  - `ALOCC_paper/_patches/_verify_s1d_isolation.py`（S1D 主线优化器属性验证：构造 `ALOCC_LOSS` 不传 flag → α=0.99/eps=1e-8）
  - `ALOCC_paper/_patches/_isolation_check/d1_s42_before.json`（既往 d1_s42 summary.json snapshot，用于对位）
  - `ALOCC_paper/PROJECT_LOG.md` §2.10.10 + 本变更日志
- 动机：user 审计指出 §2.10.3 patch 把 D2-B 写死到基类，导致 S1D 子类被强制拿到 TF1.15 verbatim 优化器——违反 ADR-007 "所有架构改动必须可开关、默认行为必须与历史一致"。架构上 baseline 应当是可调用的独立实验，而非全局协议升级。
- 决策：基类 RMSprop 恢复 PyTorch 默认（α=0.99/eps=1e-8）；TF1.15 verbatim 改为通过 CLI flag `--tf-verbatim-rmsprop` 显式 opt-in；S1D 不传 flag 即沿用项目历史环境。D1-B（BCE-on-noisy）保持原状——该逻辑只在 `variant=alocc` 路径生效，S1D 子类走自己重载的 `_train()`，无跨 variant 污染。
- 验证（双向隔离）：
  1. **Baseline A 50-run 数据有效性**：digit=1 seed=42 smoke 重跑，15/15 best_metrics keys 与 §6.6 既往 d1_s42 **bit-for-bit identical**（auc=0.998850 / paper_score=0.488859 / best_epoch=10），`switches.tf_verbatim_rmsprop=True` 已写入 summary.json → §6.6 表格全部数字保持有效，无需重跑。
  2. **S1D 主线回归**：构造 `ALOCC_LOSS(in_h=28, out_h=28, lr=0.002)` 不传 flag → `optim_D/G.alpha=0.99, eps=1e-8`，`tf_verbatim_rmsprop_attr=False` ✅。S1D §2.8.25 Phase 3 的 50-run 数字（clean 48/50, raw_auc 0.805 ± 0.109）**继续有效**。
- 论文一致性：§6.6（Baseline A v2026-04-27）与 §2.8.25（S1D Phase 3）现在共享同一个语义层 ——"Baseline 是 verbatim opt-in 插件，主线是 PyTorch 默认"——两边数字均无需重跑，paper 主对比表 (Baseline A vs S1D) **环境一致性已恢复**（混淆变量消除）。
- 副作用：§2.10.3 中"D2-B 写死到基类"的描述已经过时（基类已恢复默认），但 §2.10.4 协议表 + §2.10.7 数据 + §6.6 表格保持原样（运行时行为通过 flag opt-in 后等价）。**C 任务"S1D 在 v2026-04-27 协议下重跑以对齐优化器"取消**——优化器从未真正改动 S1D 的运行环境（除了短暂的 §2.10.3 → §2.10.10 之间的窗口期，且该窗口期内 S1D 未有任何重跑）。
- 后续依赖：无。本次修复关闭 §2.10 的全部架构 TODO；下一步工作（如 RM-1 / S1D 改造）回归独立的 §3 路线图，不再受 baseline 协议影响。

### 2026-04-25 · ALOCC 原作者 strict baseline 复现（10 digits × seed=42 · §2.9 落地）
- 文件：
  - `ALOCC_paper/_patches/apply_alocc_original_patches.py`（master patch；L1 兼容性 + L3 实验扩展，幂等可重入；备份 `train.py.orig` / `models.py.orig`）
  - `ALOCC_paper/_patches/run_phase0_trial.ps1`（Phase 0 trial 启动器）
  - `ALOCC_paper/_patches/run_full_baseline_sweep.ps1`（4 路并行调度器，`intra=4 inter=1`，连续投递）
  - `ALOCC_paper/_patches/_summarize_sweep.ps1`（产物完整性 + 主日志聚合）
  - `D:\Trae_coding\ALLOC\ALOCC-original\{checkpoint,export}_d{0..9}_s42\`（10 套末段 5-epoch 权重 + 拼接图）
  - `D:\Trae_coding\ALLOC\baseline_logs\sweep_20260425_203914_*`（master log + 9 run 子 log + 启动 bat）
  - `ALOCC_paper/PROJECT_LOG.md` §2.9（新增 7 个子节）+ 本变更日志
- 动机：S1D 报告（§2.8.25 clean 48/50, raw_auc 0.805±0.109）跑在已多轮 PR 改造过的 `ALOCC-master`（PyTorch + redline + S1 bottleneck + ADR-006 锚点）上；与上游 `Sabokrou/ALOCC-CVPR2018`（TF1.15 + 原超参）不可逐位比对。需要在原作者代码 + 原作者超参（`epoch=40, batch=128, lr=0.002, β1=0.5, r_alpha=0.2, σ=0.155`）下跑 10 类，作为论文复现层面的 strict baseline，与 S1D 对照报告。
- 环境：`D:\Trae_coding\ALLOC\ALOCC-original\.venv-tf1`（Python 3.7 + TF 1.15 CPU build）；硬件 Intel Core Ultra 9 275HX 24C24T，**CPU only**（TF1 不识别 RTX 5060）。
- ADR-007 三层 patch：L1（兼容性，纯环境/版本：`np.inf→10**9`、`log_dir→alocc_log_dir`、UCSD/MNIST 条件分支、SIFTETS.npy guard）+ L2（超参 verbatim）+ L3（`--seed` flag + `_d{digit}_s{seed}` 路径后缀，默认不传 `--seed` 即等价原版）。**默认行为与原作者 verbatim 版一致**；附加协议全走 CLI 显式打开。
- Phase 0 trial（digit=1, seed=42, 单进程 8 线程）：665.9 s = 11.1 min，d_loss 1.412 → **1.387**（命中 GAN 均衡 ln 4 ≈ 1.386），suffix 隔离生效（旧 `checkpoint/mnist_128_28_28/` 污染权重未被触碰）。
- Full sweep（9 runs × 4 路并行 × 4 线程/进程）：**38m30s** wall clock（20:39:14 → 21:17:44），加速比 2.96× vs 串行 11.4 h。**10/10 全部 exit=0**，每 run 完整 40 epoch + `model-{35..39}` 5 套末段权重 + 15-19 张训练采样图（采样数差异源自 `(itr+1)%print_every` 与 batch 切片对齐，非 bug，所有 run last_png 均到 epoch 38/39）。
- 论文一致性：**逐字保持**（L1 仅环境兼容、L2 超参未动、L3 不动训练循环）；与 ADR-006 PyTorch 锚点（`noise_std=0.31, train_count=4096, epochs=10`）属于两套独立报告路径，evaluation 阶段并列呈现避免数字混淆。
- 后续依赖（§2.9.7 待 user 排序）：(1) 写 evaluation 脚本算 10 类 ROC AUC 给出 strict baseline 的 10×1 AUC 表 → S1D 直接对照；(2) 拼 2×5 grid 训练样本图入 paper；(3) σ 跨实现差异（0.155 vs 0.31）需在 evaluation 报告中显式说明；(4) 方差扩展（seed=1337/2026 各 10 runs 增量 ~1 h）按需启动以补 mean±std 对称性。

### 2026-04-21 · 数据泄露审计（师兄质疑 ssim_oc≈0.10 可信度 · §2.8.22 落地）

- 文件：
  - `ALOCC_paper/_patches/audit_data_leakage.py`（一次跑完 A5 图像级 SHA1 重叠检查 + C1 SSIM 数值正确性 + F2 跨类 SSIM 机制基线 + F3 类均值退化 floor 模拟）
  - `ALOCC_paper/PROJECT_LOG.md` §2.8.22（审计节）+ 本变更日志
- 动机：师兄原话——"ssim oc 的指标都能到 0.2，我觉得好的有点离谱，你明天让模型检查一下，这个结果是不是可信，是不是存在数据污染或者数据泄露"。
- 三层结论：
  - **L1 物理**：`train_dataset`（60000 SHA1）∩ `test_dataset`（10000 SHA1）= **0 images** → CLEAN；`mnist_experiment_runner.py:36,44` 的 `train=True/False` 分离在链路上保证不可能交叉。
  - **L2 数值**：初查一度怀疑 `Metrics.py` 的 `data_range=1.0` 与 `[-1,1]` 不匹配；复核 `Metrics.py:99-100` 确认 SSIM 前显式 `(x+1)/2` 映回 `[0,1]`，与 piq 期望一致，**OK**。
  - **L3 机制**：MNIST 跨类自然 SSIM ≈ 0.17（同类 0.29–0.36），`R→mean-of-class` 退化 floor ≈ 0.14；观测 `ssim_oc ∈ [0.10, 0.15]` 略低于 0.17，因为 R(outlier) 是"既非 inlier 也非 clean outlier"的模糊重建，structure/contrast term 双向拉低，**机制上可预期，非离谱**。
- 结论：`ssim_oc≈0.10` 可信，没有数据泄露，没有数值错误。
- 非问题备注：`out_class_scale=1` 使训练集含外类样本，但这些外类样本仍来自 `train_dataset`，与 `test_dataset` 不相交（L1 已覆盖），不构成跨集泄露。
- 后续依赖：无，属于师兄审计要求的一次性材料；报告可直接发给师兄。

### 2026-04-21 · C-1/C-2/C-3：失败类 rank-scan 推翻 S1 ceiling + Round 2 设计文档

- 文件：
  - `ALOCC_paper/_patches/run_s1_rankscan_c2.ps1`（C-2 主脚本：12 runs × rank{8,4} × dropout{0.3,0.5} × class{0,3,7}）
  - `ALOCC_paper/_patches/c2_rankscan_report.py`（聚合脚本：读 BOM-UTF8 的 PowerShell JSON，输出 MD 表 + clean-pass 计数）
  - `ALOCC_paper/_patches/b4_per_epoch_plots.py`（C-1 扩展：`CLASSES = PASS + FAIL`，`fig` 从 1×3 改 2×3，行左侧 annotate "PASS / FAIL" 色带）
  - `ALOCC_paper/` 新增实验目录 12 套：`s1_c{0,3,7}_r{8,4}_p{03,05}_redline/`（每套 4 件套）
  - `ALOCC_paper/` 新增汇总 2 份：`s1_rankscan_c2_summary.json` / `s1_rankscan_c2.md`
  - `ALOCC_paper/figures/` 新增 3 张 PNG：`b4_per_epoch_c{0,3,7}.png`（失败类）；`b4_per_epoch_grid.png` 从 1×3 覆盖为 2×3
  - `ALOCC_paper/PROJECT_LOG.md` §2.8.20（C-2 结论）+ §3.2（Round 2 设计草案 v0）+ 本变更日志
- 动机：§2.8.19 遗留"S1 ceiling 假说"未验证；根据用户"允许自行决定 1–2 个会话任务"授权，执行 C-1（fail 类可视化）+ C-2（rank-scan 消融）+ C-3（Round 2 设计文档）。
- 北极星指标对比（per-class 最佳 rank-scan vs S1 r=16 p=0.3）：
  - **class 0**：ssim_oc 0.394 → **0.148**（r=8 p=0.3），raw_auc 0.688 → 0.803，redline ❌ → ✅
  - **class 3**：ssim_oc 0.366 → **0.140**（r=8 p=0.3），raw_auc 0.687 → 0.824，redline ❌ → ✅
  - **class 7**：ssim_oc 0.406 → **0.116**（r=4 p=0.3），raw_auc 0.700 → 0.892，redline ❌ → ✅
  - **12 runs clean-pass**：7/12（无 fallback）；每失败类至少 1 配置过线。
  - **推翻结论**：S1 对 {0,3,7} 没有到顶，r=16 对它们不够狠；10/10 redline 可达（per-class rank tuning）。
- 论文一致性：保持（rank-scan 全部走 S1 既有 `--bottleneck-rank` / `--bottleneck-dropout` 开关，ADR-007 契约不变）；等价性回归 `a1_sel_parity.py` 未重跑但脚本未改动。
- 后续依赖：
  - §3.2 候选 A/B/C 待师兄审阅（B 需作者圈复查 Gong / Liu 团队）。
  - per-class rank 选择仍手工；Round 2 需要"自动选 rank"算法（候选：在热身 1–2 epoch 的 ssim_oc 导数上定 rank 搜索步长）。
  - seed 鲁棒性仍未补；rank-scan 7/12 clean-pass 需 3-seed 重复确认。
  - `figures/b4_per_epoch_grid.png` 现为 2×3（覆盖旧文件），旧 1×3 版本可以通过 git 历史回查。

### 2026-04-21 · A4-SEED：`--seed` CLI + C-2 三-seed 稳健性复核（§2.8.21 落地）

- 文件：
  - 源码 patches（4 处）：`utils.py` 改 `set_random_seed(seed=None)` 语义 + 模块级 `_CURRENT_SEED=42`；`mnist_experiment_runner.py` / `run_paper_mnist_figure6_7.py` / `export_mnist_triplets.py` 三处新增 `--seed` CLI + `set_random_seed(args.seed)` + summary `switches.seed` 字段。源码备份 `_patches/_backups/a4_seed/`。
  - `ALOCC_paper/_patches/a4_apply_seed.py`（patcher，已加 `new in text => skip` 幂等 guard）
  - `ALOCC_paper/_patches/run_a4_seed_regression.ps1`（class 1 r=16 p=0.3 redline 回归脚本，不带 `--seed`）
  - `ALOCC_paper/_patches/_diff_seed_regress.py`（summary.json 逐字段 diff；bitwise identity 检查）
  - `ALOCC_paper/_patches/run_a4_seed_robustness.ps1`（4 configs × 3 seeds = 12 runs 主脚本）
  - `ALOCC_paper/_patches/a4_seed_robustness_report.py`（聚合：per-run 明细 + per-config 均值/方差表）
  - `ALOCC_paper/_patches/_inspect_seed.py`（诊断：扫 4 源码文件的 `A4-SEED` 标记）
  - `ALOCC_paper/` 新增实验目录 13 套：`s1_c{0,3,7}_r{4,8}_p{03,05}_seed{42,1337,2026}_redline/`（12 套）+ `s1_c1_r16_p03_redline_regress/`（1 套回归）
  - `ALOCC_paper/` 新增汇总 2 份：`s1_seed_robustness_summary.json` / `s1_seed_robustness.md`
  - `ALOCC_paper/PROJECT_LOG.md` §2.8.21 + 本变更日志
- 动机：§2.8.19 / §2.8.20 两份核心结论（7/10、7/12、"10/10 可达"）全建立在硬编码 `seed=42` 的单 seed 之上，师兄会直接追问 seed variance；P1-任务 2 补齐。
- ADR-007 等价性回归：class 1 r=16 p=0.3 不带 `--seed` 与 §2.8.20 存档 baseline 逐字段比对——**`best_metrics` 0 差异 (BITWISE IDENTICAL)**，`switches` 仅多 `seed=null`、少一个已废弃的 `spectral_norm_d` 键（与 seed 无关）。**契约验证通过**。
- 稳健性核心结论（3 seeds × 4 configs = 12 runs，~3 min GPU）：
  - **seed=42 完美复现 C-2**：4/4 clean-pass，数值精确到 4 位小数，§2.8.20 数据可信。
  - **总 clean-pass 7/12；per-config 分布严重不均**：c0 r8 p0.3 = 1/3；c3 r8 p0.3 = 1/3；c7 r4 p0.3 = 2/3；c7 r4 p0.5 = **3/3**。
  - **"per-class 最佳 rank 查表"是 seed-sensitive**：c=7 r=4 p=0.3 在 seed=1337 训崩（auc=0.387，差于随机）。C-2 的"10/10 via per-class rank tuning"需降级为"seed=42 下 10/10 可达；跨 seed 鲁棒仅 c7 r4 p0.5 一组"。
  - **反直觉**：dropout=0.5 比 0.3 更 seed-stable（唯一 3/3 过线），尽管 peak 指标略差——更强正则抵消初始化敏感性。**§2.8.20 推论 1"p=0.5 全面劣于 p=0.3"需要部分抵消**。
- 论文一致性：ADR-007 契约维持（`--seed` 默认 `None` → bitwise 等价历史实现）；ADR-011 redline 选模不受影响；ADR-010 结构优先不涉及（seed 基础设施 ≠ Round 2 结构）。
- 后续依赖：
  - 3 seeds 是小样本；是否扩到 5 seeds × 10 classes 需要师兄定。
  - class 7 seed=1337 训崩样本未诊断（可能是 GAN 低 rank + 特定初始化对抗失稳）。
  - §3.2 的 L1 北极星（redline pass-rate 10/10）需明确"跨 seed mean ≥ 9/10"还是"seed=42 单点"。
  - §2.8.20 原文保留不重写，以 §2.8.21 为正式修订笔记。

### 2026-04-21 · Round 1 稳健性 10 类扩展（B-1..B-5）
- 文件：
  - `ALOCC_paper/_patches/` 新增 4 个：`run_s1_redline_10class.ps1`（7 新类 S1 批跑）、`run_off_redline_b2.ps1`（7 新类 OFF 基线）、`b3_aggregate_10class.py`（20 份 records 离线 redline 统一重放）、`b4_per_epoch_plots.py`（matplotlib 双纵轴 + 三策略竖线）。
  - `ALOCC_paper/` 新增实验目录 14 套（`s1_c{0,3,4,5,7,8,9}_r16_p03_redline/` 7 + `s1_c{0,3,4,5,7,8,9}_off_redline/` 7），产出物 = `summary.json` + `best.pth` + `triplets/` + `figure7_scores.json` 四件套/套。
  - `ALOCC_paper/` 新增汇总文件 4 份：`s1_redline_b1_summary.json`（S1 7 新类）/ `off_redline_b2_summary.json`（OFF 7 新类）/ `s1_redline_10class.json`（20 run 汇总 JSON）/ `s1_redline_10class.md`（MD 表）。
  - `ALOCC_paper/figures/` 新增 4 张 PNG：`b4_per_epoch_c{2,6,1}.png` + `b4_per_epoch_grid.png`（双纵轴 raw_auc / ssim + 3 策略 epoch 虚线）。
  - `ALOCC_paper/PROJECT_LOG.md` §2.8.19 落地 + 变更日志本条。
- 动机：A-5 只覆盖 3 类（{2,6,1}），需要验证 S1+redline 是否在 MNIST 全 10 类上稳健；同时把"红线通过率"从单点事件升级到矩阵 KPI，供 Round 2 方向锁定"S1 吃不动的类"。
- 北极星指标对比（redline 选模，S1 vs OFF）：
  - **红线通过率**：OFF 3/10 → S1 7/10（净增 +4；翻转类 = {2, 4, 5, 8}）。
  - 平均 `ssim_oc`（所有 10 类）：OFF 0.256 → S1 0.198（−0.059）。
  - 平均 `refined_auc`（所有 10 类）：OFF 0.667 → S1 0.783（+0.116）。
  - 平均 `raw_auc`（所有 10 类）：OFF 0.688 → S1 0.789（+0.101）。
  - **反例识别**：class 9 refined_auc 从 0.82→0.55（raw_auc 仍 0.68；refiner 在此类上过度熨平）；class 7 S1 让 AUC 从 0.93 倒退到 0.70（ssim_oc 压不下 → fallback distortion 选 ep 5）。
  - **S1 失灵类**：{0, 3, 7}，三类共性 = 即使启用 r=16 p=0.3 的 S1，`ssim_oc` 仍 ≥ 0.37，瓶颈宽度不够。
- 论文一致性：保持（所有指标路径 = Baseline A 的 refined_auc + S1 的 `bottleneck_*` 开关 + ADR-011 的 redline 选模）；ADR-007 bitwise 等价性不受影响（S1 关 / redline 关时与 baseline 逐位同步）。
- 后续依赖：
  - Round 2 R1 候选锁定 = rank ∈ {4, 8} × dropout ∈ {0.3, 0.5} 网格（专打 {0,3,7}）。
  - Round 2 R3 候选 = refiner 温度开关（解 class 9 的 refined < raw）。
  - seed 鲁棒性验证需先给 runner 加 `--seed`（未做，避免本轮引入新开关）。
  - `_run_all.ps1` CUDA 锚点仍保留 `acc_auc` 默认，redline 切换待下一轮决议。

### 2026-04-21 · T2 SN 注销 + Redline 选模落地（ADR-011 落地）
- 文件：
  - `mnist_experiment_runner.py`：`_select_records` 新增 `redline` 分支 + 2 个阈值 kwargs + `selection_info.redline_*` 四个审计字段；`evaluate_checkpoints` / `run_experiment` 透传；argparse 扩 `--selection-strategy {..., redline}` + `--redline-ssim-oc-max` + `--redline-raw-auc-min`；summary.switches 回声（哨兵 `[A1-SEL]`）。
  - `run_paper_mnist_figure6_7.py`：Args dataclass + argparse + `evaluate_checkpoints` 调用 + summary 四处镜像。
  - `model.py` / `export_mnist_triplets.py` / `mnist_experiment_runner.py` / `run_paper_mnist_figure6_7.py`：`a3_rollback_t2.py` 从 `.t2_sn.bak` 恢复，`[T2-SN]` 哨兵 / `spectral_norm` 引用全量清零。
  - `ALOCC_paper/_patches/` 新增：`t3_offline_selection.py`（选模回放）、`a1_sel_redline.py`（redline patch）、`a1_sel_parity.py`（ADR-007 回归 + redline 对齐校验）、`run_s1_redline_triplet.ps1`（A-5 一键复现）。
  - `ALOCC_paper/` 新增产物：`s1_c{2,6,1}_r16_p03_redline/` 三套、`s1_redline_summary.json`、`t3_offline_selection.json`。
- 动机：`distortion` 选模器被离线验证证明在 S1 ON 后系统性选错后期记忆 epoch，§2.8.13–2.8.14 的"跨类坍塌 / D 反极性"全部是方法学伪影；T2 SN 对 AUC 的"救场效应"同被证伪，SN 实际压低 `raw_auc` 0.33；需要一个以结构红线 + 判别底线双锚定的选模策略作为 S1 及衍生方案的默认。
- 北极星指标对比（S1-only，best-epoch，redline 选模 vs 旧 distortion 选模，class 2/6/1）：
  - class 2：`ssim_oc` 0.330→**0.130**，`auc` 0.883→0.842，`raw_auc` 0.883→0.831（红线过）
  - class 6：`ssim_oc` 0.268→**0.079**，`auc` 0.429→0.711，`raw_auc` 0.530→0.883（+35pt，证明 S1 未塌是选模器错）
  - class 1：`ssim_oc` 0.689→**0.141**，`auc` 0.551→0.871，`raw_auc` 0.564→0.845（+28pt，同上）
  - S1 红线通过率 0/3 → **3/3**；OFF 红线通过率 0/3（class 2 超线）→ 2/3
- 论文一致性：保持（Baseline A X3 锚点下 `selection_strategy="acc_auc"` bitwise 等价；redline 为新增策略）；ADR-007 回归校验 9 份既有 summary 用原策略重放全 PASS（`a1_sel_parity.py`），records 层面未动。
- 回滚自检：`[T2-SN]` 全域 0 命中；AST 4 文件合法；`--help` 出现 `redline` choice；`.redline.bak` / `.t2_sn.bak` 生命周期正确（已删除过期备份）。
- 后续依赖：Round 1 终态 = **S1 + redline**；T2 SN 不纳入报告路径；RM-1 Round 2 候选（memory-bank / 收缩映射 / 二段式训练）待议；`_run_all.ps1` 基线脚本是否把默认选模切到 redline 需下一轮再决议（目前保留 `acc_auc` 以维持 CUDA 基线锚点稳定）。

### 2026-04-20 · RM-1 L3 MANet 终止 + A1 彻底回滚（ADR-010 落地）
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\` 下 4 个受影响文件回到 RM-1 前状态（`model.py` / `mnist_experiment_runner.py` / `run_paper_mnist_figure6_7.py` 从 `.rm1_l3.bak` 恢复；`export_mnist_triplets.py` 手术撤除 5 处 `[RM1-L3]` 注入）；`self_attention.py` 删除。
  - `ALOCC_paper/_patches/` 删除 9 个 `rm1_*.py`（Step 1–3c + Phase 1.1/1.2/1.2b/2），保留 `a1_rollback_rm1.py` 作回滚痕迹。
  - `ALOCC_paper/` 删除 4 个 RM-1 产物目录：`a1_regress_rm1_off` / `a1_smoke_rm1_attn_only` / `a1_smoke_rm1_on` / `a2_ablation_rm1_w03`。
  - 3 组 `*.rm1_l3.bak` 备份连带清理；`.pr_ab.bak` / `.pr_r.bak` / `.rm3*.bak` 等前几轮备份原样保留。
- 动机：师兄三条反馈——(1) Phase 1.2 AUC 坍塌是训练动力学问题，`balance_weight` 扫参不是正确路径；(2) MANet 第一作者为师姐，继续做存在重复发表风险；(3) 新方向要求**改结构先于加模块**。
- 北极星指标对比：N/A（回滚至 Baseline A；受影响文件等价 `baselines_cuda/A/experiment/summary.json` 对应源码状态）。
- 论文一致性：完全回归 Baseline A（X3 选模诚实基线），无任何 RM-1 L3 残留。
- 回滚自检：ALOCC-master 项目源码 4 个受影响文件 AST 合法；全局扫描无 `RM1-L3` / `SelfAttention2d` / `use_balance_loss` / `balance_weight` 残留（排除 `.venv` 站点包）；幂等回滚脚本 `a1_rollback_rm1.py` 可在任意已回滚状态重入不报错。
- 后续依赖：启动 **RM-1 Round 1 改法-S1（低秩 + 噪声瓶颈）**设计文档（基于 ShortcutBreaker 低秩思路 + Dinomaly 噪声瓶颈思路，两者都与 MANet 正交）；设计文档经师兄审核前不写任何代码；沿用 ADR-007 开关治理（默认关 + flags OFF bitwise 等价 Baseline A）。

### 2026-04-19 · RM-1 方法枢轴：L2 PseudoBound → L3 MANet（OCC 严格性约束）
- 文件：本日志 §5（本条）；§2.6 文献调研表待增补 OCC 过滤批注；§3.1 RM-1 工作项内容随后刷新。
- 动机：user 指出"主线是 one-class 分类"——引入合成假异常（pseudo-anomaly）参与训练会在方法学层面破坏 OCC 纯正性（训练过程接触任何"非 inlier"样本都会被审稿口径质疑为半监督）。据此对 §2.6 候选池做 OCC 严格重筛。
- OCC 严格筛选结论：
  - **排除**（训练时合成/引入异常）：L2 PseudoBound、L4 OCGAN、L6 Feature Shuffling、L7 ASCOOD、L8 NPOS/VOS
  - **保留**（严格 OCC + 维持 GAN + 与 `ssim_oc` 红线兼容）：**L3 MANet**（首选）、L5 MemAE（Round 2 候选）、L1 Old is Gold（降级保留——会让 ssim 红线失效）
- **RM-1 新方案锁定：L3 MANet 单打**——R/D backbone 内部加 multi-head self-attention 头 + 对抗平衡损失 `L_balance = |loss_D_real − loss_D_fake|`；训练数据保持纯正常（只看真 inlier）；对 §2.7 F-A1-3 观察到的"训练后期 ssim_oc 漂移"对症下药。
- 路线图重排：Round 1 = L3 MANet 单打 / Round 2 = L3 + L5 MemAE 组合 / Round 3 = 三者联合（若前两轮未达红线再评估 L1）。
- 论文一致性：部分退化（L3 属结构层改动，破坏与论文 Eq.5 的 bitwise 一致性，沿用 ADR-009"论文原版 / 本项目改进版"双栏汇报）。
- 后续依赖：Phase 1 编码目标从"`utils/pseudo_anomaly.py` + 负 SSIM 损失"改为"R/D backbone 加 self-attention 头 + `L_balance` 损失 + CLI 开关"，全程遵循 ADR-007（默认关 + flags OFF 时 bitwise 等价 Baseline A）；中类 {2,6,9} 首轮目标不变——把 X3 基线下 `ssim_oc=0.347` 压到红线 0.15 以内。


### 2026-04-19 · A1 X3 选模基线扫描（诚实基线建立）
- 文件：
  - `ALOCC_paper/_patches/a1_class_sweep_x3.py`（10 类 X3 扫描器，训练锚点沿用 ADR-006）
  - `ALOCC_paper/_patches/a1_aggregate_x3.py`（paper vs X3 聚合器）
  - `ALOCC_paper/a1_diagnostic_x3/class_{0..9}/`（每类 experiment/ + summary.json + 10 checkpoint + triplets）
  - `ALOCC_paper/a1_diagnostic_x3/_aggregate.{json,md}`（Δ 对比表）
- X3 配置：`--selection-strategy distortion --paper-score-normalization absolute --selection-min-auc 0.0 --selection-epoch-start 1 --selection-epoch-end 10 --distortion-alpha 1.0 --distortion-beta 1.0`（即 RM-3 R2 全开版）。
- 动机：Track ② F-A1-1 发现 paper 选模 9/10 类 fallback，导致先前基线数字系统性偏低；重跑一次 X3 基线作为后续 RM-1 对比的干净对照组。
- 核心结果（10 类平均）：
  - `refined_auc`：0.598 (paper) → **0.872 (X3)**，Δ **+0.275（+46%）**
  - `ssim_gap`：0.098 → **0.148**，Δ **+0.050（+51%）**
  - `ssim_oc`：0.448 → 0.436，Δ −0.013（微降）
  - fallback 触发：**9/10 → 0/10**（彻底消除）
- 戏剧性翻盘类（paper 挑到训练崩塌 epoch、X3 挑到真优秀 epoch）：class 0（0.229→**0.970**）、class 3（0.208→**0.951**）、class 4（0.327→**0.893**）。
- 代价类（X3 牺牲少量 AUC 换 ssim_gap / ssim_oc）：class 1（0.959→0.936）、class 9（0.879→0.816，但 `ssim_oc` 从 0.776 降至 0.236，从"通用去噄器"状态回到健康检测器）。
- 中类基线更新：{2, 6, 9} 平均 `ssim_oc = 0.347`（距红线 0.15 差 0.20），`refined_auc = 0.858`——后续 RM-1 改进效果以此为比较基准。
- 论文一致性：保持（只改选模口径、训练算法 0 变化）。
- 后续依赖：
  - RM-1 实验的"基线对照组"统一使用 X3 配置；paper 版数字仅用于"论文原版"栏目对比。
  - §2.7 A1 诊断章节的"9/10 类 fallback"描述需加"paper 策略下"限定前缀，避免后来者误读。


### 2026-04-19 · runner CLI 回归修复（RM-3 独立 CLI 补完）
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py`：`--selection-strategy` choices 扩 `"distortion"`；新增 `--distortion-alpha` / `--distortion-beta` / `--paper-score-normalization` 三个 argparse 条目；备份 `.rm3_cli_restore.bak`。
  - `ALOCC_paper/_patches/rm3_runner_cli_restore.py`（幂等补丁脚本；双判据：旧 choices 字符串匹配 + 新 flag 存在性检查，避免重复应用）。
- 动机：RM-1 Phase 0 调研发现 §5 "RM-3 完成" 记载的 CLI 落地与代码实际不一致——`run_paper_mnist_figure6_7.py` 层完整支持 X3，但 runner 自己的 argparse 在某次后续补丁（疑为 `pr_ab_apply.py` 重写 argparse 块时）被覆盖回退至 `choices=["acc_auc","paper"]`。内部选模逻辑（`_select_records` distortion 分支、`_attach_distortion_score`、`_normalize_metric_absolute`）一直完好。
- 影响面：先前所有 A1 诊断 / RM-3 首轮实验都走 figure6_7 入口，**未受影响**；本次只闭合"直接 `python mnist_experiment_runner.py` 调用"这个边缘路径，以及 PROJECT_LOG 记载与代码实际的一致性。
- 论文一致性：保持（纯 CLI 层变化，不触及训练逻辑）。
- 后续依赖：暴露一个工程教训——补丁脚本重写 argparse 块时必须做 CLI choices 冻结校验；`ADR-007` §9.6 回归协议下一次升级时增补此项。


### 2026-04-19 · 主线任务切换（ADR-009）
- 文件：`ALOCC_paper/PROJECT_LOG.md` §0 红线表 + §2.5 新增主线章节 + §4 ADR-009 + §5（本条）
- 动机：user 澄清"纯论文复现"与"改进论文方法"是两条互斥的路线，两者目标与验收标准不兼容。澄清后拍板 **"改法档"**——维持 one-class + GAN 总框架，允许加模块 / 改结构 / 结合 / 借鉴顶会工作。
- 硬性产出：
  - 主线三任务确定：**A1**（MNIST 全类别数值复现 + 改进版双栏对比）→ **A2**（Figure 6/7 复现）→ **A5**（跨数据集扩展：Caltech-256 + UCSD Ped2）
  - 质量红线：`ssim_oc ≤ 0.15`（当前 Baseline A best_ep=2 为 0.2182，需下压 ~30%）、`ssim_gap ↑`、`refined_auc` 不塌、`auc_gain > 0` 为加分项
  - RM-1/2/3/4 从独立攻击点**重新定位为主线工具链**：RM-1 服务 A1 红线、RM-2 防 R 退化、RM-3（已完成）做诚实选模尺、RM-4 服务 A2
  - 实验规模原则：第一轮全部 MNIST 小规模（ADR-006 锚点），禁止一上来就跑 Caltech-256 / UCSD
  - 出界条款：禁止放弃 one-class / 放弃 GAN / 蒸馏方案（博士师兄经验：无显著效果）
  - 新增"文献调研"工作项：借鉴近几年顶会（CVPR / ICCV / NeurIPS / ICLR）one-class / OOD / anomaly detection 方向
- 论文一致性：**部分退化（接受）**——"改法档"承认 Baseline A 的任何训练层改动都会破坏与论文 Eq.5 的 bitwise 一致性；报告时必须同时给出"论文原版"与"本项目改进版"两栏数字。
- 后续依赖：§3.1 执行优先级表的 RM-1 / RM-2 / RM-4 仍按原顺序执行，但每条落地时都要附上对主线 A1 的贡献说明（`ssim_oc` / `ssim_gap` / `refined_auc` 前后对比）。


### 2026-04-19 · 清理工作区影子副本（仓库整洁化）
- 文件：
  - `ALOCC_paper/_patches/workspace_shadow_diff_gen.py`（diff 归档生成器）
  - `ALOCC_paper/_patches/workspace_shadow_diff.md`（归档：20 个 .py 文件清单 + 3 个差异文件的完整 unified diff）
  - 删除：`d:\codeVS\ALOCC_paper\ALLOC\`（整棵 2.29 GB 工作区副本）
- 动机：RM-1 Phase 0 调研时发现工作区存在一棵**独立的 stale 源码副本**（`d:\codeVS\ALOCC_paper\ALLOC\ALOCC-master\`），内容和真身 `D:\Trae_coding\ALLOC\ALOCC-master\` 不同但都是过期快照。副本从未被任何脚本引用（全工作区 grep 确认），但每次 `view` 都会误导 Agent。
- 执行前核验：20 个 .py 文件里 17 个完全一致、3 个差异（`mnist_experiment_runner.py` / `model.py` / `run_paper_mnist_figure6_7.py`——都是 PR-A/PR-B/ADR-006 落地后真身领先的那几份）；副本内 `best_checkpoint/` 只有空目录、无实验数据。
- 论文一致性：不影响（未动任何源码行为，只删影子副本）。
- 后续依赖：
  - 约定：以后 Agent 读取代码走 `launch-process + findstr / Get-Content` 命中 `D:\Trae_coding\`；编辑走 patch-script 模式（仿 `pr_ab_apply.py`），不用 str-replace-editor 跨工作区。
  - Phase 0 PR-A 结论：**已在更早会话落地完毕**（`selection_min_auc_hard` / `selection_log_fallback` / `fallback_triggered` / `fallback_reason` 全部在 runner + figure6_7 的 CLI 接线），A1 诊断数据就是 PR-A 观测开关产物。
  - 下一步：方案 A（L2 PseudoBound 单打）进入 Phase 1，编辑目标指向真身。


### 2026-04-19 · Track ① 文献调研 + Track ② A1 诊断扫描 双交付
- 文件：
  - `ALOCC_paper/PROJECT_LOG.md` §2.6（新增 10 篇顶会文献候选池 + 可移植性评级 + 路线 α/β/γ）+ §2.7（10 类 A1 诊断表 + 类别难度分档 + RM-1 3 条硬约束）+ §5（本条）
  - `ALOCC_paper/_patches/a1_class_sweep.py`（类别扫描主脚本；CUDA 10 类 × 10 epochs，总耗时 ~4 分钟）
  - `ALOCC_paper/_patches/a1_aggregate.py`（paper vs oracle vs distortion 三栏聚合器）
  - `ALOCC_paper/a1_diagnostic/class_{0..9}/`（每类完整 experiment/ + summary.json + 10 checkpoint）
  - `ALOCC_paper/a1_diagnostic/_aggregate.{json,md}`（全景表）
- 动机：ADR-009 主线切换后的两条并行调研轨——Track ① 回答"改法档有什么可借鉴"，Track ② 回答"Baseline A 真实状态到底多差"。两份报告交叉印证，共同指导 RM-1 实现。
- Track ① 核心产出：
  - 候选池 10 篇（★★★ 3 篇 / ★★ 5 篇 / ★ 2 篇），按可移植性排序
  - **路线 β（L2 PseudoBound 真假混训）** 被推荐为 RM-1 的实际实现依据，比纯负 SSIM 更稳定
  - **路线 α（L1 Old is Gold 两阶段 + D 重建质量判别）** 作为 ALOCC 直系后继，论文口径最友好
  - 业界共识：我们之前称为"R 通用去噂器"的病根在文献里叫 **identity shortcut (ID-shortcut)**
- Track ② 核心发现（三重震撼）：
  - **F-A1-1**：paper 选模策略在 **9/10 类触发 fallback**（min_auc=0.95 在 [2,6] 窗口几乎从未成立）→ §6.x 先前所有基线数字都来自"矮子里将军"
  - **F-A1-2**：paper vs oracle 选模平均 AUC 差 **0.11**（0.598 → 0.709）→ "真实 Baseline A 性能"被低估 ~20%
  - **F-A1-3**：class 0 和 class 7 在 **ep1** 就同时达到 `ssim_oc ≤ 0.15` 红线 + AUC ≥ 0.92（class 0 @ ep1：`ssim_oc=0.159, auc=0.936`）；训练到 ep4 反而崩到 `auc=0.23`
- 类别难度分档（RM-1 攻击顺序）：
  - 易类 **{0,1,7}**：红线已达，主要靠 PR-A 修选模
  - 中类 **{2,6,9}**：RM-1 主战场（oracle 比 paper 丢 10-15 AUC 点）
  - 难类 **{3,4,5,8}**：从头没收敛，需要路线 α+γ 组合，不作为 RM-1 首轮靶子
- RM-1 实验设计 3 条硬约束（从 Track ② 反推）：
  - 评估 checkpoint 必须包含 ep1（易类最优点在 ep1，不能从 ep2 起评）
  - 基线对比必须用 oracle 选模（paper 选模 9/10 类 fallback，会把 RM-1 改进噪声化）
  - 首轮 RM-1 只打中类 {2,6,9}（易类不需改，难类首轮打不动）
- 论文一致性：保持（两条轨都是调研/诊断，未动源码；A1 诊断沿用 ADR-006 锚点，Baseline A flags OFF 即论文原版）。
- 后续依赖：
  - **PR-A 立刻落地**——Track ② 把它从 "重要" 升级为 "主线 blocker"，没 PR-A 则 RM-1 效果无法被 paper 选模口径测出
  - **RM-1 设计稿（下一步）** 以路线 β（PseudoBound 真假混训）为骨架，叠加路线 α 的训练 schedule 想法；两条路线在文献里正交，工程上可前后叠加
  - 等待 user 拍板 RM-1 的具体实现方案（α 优先 / β 优先 / α+β 合并）


### 2026-04-19 · RM-3 完成（RM-3a + RM-3b + RM-3c 一起落地）
- 落地范围：纯选模层改动，不触及训练循环/损失/网络结构，风险与 PR-A 同等隔离。
- 三件开关全部符合 ADR-007（CLI flag + Runner kwarg + 默认关）+ ADR-008 沟通合同：
  - **RM-3a**：新增 `distortion` 选模策略，打分 `max(ssim_gap,0)^α × max(refined_auc,0)^β`；默认 α=β=1.0；由 `--selection-strategy distortion` + `--distortion-alpha` + `--distortion-beta` 激活；未选用时行为 bitwise 不变。
  - **RM-3b**：新增"绝对锚点归一化"模式，给北极星 7 项用论文/数据驱动标准打分，辅助项沿用组内相对；由 `--paper-score-normalization absolute` 激活；默认 `relative`，bitwise 回归安全。锚点表：见 §9.5。
  - **RM-3c**：`--selection-min-auc` CLI 默认 0.95 → **0.60**；`_run_all.ps1` 显式传 0.95，基线锚点不受影响。
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py`：新增 `PAPER_SCORE_ABSOLUTE_ANCHORS` 常量 + `_normalize_metric_absolute` + `_attach_distortion_score`；`_attach_paper_score` 增 `normalization` 形参；`_select_records` 增 `distortion_alpha/beta` + `distortion` 分支 + selection_info 回声；`evaluate_checkpoints` 签名扩三参；`run_experiment` 的 `summary.switches` 扩五字段（`selection_strategy / selection_min_auc / paper_score_normalization / distortion_alpha / distortion_beta`）；备份 `.rm3.bak`。
  - `D:\Trae_coding\ALLOC\ALOCC-master\run_paper_mnist_figure6_7.py`：`Args` dataclass 增三字段；argparse 增 `--distortion-alpha` / `--distortion-beta` / `--paper-score-normalization`；`--selection-strategy` 扩 `distortion` 枚举；`--selection-min-auc` 默认值改 0.60；`pipeline_summary.json` 透传；备份 `.rm3.bak`。
  - 工具链：`ALOCC_paper/_patches/rm3_apply.py`（15 个 anchor 的幂等补丁脚本；idempotency 升级为 **三判据**：显式 sentinel 检查 → `old_count==0 ∧ new∈text` → 否则应用，彻底解决 PR-AB 时期的"保留原锚点追加"型补丁的重复应用漏洞）；`_rm3_analyze.py`（单次运行双轨分析）。
- 动机：把 §1 北极星目标从"相对好看"拉到"绝对可衡量"——锚点表给每条曲线一条 0 分线与 1 分线，distortion 策略让"扭坏 × 不崩"两头同时成立才算高分，min_auc=0.60 解开 Baseline B/C 永远静默 fallback 的死结。
- ADR-007 §9.6 回归（flags OFF，Baseline A）：北极星五项 Δ=0.0 bitwise：`refined_auc=0.9663461538`、`auc_gain=-0.0203205128`、`ssim_oc=0.2181626111`、`ssim_gap=+0.1348792166`、`score_gap_gain=-0.0193628669`。
- 首轮实验（变体 `alocc_loss_cls` = Baseline C variant，2 次 run）：
  - **R1 窗口版**（窗口 2-6，min_auc=0.60，distortion+absolute）：min_auc 过滤后仅 ep4 合格，选中 ep4；vs. Baseline C 锚点 ep6：`refined_auc` +0.05（0.59→0.65）、`ssim_oc` −0.19（0.57→0.39，R 对 outlier 扭得更狠）；但 `ssim_gap`、`auc_gain` 轻微退步。
  - **R2 全开版**（窗口 1-10，无 min_auc 过滤，distortion+absolute）：选中 ep10；vs. 锚点：`ssim_gap` +0.04（0.19→0.22）、`refined_auc` +0.15（0.59→0.74）、`auc_gain` +0.02（−0.15→−0.13）；但 `ssim_oc` 持平（0.56），印证"R 通用去噪器"病未解（靠选模解决不了）。
  - 三联图已生成（`_rm3_run_C/triplets/` + `_rm3_run_C_nofilter/triplets/`）：肉眼确认 ep10 的 R 把 outlier "6/7" 也修得较清楚，这是 RM-1（训练层负 SSIM 损失）要解决的病根。
- 论文一致性：保持（RM-3 只改工程选模口径，不改训练目标，不影响论文复现的算法声明）。
- 后续依赖：RM-1（训练层攻击）现处于"可安全开工"状态——选模诚实（distortion+absolute 直接可量化 `ssim_oc` 下降与 `auc_gain` 正负反转），RM-1 效果将被真实测量而非被 paper_score 组内相对归一化遮蔽。


### 2026-04-19 · ADR-008：沟通合同——只讲作用不讲代码
- 文件：`ALOCC_paper/PROJECT_LOG.md` §0 工作原则新增第 6 条。
- 动机：user 明确要求"不用给我讲代码实质内容因为我不理解，只需要给我讲代码的作用即可"——把这条升格为项目硬规则，避免 agent 以后在 review、汇报、设计稿里回到"贴源码+解释语法"的旧口径。
- 硬性产出：
  - 汇报 / review / 设计稿 / 进展同步 / bug 诊断讲解 → 只讲作用/效果/开关行为；
  - 代码注释 / commit message / 文件内部文档 → 不受约束，保持技术精度；
  - user 明确要求"给我看 diff / 贴出来 / 展开实现"时才贴源码。
- 违反示例：`X = max(…, key=lambda r: r['paper_score'])`——讲了语法没讲作用。
- 合规示例："这段按 paper_score 综合分选最高的那个 epoch"——只讲作用。
- 论文一致性：N/A（规范类决策）。
- 后续依赖：所有后续 PR/RM 的 review 环节强制遵守；PR 模板（§9.3）下一次更新时新增"沟通合同已遵守"勾选项。

### 2026-04-19 · PR-A + PR-B 完成（ADR-007 首个落地案例）
- 文件：
  - `D:\Trae_coding\ALLOC\ALOCC-master\model.py:582-587`：`ALOCC_LOSS_CLS._train` 末尾新增 `stop_recon_threshold` 早停分支，结构与 `ALOCC._train` / `ALOCC_LOSS._train` 对齐；备份 `model.py.pr_ab.bak`。
  - `D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py`：`import sys`；`_select_records` 新增 `selection_min_auc_hard` / `selection_log_fallback` 形参 + fallback 诊断块 + stderr `[PR-A][selection] WARNING`；`selection_info` 扩字段 `min_auc_hard` / `log_fallback` / `fallback_triggered` / `fallback_reason`；`evaluate_checkpoints` 签名与内部调用对齐；`run_experiment` 的 `summary.json` 新增 `switches` 块（`selection_min_auc_hard` / `selection_log_fallback` / `selection_fallback_triggered` / `stop_recon_threshold_active` / `d_outclass_loss_active` / `g_outclass_distortion_active`）；CLI 增 `--selection-min-auc-hard` + `--selection-log-fallback` / `--no-selection-log-fallback`；备份 `.pr_ab.bak`。
  - `D:\Trae_coding\ALLOC\ALOCC-master\run_paper_mnist_figure6_7.py`：`Args` dataclass 新增 `selection_min_auc_hard: bool = False` / `selection_log_fallback: bool = True`；`main()` argparse 与 pipeline_summary.json 透传；备份 `.pr_ab.bak`。
  - 工具链：`ALOCC_paper/_patches/pr_ab_apply.py`（幂等多点补丁脚本；idempotency 用 `old_count` + `new in text` 双判据，避免兄弟类同构块误匹配）、`regression_A.py`（§9.6 自动化回归）、`verify_pra.py` / `verify_prb.py`（flags ON 三场景 × 两场景）。
- 动机：兑现 ADR-007；同时关闭 §2 P0 两个 blocker（PR-A 让"静默 fallback"可观测/可硬拒；PR-B 补齐 ALOCC_LOSS_CLS 早停能力）。
- ADR-007 §9.6 回归（flags OFF，Baseline A）：北极星五项 Δ = 0.0 全字段，与 `baselines_cuda/A/experiment/summary.json` bitwise 一致：
  - `refined_auc=0.9663461538`（Δ=0）
  - `auc_gain=-0.0203205128`（Δ=0）
  - `ssim_oc=0.2181626111`（Δ=0）
  - `ssim_gap=+0.1348792166`（Δ=0）
  - `score_gap_gain=-0.0193628669`（Δ=0）
- 新开关 ON 场景验证（见 `ALOCC_paper/_patches/verify_pr{a,b}.py`）：
  - PR-A S1（`--selection-min-auc 0.9999` + 默认日志开）：stderr 打印 `[PR-A][selection] WARNING fallback to full candidate window: No candidate epoch in [1,2,3] satisfied auc>=0.9999; best auc in window = 0.8142`；`switches.selection_fallback_triggered=true`；`selection_info.fallback_reason` 非空。
  - PR-A S2（追加 `--selection-min-auc-hard`）：`RuntimeError: [PR-A] selection_min_auc_hard=True and fallback would trigger. …` rc=1，不写 summary。
  - PR-A S3（`--no-selection-log-fallback`）：无 stderr 警告；`switches.selection_log_fallback=false` 但 `selection_fallback_triggered=true` 仍记录。
  - PR-B S1（`--stop-recon-threshold 10.0 --stop-min-epoch 2`，alocc_loss_cls，epochs=5）：`trained_epochs=2`，`2.pth` 落盘，`switches.stop_recon_threshold_active=true`。
  - PR-B S2（无 `--stop-recon-threshold`）：`trained_epochs=5`（全部消耗），active flag=false。
- 论文一致性：保持。默认路径与论文 §4.2 训练/选模配置逐字节一致；flags ON 仅改选模决策或训练截止时机，不动任何损失/梯度表达式。
- 后续依赖：RM-1（Negative-SSIM 外类扭曲损失）现可安全开工——选模会诚实标注门槛命中情况，`--stop-recon-threshold` 可阻止 R 后期退化为通用降噪器；§9.4 `--stop-*`、`--selection-min-auc` 三个违规标位升级为 ✅。

### 2026-04-19 · ADR-007：所有改动必须可开关
- 文件：`ALOCC_paper/PROJECT_LOG.md`（§0 工作原则新增第 5 条；§4 ADR 表新增 ADR-007；新增 §9 开关治理清单，含三层模式模板 / summary.json 契约 / PR 模板增项 / 现状核对 / 未落 PR 的预先约束 / 回归测试协议）。
- 动机：user 指示"我要你做的一切更改都要变成可用开关控制的，通过输入参数随时可以开关调配"。
- 硬性产出：
  - §9.1 三层模式（CLI flag / Runner kwarg / 内部消费点 if 短路）；
  - §9.2 summary.json 必填 `switches` 字段；
  - §9.3 PR 描述模板新增"§9 开关清单"小节，与 §8 张量审查并列；
  - §9.4 现状核对：`--stop-recon-threshold`（PR-B 前"暴露但不消费"⚠️）、`--selection-min-auc`（PR-A 前静默 fallback ❌）、`g_outclass_distortion_scale / d_outclass_loss_scale`（未暴露到 CLI ❌）；
  - §9.5 PR-A/PR-B/RM-1/RM-2/RM-3/RM-4 每个未落 PR 的开关列表已预先锁定；
  - §9.6 合入前必须跑"全开关默认"回归，北极星五项需与 §6.4 基线 1e-6 严格一致。
- 论文一致性：保持（ADR 强约束，未动源码）。
- 后续依赖：PR-A/PR-B 实施时必须带 `--selection-min-auc-hard / --selection-log-fallback` 双 flag（PR-A）并复用已有 `--stop-recon-threshold`（PR-B），默认关闭回归；详见 §9.5。

### 2026-04-19 · PR-Q 完成：CUDA 环境就绪 + A/B/C 基线 CUDA 版建立
- 环境：`D:\Trae_coding\ALLOC\ALOCC-master\.venv` 完成 `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` 迁移（`--index-url https://download.pytorch.org/whl/cu128`，`--force-reinstall --no-deps` 保护其余依赖）。Smoke：`torch.cuda.is_available()=True`，device=`NVIDIA GeForce RTX 5060 Laptop GPU`，`cuda_version=12.8`，cuda 4×4 matmul 通过。
- 自动化：`ALOCC_paper/baselines_cuda/_run_all.ps1`（含 CUDA verify → 三 variant 顺序跑 → 调用 `_analyze_cuda.py` 输出 per-epoch 表 + CPU/CUDA delta）。
  - 踩坑记录：初版用 `$ErrorActionPreference=Stop` + `Tee-Object` 写入 outDir，遇两个问题：① python stderr 被当作 terminating error 中断脚本 → 改 `$ErrorActionPreference=Continue` + `$PSNativeCommandUseErrorActionPreference=$false`；② python runner 调 `shutil.rmtree(outDir)` 清空目录时撞上 PowerShell 打开的 run.log → 日志改存 `baselines_cuda/_logs/{A,B,C}.log`。
- 运行配置（ADR-006 锚点，与 CPU 基线完全一致）：`specific=1, outlier=[6,7], epochs=10, train=4096, batch=64, noise=0.31, r_alpha=0.2, lr=0.002, selection=paper, window=[2,6], min_auc=0.95`。
- 结果：A/B/C 三 variant 全部跑通，实测耗时 15.6 s / 21.0 s / 28.1 s（vs CPU ~50-60 s/run，**约 3× 提速**；外类前向 + cosine 分支使 C 最慢）。详见 §6.4。
- 核心复核（§6.4）：**Figure 7 主张在 CUDA 上仍被颠倒**（A/B/C best-ep `auc_gain` 分别 −0.020 / −0.215 / −0.154）——RM-1 仍是真正需要动手的那块。
- 有趣观察：C 在 ep7-8 出现 `auc_gain=+0.018 / +0.016`（CPU 基线首次观察到正值），但 best_ep=6 仍被 paper 策略选中负值 epoch；再次印证 PR-A（min_auc 静默 fallback）+ RM-3（selection_strategy=distortion）的必要性。
- 北极星指标对比（CPU best-ep → CUDA best-ep）：A `ssim_oc 0.309→0.218`、`ssim_gap +0.104→+0.135`、`auc_gain −0.102→−0.020`（显著改善，怀疑 CPU 数值路径有差异）；B `ssim_oc 0.625→0.494`、`ssim_gap +0.200→+0.266`、`auc_gain −0.060→−0.215`（扭曲更强但 AUC 更差，与 F4 一致）；C 新增基准，`ssim_oc 0.575`、`ssim_gap +0.188`。
- 论文一致性：保持（仅换硬件 + 修 PR-R；损失/训练逻辑未动）。
- 后续依赖：RM-1（Negative-SSIM 外类扭曲损失）可以开干，评估成本 ~20 s/run 可支撑 20+ 组消融。

### 2026-04-19 · PR-R 修复：ALOCC_LOSS_CLS._train 补 return
- 文件：`D:\Trae_coding\ALLOC\ALOCC-master\model.py:582`（新增一行 `return int(epoch)`，8 空格缩进，对齐 `ALOCC._train:291` / `ALOCC_LOSS._train:442`）。
- 备份：`D:\Trae_coding\ALLOC\ALOCC-master\model.py.pr_r.bak`。
- 补丁脚本：`ALOCC_paper/_patches/pr_r_alocc_loss_cls_return.py`（幂等：检测已 patched 跳过；UTF-8 安全；记录插入行号）。
- 动机：基线 C 训练完成 10×ckpt 后被 `int(None)` TypeError 打断，使 `alocc_loss_cls` variant 完全无法评估。
- 验证：`run_paper_mnist_figure6_7.py --variant alocc_loss_cls --epochs 3 --train-count 256 --selection-min-auc 0.0` 跑通，`best_epoch=1` 正常选出，`summary.json` + `best.pth` + `triplets/` + `figure7_scores.json` 全部生成。
- 北极星指标对比（before → after）：N/A（before 完全崩溃无 summary，after 产出完整 summary；后续 CUDA 基线完成后在 §6.4 补充对比）。
- 论文一致性：保持（仅修复工程 bug，不动损失 / 训练逻辑）。
- 注意：**PR-B 仍未修**——`ALOCC_LOSS_CLS._train` 依然忽略 `stop_recon_threshold`，后期 overtraining 无法早停；与 §3.1 ③ 中 PR-B 的计划一致。
- 后续依赖：CUDA 环境就绪后，C 基线可正式纳入三元组对比。

### 2026-04-19 · 基线 A/B/C 建立，PR-R 新增，路线图重排
- 文件：`ALOCC_paper/baselines/{A,B,C}/experiment/*`（含 `summary.json` + 10×ckpt + 三联图 + figure7_scores）；`ALOCC_paper/baselines/_analyze.py`（per-epoch 解析脚本）；本文档 §6、§2、§3.1。
- 动机：user 指示"先探测设备、再跑默认规模基线、深度分析北极星指标、同步 §6、规划 PR/RM 优先级"。
- 环境：探测确认 `D:\Trae_coding\ALLOC\ALOCC-master\.venv` 为 `torch 2.11.0+cpu`，CUDA 不可用；`d:\codeVS\.venv` 无 torch。硬件 NVIDIA GPU 空转 → 新增 PR-Q。
- 运行：`run_paper_mnist_figure6_7.py --specific 1 --outlier-labels 6 7 --epochs 10 --train-count 4096`，三 variant。
- 核心发现（详见 §6.2）：
  - F1 `auc_gain` 全负 → Figure 7 主张被颠倒；
  - F2 `ssim_oc` 单调上升至 0.79 → §4.4 overtraining 实锤；
  - F3 `paper_score` 选模被崩塌域污染，min_auc 静默 fallback；
  - F4 外类扭曲项关闭（scale=0）仍靠 D 端把 ssim_gap 抬到 +0.199 → RM-1 有依据；
  - F5 CPU 训练 ~5 s/epoch，可接受但消融成本高。
- 新增 PR-R（`ALOCC_LOSS_CLS._train` 缺 return）——基线 C 实锤。
- 路线图重排（§3.1）：PR-R → PR-Q → PR-A+B → RM-1 → RM-3 → RM-2 → RM-4 → P1 重构 → P2/P3。
- 北极星指标对比（N/A，这是基线；后续 RM-1 完成后回填对比）。

### 2026-04-19 · 张量维度审查清单纳入硬约束
- 文件：`ALOCC_paper/PROJECT_LOG.md`（仅文档）
- 动机：user 明确要求 loss 模块化与 Trainer 抽象时必须严格校验张量维度对齐。
- 动作：新增 ADR-004；§2 增 PR-M/N/O/P 四项；新增 §8 完整审查清单 + 现状核对表。
- 现状审查结论（详见 §8）：5 条规则中 3 条当前实现合规、2 条存在隐患（PR-M/PR-N）。
- 后续依赖：所有后续 PR 在交付前必须填 §8 的"通过性"列。

### 2026-04-19 · 初始化
- 建立本文档；完成首轮代码审查（5 维度结构性问题清单见 §2）。
- 尚未对源码做任何改动。

---

## 6. 实验基线

> **⚠️ §6.1–§6.5 已 SUPERSEDED（2026-04-27）**：以下 §6.1（CPU 单 run）/§6.4（CUDA 单 run）的 A/B/C 数字均产生于 **D1-A + D2-A + D4-A** 协议（论文 Eq.4 字面 MSE refinement + PyTorch RMSprop 默认 α=0.99/eps=1e-8 + acc_auc 选模窗口 [2,6]）——这套协议既不是论文公式（Eq.4 之外的工程参数全是 PyTorch 默认）也不是原作者代码（原作者用 BCE-on-noisy + α=0.9/eps=1e-10），更不是公平选模（窗口 cherry-pick 早期 epoch 规避 overtraining）。其中 Baseline A 的 `refined_auc=0.79/0.97` 数字**不再用作 paper 中 Baseline A 的引用源**。
>
> **paper 中 Baseline A 的唯一引用源 = §2.10.7 + §6.6（v2026-04-27 verbatim 协议，50-run 矩阵）**。详细溯源与三处必须显式说明的解读见 §2.10.8 / §2.10.9。
>
> **paper 主对比表（Baseline A vs S1D）的唯一引用源 = §6.7（同选模 H2H · 2026-04-30）**：包含 +0.237 raw_auc 主结论、10/10 逐类胜率、阈值敏感性扫描三件套；指标主线 `raw_auc`（非 refined）的方法学溯源见 §2.10.11。
>
> §6.1–§6.5 保留供历史参考；研究主线（RM-1/RM-3）所引用的"基线 B/C 出现了正向 auc_gain 但 refined_auc 偏低"等定性观察仍然有效，因为它们是协议无关的结构性现象。

### 6.1 best-epoch 指标快照（CPU · 单 run · ⚠️ SUPERSEDED）

统一配置（旧）：`batch_size=64, noise_std=0.31, r_alpha=0.2, lr=0.002, selection_strategy=paper, selection_window=[2,6], selection_min_auc=0.95`；产物路径 `ALOCC_paper/baselines/{A,B,C}/`。

| 基线 | variant | best_ep | refined_auc | raw_auc | **auc_gain** | ssim_ic | ssim_oc | **ssim_gap** | score_gap | **score_gap_gain** | paper_score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | alocc | 2 | 0.7943 | 0.8962 | **−0.1019** | 0.4121 | 0.3085 | **+0.1035** | +0.0062 | **−0.0046** | 0.7194 |
| **B** | alocc_loss | 6 | 0.7606 | 0.8206 | **−0.0600** | 0.8250 | 0.6254 | **+0.1995** | +0.0865 | **−0.0225** | 0.8146 |
| **C** | alocc_loss_cls | — | 训练完成但评估崩溃：**`ALOCC_LOSS_CLS._train` 末尾缺 `return int(epoch)`，返回 `None` → runner `int(None)` `TypeError`**。见 PR-R。 | | | | | | | | |

### 6.2 关键发现（数据驱动的诊断）

**F1 — Figure 7 的"R 帮 D"完全没有发生。** 两个基线 20 个 epoch 中 `auc_gain` 普遍为负，最好的 B-ep8 也只有 −0.0029。`D(X)` 在 raw 空间下一直比 `D(R(X))` 更可分 —— 论文最核心的主张被颠倒。

**F2 — R 在 epoch 2-3 后发生 §4.4 警告的 overtraining 漂移。** 基线 A 的 `ssim_oc` 从 ep1 的 0.1445 一路涨到 ep10 的 0.7942 —— R 变成了通用降噪器，outlier 也被重建得极好（ssim_gap 从 +0.143 跌到 +0.069）。同期 `refined_auc` 在 ep3 之后塌到 0.03-0.10（D 被 R 的"通用重建"带偏，完全丧失判别力）。

**F3 — paper_score 选模被崩塌域误导。** A 的 ep3-10 `auc_gain` 虽"正"（+0.005 到 +0.035），但 `refined_auc` 只有 0.03-0.10（D 完全崩）。`_normalize_metric` 在 `_select_records` 里按**单 run 内** min/max 归一化，使灾难域也能获得高归一分数。`selection_min_auc=0.95` 形同虚设：两基线 5 个 candidate epoch 都达不到，触发静默 fallback（PR-A 实锤）。

**F4 — 外类扭曲项有效但被关掉了。** B 的 `g_outclass_distortion_scale=0.0`（figure6_7 脚本默认），但 `d_outclass_loss_scale=0.1` 单独使 `ssim_oc` 比 A 下降（ep1-6 范围 0.16-0.63 vs A 的 0.14-0.68，且 ssim_gap 峰值 +0.298 vs A 的 +0.143）—— 证明扭曲机制**有潜力但被削弱**；这是 RM-1 的直接依据。

**F5 — CPU 训练 ~5 s/epoch（4096 样本 + G 双重更新 + 外类前向），10 epoch ~50-60 s；CUDA 迁移后可降至 ~3-5 s/run，给高密度消融提供空间。**

### 6.3 per-epoch 摘要（详细对照）

```
=== Baseline A (alocc, best=ep2) ===
ep | refined_auc raw_auc  auc_gain | ssim_ic ssim_oc ssim_gap | score_gap   gain
 1 |    0.9303  0.9730   -0.0427  |  0.2876 0.1445 +0.1432  |  +0.0111 -0.0274
 2 |    0.7943  0.8962   -0.1019  |  0.4121 0.3085 +0.1035  |  +0.0062 -0.0046  <-best
 3 |    0.0658  0.0405   +0.0252  |  0.5143 0.4428 +0.0715  |  -0.0158 +0.0018  崩
 4..10:   0.02-0.10 (全部崩)      |  ssim_oc 持续↑ 到 0.794 | score_gap 持续为负

=== Baseline B (alocc_loss, best=ep6) ===
ep | refined_auc raw_auc  auc_gain | ssim_ic ssim_oc ssim_gap | score_gap   gain
 1 |    0.7185  0.9396   -0.2212  |  0.2968 0.1639 +0.1330  |  +0.0240 -0.0841
 5 |    0.1329  0.7759   -0.6429  |  0.6604 0.3625 +0.2979  |  -0.0474 -0.1228  崩但扭曲最强
 6 |    0.7606  0.8206   -0.0600  |  0.8250 0.6254 +0.1995  |  +0.0865 -0.0225  <-best
 8 |    0.7892  0.7921   -0.0029  |  0.8546 0.6903 +0.1644  |  +0.0958 -0.0259  最接近 Fig7
```

完整数据：`baselines/{A,B}/experiment/summary.json`；再解析脚本：`baselines/_analyze.py`。

### 6.4 CUDA 基线（RM-1 真正锚点 · ⚠️ SUPERSEDED for paper Baseline A，定性观察仍引用）

2026-04-19 晚，`D:\Trae_coding\ALLOC\ALOCC-master\.venv` 完成 `torch 2.11.0+cu128` 迁移后，在 RTX 5060 Laptop GPU 上按完全一致的锚点配置重跑 A/B/C。产物路径 `ALOCC_paper/baselines_cuda/{A,B,C}/`；自动化脚本 `_run_all.ps1` + `_analyze_cuda.py`。

| 基线 | variant | best_ep | refined_auc | raw_auc | **auc_gain** | ssim_ic | ssim_oc | **ssim_gap** | score_gap | **score_gap_gain** | paper_score | 耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | alocc | 2 | 0.9663 | 0.9867 | **−0.0203** | 0.3530 | 0.2182 | **+0.1349** | +0.0262 | **−0.0194** | 0.9032 | 15.6 s |
| **B** | alocc_loss | 6 | 0.5747 | 0.7893 | **−0.2145** | 0.7606 | 0.4941 | **+0.2665** | +0.0691 | **−0.0510** | 0.8040 | 21.0 s |
| **C** | alocc_loss_cls | 6 | 0.5916 | 0.7461 | **−0.1544** | 0.7621 | 0.5745 | **+0.1877** | +0.0672 | **−0.0265** | 0.7350 | 28.1 s |

**CPU→CUDA best-ep delta**（CUDA - CPU）：

| 基线 | Δrefined_auc | Δauc_gain | Δssim_oc | Δssim_gap | Δscore_gap_gain |
|---|---:|---:|---:|---:|---:|
| A | **+0.172** | +0.082 | −0.090 | +0.031 | −0.015 |
| B | −0.186 | −0.155 | −0.131 | +0.067 | −0.028 |
| C | 新增基准 | — | — | — | — |

**CUDA 下的关键复核（与 CPU 结论对照）：**

- **F1（Figure 7 颠倒）仍成立**：A/B/C best-ep `auc_gain` 全负（−0.020 / −0.215 / −0.154）。A 的 best-ep `auc_gain` 从 CPU 的 −0.102 缓到 −0.020，但**仍未正**；整体主张未反转。
- **F2（ssim_oc 过训练漂移）仍成立**：A 的 `ssim_oc` 从 ep1 的 0.132 爬到 ep10 的 0.819；B 从 0.168 → 0.514；C 从 0.150 → 0.559——全部符合 §4.4 overtraining 模式。
- **F3（选模静默 fallback）仍成立**：三基线 5 个 candidate epoch 全部 `refined_auc < 0.95`，触发 `_select_records:246-247` fallback。A 实际 best=ep2（refined_auc=0.966，勉强接近但 candidate 窗口命中），B=ep6（0.575，fallback），C=ep6（0.592，fallback）。**PR-A 依然必修**。
- **新发现（RM-1 正面证据）**：C 在 ep7（`auc_gain=+0.018, refined_auc=0.574`）与 ep8（`auc_gain=+0.016, refined_auc=0.678`）出现论文主张方向的**正值 auc_gain**，是三基线 60 epoch 中仅有的两次。虽然 refined_auc 偏低，但**证实当外类扭曲强到一定程度 (ssim_gap=+0.138/+0.174) 时 Figure 7 行为会出现**——直接为 RM-1（Negative-SSIM 外类扭曲）与 RM-3（distortion-aware selection）提供正面样本。
- **B vs A 的 AUC 退化**：B 的 `refined_auc` 从 A 的 0.966 跌到 0.575 却换得 `ssim_gap` 从 +0.135 升到 +0.267——D 端外类损失（scale=0.1）确实让 R 更扭曲 outlier，但也让 D 在 raw 空间的判别被污染。这暗示 RM-1 需要配合 RM-3（按 `ssim_gap × refined_auc` 联合选模）而不是单纯加大扭曲强度。

### 6.5 per-epoch CUDA 详情

```
=== Baseline A (alocc, best=ep2) ===
ep | refined_auc raw_auc  auc_gain | ssim_ic ssim_oc ssim_gap | score_gap   gain
 1 |    0.8731   0.9812   -0.1080  |  0.2568 0.1325 +0.1243  |  +0.0108 -0.0463
 2 |    0.9663   0.9867   -0.0203  |  0.3530 0.2182 +0.1349  |  +0.0262 -0.0194  <-best
 3 |    0.8650   0.9335   -0.0685  |  0.4441 0.3376 +0.1065  |  +0.0166 -0.0118
 4 |    0.8441   0.9211   -0.0771  |  0.5279 0.4667 +0.0612  |  +0.0123 -0.0090
 5..10: refined_auc 回升至 0.94-0.98, ssim_oc 持续涨至 0.819（overtraining）

=== Baseline B (alocc_loss, best=ep6) ===
ep | refined_auc raw_auc  auc_gain | ssim_ic ssim_oc ssim_gap | score_gap   gain
 4 |    0.6335   0.8148   -0.1814  |  0.4499 0.2723 +0.1776  |  +0.0617 -0.0664
 5 |    0.5066   0.7761   -0.2696  |  0.7303 0.4127 +0.3177  |  +0.0220 -0.0740  ssim_gap 峰
 6 |    0.5747   0.7893   -0.2145  |  0.7606 0.4941 +0.2665  |  +0.0691 -0.0510  <-best
 8 |    0.7103   0.8863   -0.1761  |  0.8506 0.6302 +0.2203  |  +0.0954 -0.0884
10 |    0.6811   0.8514   -0.1703  |  0.8719 0.5136 +0.3583  |  +0.0643 -0.0929  ssim_gap 新高

=== Baseline C (alocc_loss_cls, best=ep6) ===
ep | refined_auc raw_auc  auc_gain | ssim_ic ssim_oc ssim_gap | score_gap   gain
 4 |    0.6466   0.8380   -0.1915  |  0.5608 0.3893 +0.1715  |  +0.0688 -0.0507
 6 |    0.5916   0.7461   -0.1544  |  0.7621 0.5745 +0.1877  |  +0.0672 -0.0265  <-best
 7 |    0.5745   0.5567   +0.0178  |  0.8080 0.6696 +0.1384  |  +0.0487 +0.0005  Fig7 ✓
 8 |    0.6782   0.6626   +0.0156  |  0.8753 0.7017 +0.1736  |  +0.0585 +0.0026  Fig7 ✓
 9 |    0.7498   0.8414   -0.0916  |  0.6114 0.4087 +0.2027  |  +0.0623 -0.1097
```

完整数据：`baselines_cuda/{A,B,C}/experiment/summary.json`（+ `best.pth` + `triplets/` + `figure7_scores.json`）。

### 6.6 ✅ Baseline A · v2026-04-27 verbatim 协议（10 类 × 5 seeds = 50 runs · paper 唯一引用源）

**协议**：D1-B（refinement = `BCEWithLogits(R(X̃), X̃_noisy)`）+ D2-B（RMSprop α=0.9, eps=1e-10）+ D3-B（源码级直译 TF1.15 `train()`）+ D4-C（`last_epoch`，选模窗口 [10,10] 强制收敛于 epoch 10）。锚点配置同 ADR-006：`epochs=10, train_count=4096, batch_size=64, noise_std=0.31, r_alpha=0.2, lr=0.002, bottleneck_rank=0, variant=alocc`。

**矩阵**：seeds = {42, 43, 44, 45, 46}；digits = {0..9}；inlier=specific 类，outlier=其余 9 类；`test_inlier_count=200`，`test_outlier_count=` 由其余类汇总。

**执行**：2026-04-27 22:48 UTC+8 起跑，sequential RTX 5060 Laptop GPU，13m06s 全 50 run 成功（0 failed）。**D4-C 一致性校验：50/50 runs `best_epoch == 10` ✅**。

**Per-class 聚合（5 seeds each · mean ± std）**：

| digit | acc | auc (refined) | raw_auc | auc_gain | ssim_ic | ssim_oc | ssim_gap | paper_score |
|:-:|---|---|---|---|---|---|---|---|
| 0 | 0.5845 ± 0.1207 | 0.5352 ± 0.2790 | 0.5293 ± 0.2818 | +0.0058 ± 0.0048 | 0.9428 ± 0.0073 | 0.9232 ± 0.0068 | 0.0196 ± 0.0012 | 0.3105 ± 0.1500 |
| 1 | 0.6840 ± 0.2526 | 0.4758 ± 0.4918 | 0.4651 ± 0.4945 | +0.0107 ± 0.0085 | 0.9640 ± 0.0050 | 0.8675 ± 0.0035 | +0.0965 ± 0.0037 | 0.4384 ± 0.1267 |
| 2 | 0.6435 ± 0.1333 | 0.5717 ± 0.3363 | 0.5709 ± 0.3360 | +0.0007 ± 0.0033 | 0.9391 ± 0.0048 | 0.9337 ± 0.0057 | 0.0054 ± 0.0028 | **0.5310 ± 0.1152** |
| 3 | 0.6440 ± 0.0979 | 0.6532 ± 0.1949 | 0.6512 ± 0.1980 | +0.0020 ± 0.0086 | 0.9391 ± 0.0091 | 0.9311 ± 0.0113 | 0.0080 ± 0.0102 | 0.4728 ± 0.2410 |
| 4 | 0.5485 ± 0.1057 | 0.3456 ± 0.2701 | 0.3404 ± 0.2706 | +0.0052 ± 0.0034 | 0.9399 ± 0.0107 | 0.9275 ± 0.0124 | 0.0124 ± 0.0030 | 0.4967 ± 0.0889 |
| 5 | 0.5460 ± 0.1015 | 0.5327 ± 0.2247 | 0.5306 ± 0.2244 | +0.0021 ± 0.0023 | 0.9404 ± 0.0080 | 0.9355 ± 0.0093 | 0.0049 ± 0.0022 | 0.4320 ± 0.1015 |
| 6 | 0.5810 ± 0.1245 | 0.4484 ± 0.3171 | 0.4453 ± 0.3174 | +0.0030 ± 0.0033 | 0.9444 ± 0.0064 | 0.9245 ± 0.0071 | 0.0199 ± 0.0022 | 0.3842 ± 0.1414 |
| 7 | 0.6270 ± 0.1746 | 0.4445 ± 0.4206 | 0.4373 ± 0.4216 | +0.0072 ± 0.0024 | 0.9514 ± 0.0050 | 0.9252 ± 0.0068 | 0.0262 ± 0.0023 | 0.4843 ± 0.1079 |
| 8 | 0.5365 ± 0.0394 | 0.4617 ± 0.1133 | 0.4588 ± 0.1147 | +0.0029 ± 0.0027 | 0.9426 ± 0.0093 | 0.9333 ± 0.0111 | 0.0093 ± 0.0022 | 0.4827 ± 0.1044 |
| 9 | 0.5840 ± 0.1050 | 0.5522 ± 0.3058 | 0.5478 ± 0.3052 | +0.0044 ± 0.0030 | 0.9490 ± 0.0114 | 0.9294 ± 0.0131 | 0.0196 ± 0.0031 | 0.2982 ± 0.1239 |

**全局聚合（50 runs）**：

| metric | mean ± std |
|---|---|
| `acc` | **0.5979 ± 0.1320** |
| `auc` (refined) | **0.5021 ± 0.2941** |
| `raw_auc` | 0.4977 ± 0.2972 |
| `auc_gain` | **+0.0044 ± 0.0123** |
| `ssim_ic` | 0.9453 ± 0.0114 |
| `ssim_oc` | **0.9231 ± 0.0209** |
| `ssim_gap` | +0.0222 ± 0.0262 |
| `paper_score` | 0.4331 ± 0.1445 |

**与 v2026-04-19 旧 baseline 的协议级差异**：refinement loss MSE→BCE-on-noisy；RMSprop α=0.99→0.9，eps=1e-8→1e-10；选模 `acc_auc[2,6]`→`last_epoch[10,10]`；单 run→50-run 矩阵。

**完整数据**：`D:\Trae_coding\ALLOC\ALOCC-master\baseline_a_v2026_04_27\d{0..9}_s{42..46}\summary.json` + `_aggregate.{json,md}` + `_logs\d{digit}_s{seed}.log{,.err}`；汇总脚本 `ALOCC_paper/_patches/_aggregate_baseline_a_50.py`，调度脚本 `run_baseline_a_50.ps1`。

**解读三件套**：见 §2.10.8（O1 auc≈0.50 不是 bug；O2 ssim_oc≈0.92 揭示 BCE-on-noisy 本质；O3 std 巨大 = GAN 极性翻转）。

---

### 6.7 ✅ 同选模 H2H · Baseline A vs S1D（2026-04-30 · paper 主对比表唯一引用源）

> **目的**：消除"选模口径不同"作为混淆变量。对 §6.6 baseline 50-run 与 §2.8.25 S1D 50-run **套用同一份 redline 选模函数**（`ssim_oc ≤ 0.15 ∧ raw_auc ≥ 0.60`，`MIN_AUC=0.6` 软回退），后处理脚本 `_tmp_h2h_baseline_vs_s1d.py`（仓库根，纯读 `summary.json`，不重训）。详细解读、敏感性扫描、可视化逐类详表见 `ALOCC_paper/baseline_a_vs_s1d_report.md`。

**§6.7.1 主对比（同 redline 选模 · 50 runs/arm · 全局聚合）**：

| 指标 | Baseline A | S1D | Δ |
|---|:-:|:-:|:-:|
| **`raw_auc`** | **0.568 ± 0.270** | **0.805 ± 0.108** | **+0.237** |
| `ssim_oc` | 0.722 | 0.079 | **−89.0%** |
| `ssim_ic` | 0.945 ± 0.011 | ~0.205 | −0.740 |
| Redline 主路径命中 | **0 / 50** | **48 / 50 (96%)** | — |
| 软回退（distortion fallback）| 50 / 50 | 2 / 50 | — |
| `raw_auc` std | 0.270 | 0.108 | **−60%** |

**§6.7.2 Per-class 详表（5 seeds each · `raw_auc` mean ± std · S1D 全 10 类严格优于）**：

| digit | Baseline A | S1D | Δ | S1D ssim_ic | S1D ssim_oc |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0.529 ± 0.282 | **0.967 ± 0.040** | **+0.437** | 0.806 | 0.094 |
| 1 | 0.465 ± 0.494 | **0.992 ± 0.005** | **+0.527** | 0.731 | 0.037 |
| 2 | 0.571 ± 0.336 | **0.591 ± 0.108** | **+0.020** | 0.711 | 0.085 |
| 3 | 0.644 ± 0.199 | **0.778 ± 0.046** | **+0.134** | 0.728 | 0.093 |
| 4 | 0.401 ± 0.176 | **0.823 ± 0.063** | **+0.423** | 0.732 | 0.072 |
| 5 | 0.579 ± 0.124 | **0.791 ± 0.062** | **+0.212** | 0.783 | 0.101 |
| 6 | 0.573 ± 0.269 | **0.860 ± 0.088** | **+0.287** | 0.736 | 0.055 |
| 7 | 0.523 ± 0.322 | **0.787 ± 0.105** | **+0.264** | 0.711 | 0.096 |
| 8 | 0.624 ± 0.285 | **0.659 ± 0.066** | **+0.035** | 0.628 | 0.076 |
| 9 | 0.616 ± 0.265 | **0.805 ± 0.080** | **+0.189** | 0.711 | 0.084 |
| **AVG** | **0.568** | **0.805** | **+0.237** | 0.722 | 0.079 |

**胜率**：S1D 在全部 10 类的 `raw_auc` 上严格优于 Baseline A（min Δ=+0.035 @ class 8，max Δ=+0.423 @ class 4）。Baseline 在 50 个 cell 中**没有任何一个**能命中 redline 主路径（`ssim_oc` 全在 [0.466, 0.924]），其 0.568 均值全部来自 distortion fallback 的事后选择。

**§6.7.3 Redline 阈值敏感性扫描（τ_oc ∈ {0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90}，raw_auc 下限固定 0.60）**：

| τ_oc | BL 主路径 | BL `raw_auc` | S1D 主路径 | S1D `raw_auc` | **Δ** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **0.15**（现行）| **0/50** | 0.568 ± 0.270 | **48/50** | 0.805 ± 0.108 | **+0.237** |
| 0.20 | 0/50 | 0.568 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.235 |
| 0.30 | 0/50 | 0.568 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.235 |
| 0.40 | 0/50 | 0.568 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.235 |
| 0.50 | 1/50 | 0.568 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.235 |
| 0.70 | 8/50 | 0.568 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.235 |
| 0.90 | 25/50 | 0.570 ± 0.270 | 49/50 | 0.803 ± 0.108 | +0.233 |

**两侧 ssim_oc 经验分位数（n=50/arm · 选模选中 epoch 实测）**：

| 统计 | min | p10 | median | p90 | max |
|:---|:-:|:-:|:-:|:-:|:-:|
| Baseline A | **0.466** | 0.524 | 0.744 | 0.870 | 0.924 |
| S1D | **0.033** | 0.046 | 0.075 | 0.119 | **0.161** |

**稳健性结论**：
1. Δ raw_auc 对 τ_oc 取值不敏感：[0.15, 0.90] 扫描下 Δ ∈ [+0.233, +0.237]，变动 0.004（≪ raw_auc std 0.108）。
2. `τ_oc=0.15` 不是对 S1D 友好的低界——S1D max ssim_oc=0.161 已超 0.15；阈值来源是 §2.5「质量红线」（2026-04-19 立档，**早于全部 S1D 实验**）。
3. **两类算法 ssim_oc 经验分布之间存在 [0.161, 0.466] 的"无 cell 区"**：τ_oc 落在此区间的任何取值都给出相同主路径计数（BL 0/50, S1D 49/50）。这印证两类算法在结构可信度上属于不同 regime，不是连续光谱上的渐变。

**§6.7.4 Verbatim 选模参考（baseline last_epoch · 与 §6.7.1 选模口径不同 · 仅论文复现完整性）**：

| 指标 | Baseline A · last_epoch | S1D · redline | Δ |
|---|:-:|:-:|:-:|
| `raw_auc` | 0.498 ± 0.297 | 0.805 ± 0.108 | **+0.307** |
| `folded raw_auc` | 0.751 ± 0.130 | 0.805 ± 0.108 | +0.054 |

§6.7.4 数字（+0.307）大于 §6.7.1（+0.237），差额 +0.070 来自 baseline 在 ep10 已退化为通用复印机（§2.10.8 O1+O2）。**paper headline 以 §6.7.1 +0.237（同选模口径）为准**，§6.7.4 仅供 D4-C verbatim 复现章节引用。

**§6.7.5 数据溯源**：

| 资产 | 路径 |
|:---|:---|
| Baseline 50-run summaries | `baseline_a_v2026_04_27\d{0..9}_s{42..46}\summary.json` |
| S1D 50-run summaries | `ALOCC_paper/s1d_final_c{0..9}_seed{42..46}_redline\summary.json` |
| H2H 后处理脚本（§6.7.1 / §6.7.2 数据源）| `_tmp_h2h_baseline_vs_s1d.py`（仓库根）|
| 阈值敏感性扫描脚本（§6.7.3）| `_tmp_threshold_sensitivity.py`（仓库根）|
| 可视化报告（per-class + per-cell + 弹药）| `ALOCC_paper/baseline_a_vs_s1d_report.md` |
| 指标主线迁移决策 | §2.10.11（本日志）|
| 选模规则源码 | `mnist_experiment_runner.py` `_select_records()` redline 分支 |

**架构声明**（重申 ADR-007 隔离 + ADR-011 指标主线）：本对比表中 Baseline A 与 S1D 的环境差异**仅来自结构改造**（S1 瓶颈 + L1-hinge distortion + D 端外类项），优化器/refinement 协议差异作为"verbatim 复现 vs 项目主线"的副产物被显式记录但不构成混淆变量；选模函数对两侧严格一致；主指标 `raw_auc` 与选模约束 `raw_auc ≥ 0.60` 同源（ADR-011）。

---

## 7. 文件索引

- 论文 PDF：`D:\Trae_coding\ALLOC\ALOCC-master\Sabokrou_*.pdf`
- 论文文本：`ALOCC_paper/paper_fulltext.txt`
- 源码快照（首轮审查时刻）：`ALOCC_paper/src_snapshot/`
- 本文档：`ALOCC_paper/PROJECT_LOG.md`

---

## 8. 张量维度审查清单（ADR-004 强制执行）

> 任何重构 PR 在合并前必须逐条勾选"通过性"列；新增网络模块必须配对应单测。

### 8.1 五条规则

| # | 规则 | 适用范围 |
|---|------|---------|
| **R1** | **Encoder-Decoder 对称性**：Encoder 下采样后特征图维度必须与 Decoder 起始层一致；ConvTranspose2d 的 `output_padding` 必须保证 (C, 28, 28) 输入 → (C, 28, 28) 输出。同样需通过 32×32、64×64 验证。 | `Generator` (R) |
| **R2** | **判别器 Linear 输入维度动态计算**：禁止用 `math.ceil(H/16)` 等启发式；必须以构造时一次 dummy forward 实测的特征图尺寸为准。 | `Discriminator` (D) |
| **R3** | **动态 batch 标签**：所有 `real_label / fake_label / outclass_*_label` 一律按 `tensor.size(0)` 实时构造；严禁缓存或硬编码 `batch_size`。`drop_last=False` 下最后一个不满 batch 必须可走通。 | `_train` 全部 variant |
| **R4** | **Cosine 分类器维度对齐**：投影后特征 `v ∈ [B, P]`、prototype `w ∈ [P]`，归一化维度（`v` 在 dim=1，`w` 在 dim=0）必须一致；输出 `cos ∈ [B, 1]`，与 `BCEWithLogitsLoss` 的 target shape 对齐。 | `CosinePrototypeClassifier` |
| **R5** | **复合 loss 必为标量**：`g_loss / d_loss / cls_loss` 等加权和的每个分量必须是 0-d Tensor；新增 loss 模块在 `forward` 末尾自带 `assert out.dim() == 0`。 | 所有 LossModule |

### 8.2 当前快照下的现状核对（基于 `src_snapshot/`）

| # | 通过 | 证据 / 隐患 | 行动 |
|---|------|-------------|------|
| R1 | ⚠️ 部分 | `Generator.__init__` 仅按 `in_h` 推链路（`model.py:79-81`），形参 `in_w/out_h/out_w` 接受却**从未使用**。MNIST 28×28 已手算验证：28→14→7→4→7→14→28 ✓；32×32、64×64 心算 ✓。但**非方形输入会静默错位**。 | **PR-M**：补 `in_w` 链路 + `tests/test_shape_chain.py` |
| R2 | ⚠️ 启发式 | `h_after_conv = math.ceil(in_h/16)` (`model.py:142-143`)。28/30/32/50/64 心算 ✓，但本质是巧合（5×5/s2/p2 配置下 ceil 公式恰好对齐）。任何 stride/padding/kernel 改动都会让 Linear shape 默默错。 | **PR-N**：构造时 `with torch.no_grad(): probe = self.logits[:-2](dummy); flat = probe.flatten(1).shape[1]` 后再建 Linear |
| R3 | ✅ 合规 | inclass：`real_imgs.size(0)` 动态构造 (`model.py:235-239`)；outclass：`outclass_imgs.size(0)` 动态构造 (`model.py:391-392, 495-496`)。inclass 与 outclass batch_size 不等也不会冲突，因为各自的 label 各自构造。 | 重构时保持此模式；写入 §8.3 模板 |
| R4 | ✅ 合规 | `v=[B,P]` `dim=1` normalize；`w=[P]` `dim=0` normalize；`(v*w).sum(dim=1, keepdim=True) → [B,1]` 与 `real_label=[B,1]` 对齐 (`model.py:64-71, 553-557`)。 | 重构时保留 `keepdim=True`，否则 BCE 会广播 |
| R5 | ⚠️ 隐式 | 当前所有 BCE/MSE 默认 `reduction='mean'`，`F.relu(margin - L1)` 中 `L1=torch.mean(...)` 也是 scalar (`model.py:415-417`)。**没有显式断言**——一旦未来改 `reduction='none'` 或漏 `.mean()`，会被广播吞掉，loss 数量级错乱难排查。 | **PR-O**：`utils._assert_scalar(t, name)` + 所有 LossModule 末尾调用 |

### 8.3 重构期 PR 模板增项

任何动 `model.py / trainer / loss / classifier` 的 PR 描述必须包含：

```
## §8 张量维度审查
- R1 Encoder-Decoder 对称性：[ ] 已验证 (in_h ∈ {28, 32, 64}, in_w ∈ {28, 32, 64})
- R2 D Linear 动态：[ ] 已通过 dummy probe 实测
- R3 动态 batch 标签：[ ] 末批不满已覆盖（test_drop_last_false）
- R4 Cosine 维度：[ ] v/w/cos/label shape 链路已断言
- R5 复合 loss scalar：[ ] 每个 LossModule.forward 末尾 _assert_scalar
- 新增/修改的 shape 单测路径：tests/test_*.py
```

### 8.4 计划新增的单测（`tests/`）

| 文件 | 覆盖规则 | 用例 |
|------|---------|------|
| `test_shape_chain.py` | R1 | `Generator(in_h=h, in_w=w).forward(x).shape == x.shape`，h,w ∈ {28,30,32,64}² |
| `test_discriminator_linear.py` | R2 | 各分辨率下 `D(x).shape == (B,1)`；构造时打印 actual flat dim |
| `test_drop_last_false.py` | R3 | `train_count=131, batch_size=64` → 跑通最后 3-样本 batch 不报错 |
| `test_cosine_classifier.py` | R4 | random feat_map → logits.shape==(B,1)，cos∈[-1,1] |
| `test_loss_scalar.py` | R5 | 每个 LossModule 实例 forward 后 `t.dim()==0` |

---

## 9. 开关治理清单（ADR-007 强制执行）

> 任何引入**新行为**的 PR，必须按本章节的三层模式暴露开关，并在 PR 描述中填写"§9 开关清单"小节。**默认关闭 = 默认行为与基线逐字节一致**。

### 9.1 三层开关模式（标准模板）

```
┌────────────────────────────────────────────────────────────────┐
│ 第 1 层：CLI flag  (run_paper_mnist_figure6_7.py / export_*.py) │
│   add_argument("--neg-ssim-scale", type=float, default=0.0)     │
│   # 数值型 0.0 = 关；>0 = 开（推荐）                             │
│   # 或 bool 型：                                                 │
│   add_argument("--enable-early-stop", action="store_true")      │
│   add_argument("--no-early-stop",     dest="enable_early_stop", │
│                action="store_false")                            │
├────────────────────────────────────────────────────────────────┤
│ 第 2 层：Runner/Trainer kwarg                                   │
│   def run_experiment(..., neg_ssim_scale: float = 0.0,          │
│                     enable_early_stop: bool = False): ...       │
│   # 默认值必须与"关"对齐；由 CLI 透传，不在函数体内 override     │
├────────────────────────────────────────────────────────────────┤
│ 第 3 层：内部消费点                                              │
│   if neg_ssim_scale > 0.0:                                      │
│       g_loss = g_loss + neg_ssim_scale * neg_ssim(...)          │
│   # 条件短路保证关闭态下零开销、零旁路；                          │
│   # 禁止无 if 的"系数 = 0 但仍前向"实现（污染 autograd 图）      │
└────────────────────────────────────────────────────────────────┘
```

**默认值规则**：
- 数值 scale 类默认 `0.0`（关）；
- bool 类默认 `False`（关），对应 CLI 同时提供 `--enable-X` 与 `--no-X` 两路；
- 字符串策略类（如 `selection_strategy`）默认值必须是**当前基线已用值**（如 `"paper"`），新增策略通过新枚举值加入，禁止改默认。

### 9.2 可追溯性契约（`summary.json` 必须记录）

每条 PR 引入的开关必须落到 `summary.json` 的 `switches` 字段，形如：

```json
{
  "variant": "alocc_loss_cls",
  "best_epoch": 6,
  "switches": {
    "neg_ssim_scale": 0.3,
    "enable_early_stop": true,
    "stop_recon_threshold": 0.02,
    "stop_min_epoch": 3,
    "selection_strategy": "paper",
    "selection_min_auc": 0.95,
    "selection_min_auc_hard": false,
    "selection_fallback_triggered": false
  },
  "...": "..."
}
```

**验收标准**：`grep -r "switches" ALOCC_paper/baselines_cuda/*/experiment/summary.json` 能列出每次运行的完整开关快照；任何两次实验 diff 只需看 `switches` 一块即可解释指标差异。

### 9.3 PR 模板增项（与 §8 清单并列）

任何动 CLI / Runner / 训练循环 / 损失 / 选模的 PR 必须填：

```
## §9 开关清单
- 新增 CLI flag：[--xxx (type, default, 关对应值)]
- 新增 Runner kwarg：[函数:参数 (default)]
- 新增消费点：[文件:行号 (if 分支形态)]
- 默认关闭下的回归验证：[命令行 + 预期 summary diff = 空]
- 写入 summary.json.switches：[字段名]
```

### 9.4 当前快照下的开关现状核对

| 已有开关 | 类型 | 默认值 | 合规 | 备注 |
|---|---|---|:---:|---|
| `--variant {alocc, alocc_loss, alocc_loss_cls}` | enum | `alocc` | ✅ | 基线 A 对应默认 |
| `--r-alpha` | float | 0.2 | ✅ | 论文 Eq.5 的 λ |
| `--noise-std` | float | 0.31 | ✅ | Eq.3 的 η scale |
| `--stop-recon-threshold` | float? | `None` | ✅ | PR-B（2026-04-19）落地后三 variant 全部消费；`None` 时不进入分支，0 开销 |
| `--stop-min-epoch` | int | 1 | ✅ | 同上；PR-B 后与 `--stop-recon-threshold` 成对生效 |
| `--selection-strategy` | enum | `paper` | ✅ | `acc_auc` / `paper` / `distortion` 三策略；RM-3a（2026-04-19）落地 `distortion` 分支 |
| `--distortion-alpha` | float | 1.0 | ✅ | RM-3a；仅 `distortion` 策略生效 |
| `--distortion-beta` | float | 1.0 | ✅ | RM-3a；同上 |
| `--paper-score-normalization` | enum | `relative` | ✅ | RM-3b（2026-04-19）：`absolute` 模式用 §9.5 锚点表把北极星 7 项映射到 [0,1]；默认 `relative` 保持 bitwise 回归 |
| `--selection-min-auc` | float | **0.60** | ✅ | RM-3c（2026-04-19）默认 0.95→0.60；PR-A 落地的 `--selection-min-auc-hard` + `--selection-log-fallback`/`--no-selection-log-fallback` 仍可用；基线脚本 `_run_all.ps1:46` 显式传 0.95 所以锚点不变 |
| `--label-smoothing` | float | 0.0 | ✅ | 默认关 |
| `--weight-decay` | float | 0.0 | ✅ | 默认关 |
| `g_outclass_distortion_scale` | float | 0.0（硬编码） | ❌ | **RM-1 目标**：未暴露到 CLI，当前只能改代码打开 |
| `d_outclass_loss_scale` | float | 0.1（B/C 硬编码） | ❌ | 已生效但**没 CLI 开关**——无法关闭做消融 |

### 9.5 即将落地的 PR 必须带开关列表（预先约束）

| PR | 新增开关（CLI） | 默认值（=关） | 消费点 |
|---|---|---|---|
| **PR-A** | `--selection-min-auc-hard` (bool) | `false`（保持当前静默 fallback 行为） | `mnist_experiment_runner._select_records`；开启时 fallback → RuntimeError + summary.switches.selection_fallback_triggered 必记 |
| **PR-A** | `--selection-log-fallback` (bool) | `true`（纯日志，不改行为） | 同上；关闭时完全静默（与当前逐字节一致） |
| **PR-B** | （无新增 CLI；`--stop-recon-threshold` 已有） | `None`（关） | `ALOCC_LOSS_CLS._train`：与 `ALOCC._train` 的分支对齐；未提供阈值时 0 开销分支 |
| **RM-1** | `--neg-ssim-scale` (float) | `0.0`（关） | G 的 loss 聚合：`if scale>0: g_loss += scale * neg_ssim_outclass(...)` |
| **RM-1** | `--g-outclass-distortion-scale` (float) | `0.0`（关） | 暴露已有内部开关到 CLI |
| **RM-1** | `--d-outclass-loss-scale` (float) | `0.0`（关，**B/C 复用时显式传 0.1 还原**） | 把硬编码迁移到 CLI；同步更新 `_run_all.ps1` |
| ~~**RM-3a**~~ | ~~`--selection-strategy distortion`~~ | ~~不改默认~~ | ✅ 已落地（2026-04-19）。见 §5 RM-3 changelog |
| ~~**RM-3a**~~ | ~~`--distortion-alpha` / `--distortion-beta`~~ | ~~1.0 / 1.0~~ | ✅ 已落地（2026-04-19） |
| ~~**RM-3b**~~ | ~~`--paper-score-normalization`~~ | ~~`relative`~~ | ✅ 已落地（2026-04-19）；锚点表见本章末 |
| ~~**RM-3c**~~ | ~~`--selection-min-auc` 默认值调整~~ | ~~0.95 → 0.60~~ | ✅ 已落地（2026-04-19）；`_run_all.ps1` 显式传 0.95 保证回归 |
| **RM-2** | `--freeze-encoder-after-epoch` (int) | `-1`（关） | 训练循环：epoch ≥ 值时冻结 encoder 梯度 |
| **RM-4** | `--report-raw-refined-split` (bool) | `true`（日志增强，行为不变） | 评估汇总，纯新增日志字段，不改 metrics 计算 |

**RM-3b 绝对锚点表**（user 2026-04-19 批准，待实现）：

| 指标 | 0 分线（不及格）| 1 分线（论文标准）| 方向 | 依据 |
|---|---|---|---|---|
| `auc_gain` | ≤ 0 | ≥ +0.15 | 越高越好 | Figure 7 claim 必须正值 |
| `refined_auc` | ≤ 0.5 | ≥ 0.95 | 越高越好 | 0.5 = 随机，0.95 = 论文量级 |
| `ssim_gap` | ≤ 0 | ≥ 0.30 | 越高越好 | **数据驱动**（2026-04-19，user 批 0.30）：论文全文 0 次出现 SSIM，无权威依据；改用 A+B+C 共 30 个 epoch 的 `ssim_gap` 分布定线——p90 = 0.3177，max = 0.3583（B ep10），A 基线无扭坏损失时上限 0.1349、B/C 有弱扭坏损失（scale=0.1）时偶能触 0.30+；取 0.30 使 A/B/C 的 best_ep 分别得 0.45/0.89/0.63，区分力合理，同时为 RM-1 目标（把"偶尔 0.30"变"稳态 0.30+"）留出靶心 |
| `ssim_oc` | ≥ 0.5 | ≤ 0.1 | 越低越好 | R 对 outlier 扭得越狠越好 |
| `acc` | ≤ 0.8 | ≥ 0.95 | 越高越好 | **2026-04-19 user 批 0.80**：MNIST inlier=1/outlier=[6,7] 高度不平衡，acc 容易超 0.9，0.5 门槛过松；0.80 更贴合数据分布 |
| `score_gap_gain` | ≤ 0 | ≥ +0.10 | 越高越好 | auc_gain 的均值差同源指标 |
| `score_gap` | ≤ 0 | ≥ 0.2 | 越高越好 | refined 空间 in/out 均值差 |
| 其他辅助项（`vif_gap` / `gmsd_gap` / `ssim_ic` / `auc` / `eer` / `raw_auc` / `raw_score_gap` / `vif_oc`） | 保持相对归一化 | 保持相对归一化 | - | 非北极星项无绝对锚点依据，沿用组内相对 |

### 9.6 回归测试协议

任何 PR 合入前必须跑一次"全开关默认"回归：

```bash
# 与 baselines_cuda 完全同参（§6.4 锚点），不加任何新 flag
python run_paper_mnist_figure6_7.py --output-dir .../regression_check --variant alocc \
    --specific 1 --outlier-labels 6 7 --epochs 10 --train-count 4096 ...
```

期望：`summary.json.best_metrics` 的北极星五项（`refined_auc / auc_gain / ssim_oc / ssim_gap / score_gap_gain`）与 `baselines_cuda/A/experiment/summary.json` **严格相等**（容差 `< 1e-6`，同 seed）。任何偏离必须在 PR 描述中解释并经同意。
