# JEPA-ARKit 运行结果与高难度排查记录（2026-08-11 ~ 08-13）

> 本报告只收录**有研究价值的真实数据结果**与**高难度排查过程**。合成冒烟、基础设施"能跑通"级别的验证一律不收录（见第 5 节），避免低价值信息污染上下文。
> 所有数字来自 `runs/`、`artifacts/` 与执行会话，路径见第 6 节。

---

## 1. 结论速览

| 结论 | 依据 |
|---|---|
| **E01（直接因果回归）是当前唯一推荐**：确定性离线 + 流式都用它 | 全容量 2e-4 模型验证集 mouth MAE 最低、音频干预最强 |
| **E03 是可选离线风格层**，感知多样性待 UE/人工验证 | 三 seed 嘴部采样严格不变、E01 均值逐 seed 完全保持 |
| **E11 不推荐**：纯音频下预训练与随机初始化打平 | audio-only test mouth MAE 0.04024 vs 0.04020 |
| **E10 训练链路通过但表征结论未解锁** | 有效秩比 0.284 超 25% 门槛，但 E10 probe 仍标 blocked |
| 已建成可复现真实链路 | 24 演员 / 1,440 片段 / 身份不相交 840/300/300 / 确定性评测 / 流式 reference / 60fps 导出 |

---

## 2. 有价值的运行结果（真实数据）

### 2.1 数据与标签管线

- **RAVDESS 下载**：24 个官方压缩包共 `13,357,344,723` 字节（约 12.4 GiB），**24/24 官方 MD5 全部通过**（Zenodo DOI 10.5281/zenodo.1188976）。
- **MediaPipe 规范动作求解**：1,440/1,440 片段 **0 失败**，平均逐帧有效人脸率 **100%**，全量耗时约 **12.3 分钟**。
  - 关键策略：`tongueOut` 是 MediaPipe 不可观测维度，采用**版本化缺失曲线策略**显式标记降级，而不是静默补零（见 3.6）。
- **Pilot release（D0P）**：24 身份，1.484 小时，**840 train / 300 validation / 300 test 按演员完全不相交**；全文件审计通过，唯一警告是"单一数据源"。
- **WavLM v2 特征**：768 维 FP16，266,110 帧，卷积感受野**精确中心对齐**（首帧 12.46875 ms、步长 20 ms），模型 revision 冻结为 `efa81aae7ff777e464159e0f877d54eac5b84f81`。

### 2.2 E01 直接回归（当前推荐模型）

**模型选择**（验证集 physical mouth MAE，`artifacts/e01_model_selection.json`）：

| 候选 | validation mouth MAE | curve MAE | 500ms shift ratio | silence ratio | 结论 |
|---|---|---|---|---|---|
| **e01_ravdess_v2（256 维 6 层, 2e-4）** | **0.04938** | 0.06621 | 1.127 | 1.307 | **选中** |
| e01_ravdess_v2_small | 0.05046 | 0.06581 | 1.080 | 1.115 | 略小但嘴部略差 |
| e01_ravdess_v2_mouth2x | 0.04969 | 0.06696 | 1.107 | 1.223 | 嘴部 2 倍加权被否 |
| e01_ravdess_v2_lr1e4 | 0.05066 | 0.06710 | 1.095 | 1.212 | 学习率过低 |

**测试集（test, 377 窗口）**：mouth MAE **0.04164**、curve MAE **0.06571**、500 ms 音频移位比 **1.124**、静音比 **1.146**、音频打乱比 1.171、音频反转比 1.138。

**跨身份差异（重要发现）**：不能只看总体 MAE——

| 身份 | mouth MAE |
|---|---|
| Actor 23 | **0.0272**（最好） |
| Actor 22 | 0.0344 |
| Actor 21 | 0.0417 |
| Actor 24 | 0.0478 |
| Actor 20 | **0.0532**（最差） |

→ 最差样本集中在 Actor 20/24，跨身份差异显著，模型存在身份泛化短板。

### 2.3 E03 区域条件残差（可选风格层）

**三 seed 汇总**（`artifacts/e03_regional_three_seed_summary.json`）：

- 参数预算：仅比 E01 多 **4.51%**（parameter_ratio 1.045）。
- **嘴部采样最大绝对差 = 0**（三 seed 全部），即随机采样**严格不改变嘴部**。
- E01 均值**逐 seed 完全保持**（baseline_mean_exactly_preserved_across_seeds = true）。
- 随机性只作用于眼眉/视线/头部：test head variance 0.0074–0.0223，stochastic variance 0.0026–0.0066，非头部方差 ≈0.0006–0.0009。
- 选定 **seed2**（按验证集中位随机方差选取，未使用测试集）。
- 导出不变量（`e03_regional_export_invariants.json`）：**同 seed 两次导出 SHA-256 完全一致**（确定性可复现）；跨 seed 嘴部/鼻颊差异 = 0，眼眉均差 0.021、头部平移均差 0.0436。
- 结论：架构通过、感知多样性未证实（gate: `architecture_passed_perceptual_diversity_pending`）。

### 2.4 E10 / E11（负面结果，但有明确信息量）

- **E10 Motion-JEPA**：causal validation loss 0.237；latent effective-rank ratio **0.284**（causal）/ 0.296（random-span），physical mouth MAE 0.0226（causal）。训练链路通过；但 E10 probe（冻结编码器 + 线性头）未完成，表征结论仍标 `blocked`。注意：smoke 阶段该指标仅 8.79% 被正确阻断，正式长训后才到 0.284——说明**必须用正式长训评估表示质量，smoke 不能代表结论**。
- **E11 Audio-Motion-JEPA**：teacher-forced（有历史）validation mouth MAE 0.02263 确实低于 E01，但**纯音频（无历史）test mouth MAE 0.04024 与随机初始化 0.04020 打平** → 预训练没有带来纯音频部署增益，不晋级。

### 2.5 流式与导出（工程价值高）

- **流式 reference**（36.13 s 真实拼接轨迹）：1,084 帧全部连续、217 个重叠 chunk、80 ms look-ahead、1 s 历史、**42.5× realtime**（RTX 3060 Ti）。
- **chunk 边界连续性**：固定 120 帧窗口 + "有效上下文右对齐 + 左侧 neutral padding"后，边界中位跳变从 **5.9× 降到 1.174×**（修复前位置编码随窗口长度漂移，见 3.9）。
- **30→60 fps 导出**：曲线/平移线性插值 + 头部四元数最短路径 SLERP；源帧 round-trip 误差 **0**、四元数误差 **< 6e-8**（UE 端独立参考实现）。

### 2.6 计算环境

- RTX 3060 Ti 8 GB，torch 2.13.0+cu132；E01 峰值显存仅 **163 MB**、训练吞吐 12.2 updates/s——单卡 8GB 即可完整跑通，符合 24 GB 上限约束。

---

## 3. 高难度排查记录（现象 → 根因 → 修复 → 教训）

### 3.1 E01 测试指标整体失效：评估器漏调 `model.eval()`
- 现象：E01 与 E03 的 mouth MAE 有 0.00014 的"口径差"，无法解释。
- 根因：E01 评估器加载 checkpoint 后**没有调用 `model.eval()`**，dropout 在测试时仍开启（随机性污染）。
- 修复：补 `model.eval()` 并重算 E01；**旧 E01 测试指标全部作废**，不得继续引用。
- 教训：任何"同口径比较"前先检查 eval 模式；测试期 dropout 会让结果偏乐观且不可复现。

### 3.2 E03 冻结分支被 `model.train()` 打开 dropout
- 现象：E03 的确定性 E01 分支在训练时被破坏。
- 根因：`model.train()` 递归打开所有子模块的 dropout，包括应冻结的分支；且 checkpoint 未记录参数预算比。
- 修复：显式冻结分支全程确定性，参数预算比写入 checkpoint；评估把冻结分支作为严格 eval 子模块。
- 教训：冻结权重 ≠ 冻结行为，需同时固定 `requires_grad=False` 与 eval 状态。

### 3.3 流式 chunk 边界跳变：位置编码随窗口长度漂移（5.9× → 1.174×）
- 现象：真实流式轨迹在 chunk 切换处出现明显跳变（中位约内部步长 5.9 倍）。
- 根因：每个 chunk 以**不同长度**调用 Transformer，位置编码随窗口长度变化，同一时间点在不同 chunk 里坐标不一致。
- 修复：固定 120 帧窗口，"有效上下文右对齐、左侧 neutral padding"，让每个输出块落在一致的模型位置坐标上；并写进协议契约（含 1 s 历史、80 ms look-ahead）。
- 教训：padding 方向 + 窗口长度必须固定，否则流式 chunk 边界不连续；这是流式部署最容易踩的隐性坑。

### 3.4 JEPA 表示质量：effective-rank 硬门（smoke 8.79% 阻断）
- 现象：Motion-JEPA 训练 loss 持续下降，但 latent 有效秩比例仅 8.79%。
- 根因：有效秩过低意味着表示退化为低维/塌缩，loss 下降不能代表学到了好表示。
- 修复：把 effective-rank 比例 ≥25% 设为硬门，失败以 `blocked` 明确暴露（退出码 2），不把"能训练"误报为"JEPA 有效"；正式长训后该指标到 0.284。
- 教训：自监督表示任务必须同时盯 loss 与表示质量指标，且**smoke 结论不代表正式训练结论**。

### 3.5 E10 任务定义错误：同帧随机遮罩 ≠ causal future masking
- 现象：Motion-JEPA 训练的是同帧随机遮罩恢复，不能据此判断"只学插值"。
- 根因：实现未区分 mask mode。
- 修复：增加显式 `mask mode` 与 `horizon`；smoke 默认随机遮罩，正式 E10 用 `causal_future`，两种任务结果分开记录。
- 教训：研究设计级参数要在代码里显式化，否则实现会悄悄偏离实验意图。

### 3.6 MediaPipe 缺 `tongueOut`：标签空间差一条曲线
- 现象：真实视频求解 100% 成功，但保存被数据契约拦截。
- 根因：MediaPipe 不输出 ARKit 的 `tongueOut`（不可观测维度）。
- 修复：默认**严格阻断**；只有显式传 `--missing-curve-policy` 才允许保存，并把策略 ID/哈希/退化曲线写进 NPZ，将 `tongueOut` 标记为不可观测维度而非静默补零。
- 教训：伪标签管线的"缺失维度"必须显式登记，否则会把零值误当真实人工标签。

### 3.7 训练量纲失衡：嘴形结论会被头部距离淹没
- 现象：52 条 `[0,1]` 曲线、四元数、约 −27 cm 头部平移混用一个未归一化损失。
- 根因：不同量纲直接相加 → 模型优先拟合头部距离，嘴形训练结论失真。
- 修复：按 `motion_normalization.json` 归一化后训练。
- 教训：混合量纲损失前必须先归一化，否则指标好看但子区域失真。

### 3.8 D0P 审计错误假设："同一数据集不能跨 split"
- 现象：审计器把"同一数据源分布到 train/val/test"判为泄漏。
- 根因：把 source 与 identity 混淆。
- 修复：真正必须不相交的是**身份/脸部身份与撤回键**；同一授权数据源在身份不相交后可分布到三个 split。
- 教训：泄漏检查要看身份/样本级，而不是数据源级。

### 3.9 E03 实现级踩坑三连
- `nn.Module` 自带 `_parameters` 字典：自定义"参数拆分"同名导致前向直接失败 → 改名。
- 20-step GPU smoke 抓出**区域损失把 batch 级质量权重当维度向量**（嘴部 29 维与 B×29 展平形状冲突）→ 统一 batch 平均。
- 原型多堆 4 层 Transformer 违反参数预算 ±20% 约束，且验证用 posterior 存在**目标信息泄漏** → 收紧为"共享 E01 隐状态 + 小型 GRU 随机头"，评估只用 audio prior。
- 教训：先单测 + 小步 smoke 再长训；架构必须满足预注册的可比性约束。

### 3.10 E03 8GB 显存 OOM（诊断脚本）
- 现象：诊断脚本因保留 8 次采样的计算图触发 8 GB 显存上限。
- 根因：采样计算图未释放。
- 修复：改用"非头部/头部分离温度"（对方差是精确平方缩放、可复现），不靠挑 seed；训练/checkpoint 本身无问题。
- 教训：诊断脚本也要管内存，采样图要 detach/释放。

### 3.11 CREMA-D Git LFS 系列坑（第二来源候选）
- 现象：git clone 超时被工具终止；拉完才发现 22,326 个"媒体文件"都是 130 字节左右的 **LFS 指针**，不是真实数据。
- 根因/处理：`GIT_LFS_SKIP_SMUDGE=1` 分离元数据与媒体再单独 LFS 拉取；`AudioMP3` 是有损重复副本（跳过省空间）；GitLab 逐个对象协商导致吞吐低；Windows 无所有权盘触发 `safe.directory` 保护 → 逐命令显式可信目录而非全局修改；最终不只看出错码，还做对象完整性与 WAV/FLV 可解码性二次校验。
- 教训：git LFS 仓库"克隆成功 ≠ 数据到手"，必须核对指针是否物化 + 文件可解码。

### 3.12 BIWI 下载地址误导
- 现象：所谓 BIWI 下载地址实际指向 Kinect 头姿数据库（非带音频的 B3D(AC)²）。
- 处理：拒绝把错误数据当已下载；改用 Zenodo 官方 RAVDESS。
- 教训：下载前核对内容与权利，不将错就错；错误数据混入会污染整条管线。

### 3.13 环境与依赖排查
- **uv 默认解析到 CPU 版 PyTorch** → 先 CPU 验证正确性，再单独配置 CUDA wheel 源，不把环境问题与代码问题混在一起。
- **WavLM revision 漂移**：不能用 `main` 作实验指纹，冻结 `efa81aae...`；transformers 5.x API 变化时固定兼容版本；首次导入超时≠模型不可用，拆开定位。
- **公开搜索索引错误命中**：不可靠页面不写进计划，改从 Epic 官方文档/UE 源码 RigLogic 接口推导集成边界。

---

## 4. 已被诚实标注的边界（勿误用为结果）

- E00/D0B Gold：需 200 片段人工审计 + 双标注 + 仲裁 + UE 渲染验证（未做）。
- UE 5.6/MetaHuman：本机无工程，A0/E20 blocked。
- 产品轨：无自录授权数据，所有 checkpoint 均 `research_only` / `pilot_non_comparable`。
- RAVDESS 为 CC BY-NC-SA：研究可用，不能初始化商业产品模型。

---

## 5. 明确排除的"垃圾结果"（不入报告）

| 项目 | 为什么排除 |
|---|---|
| t0_direct_smoke / t0_jepa_smoke | 40 步合成数据冒烟，只是"能跑通"基础设施验证 |
| e03_regional_ravdess_smoke | 20 步 GPU smoke |
| d0a_demo_audit / 合成 fixture | 合成数据 25 条，无研究价值 |
| 任何 `synthetic_non_comparable` 产物 | 合成结果不可作为研究结论 |

这些文件保留在 `runs/`、`artifacts/` 供复现审计，但不应进入任何研究结论、论文或报告。

---

## 6. 原始数据位置

- 实验指标：`runs/<experiment>/metrics.json`、`evaluation*.json`、`per_clip_metrics_test.json`
- 决策与汇总：`artifacts/e01_model_selection.json`、`artifacts/e03_regional_three_seed_summary.json`、`artifacts/e03_regional_export_invariants.json`、`artifacts/unitalker_candidate_audit.json`
- 总状态：`artifacts/project_status.json`
- 完整执行叙述：`artifacts/research_execution_report.md`
