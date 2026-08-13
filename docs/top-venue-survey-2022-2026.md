# 语音驱动面部动画（Speech-Driven Facial Animation / Talking Head）顶会深度调研报告

> 调研日期：2026-08-13 ｜ 范围：CVPR / ICCV / ECCV / NeurIPS / ICLR / AAAI / IJCAI / ACM MM / SIGGRAPH 等，覆盖 2022–2026，另含 2025–2026 高价值 arXiv。
> 用途：为 JEPA-ARKit 项目的发表定位、方法对标、评测体系设计提供依据。所有论文的会议/年份均来自公开检索结果；仅 arXiv 的已标注。

---

## 0. 摘要

- 该领域在顶会是**稳定高热赛道**：CVPR/ICCV/ECCV 每年相关论文约 20–40 篇（含 2D 说话人视频与 3D 面部动画两个子方向），另有大量论文落在 ACM MM、AAAI、IJCAI、SIGGRAPH。
- 三年主线：**Transformer/离散 token（2023）→ 扩散模型（2023–2025）→ 3D Gaussian Splatting / 视频 Diffusion Transformer（2025）→ 生产级部署与评测标准化（2025–2026）**。
- 2025–2026 热点：写实渲染（3DGS）、视频扩散说话人、实时流式、情感/风格/个性化可控、音频语义解耦（SSL 特征）、评测指标创新、UE/ARKit 生产部署、AV-JEPA 表示学习。
- 对本项目的直接警示：**「MediaPipe → ARKit blendshape → UE 部署」路线已被 3DMEAD-ARKit（arXiv 2606.10753，2026-06）抢先发表**；**「预训练 SSL 音频特征」的缺陷已被 Wav2Sem（CVPR 2025）专门解决**；**「实时流式」已有 Teller（CVPR 2025）**。

---

## 1. 方法脉络与分类体系

```
输入音频
  │
  ├─ 音频编码：DeepSpeech/HuBERT/WavLM（预训练 SSL 特征已成标配）
  │             └─ 2025 新问题：SSL 特征"近音词耦合"→ Wav2Sem 语义解耦
  │
  ├─ 时序建模：RNN/GRU → Transformer（FaceFormer）→ 离散 token+Transformer（CodeTalker）
  │             → 扩散（FaceDiffuser/DiffPoseTalk/FaceTalk）→ Mamba/SSM（ACTalker）
  │             → 自回归流式（Teller）→ 视频 DiT（Hallo3/EmotiveTalk）
  │
  ├─ 可控性：情感（EmoTalk/EcoFace/EmoRLTalk）、风格（DiffPoseTalk/MemoryTalker/PTalker）、
  │            个性化（Imitator/MemoryTalker）、多模态（AnyTalk/ACTalker/MEDTalk）
  │
  ├─ 输出空间：3D 网格/FLAME（VOCA 系）→ ARKit/Blendshape（生产兼容，3DMEAD-ARKit）
  │             → NeRF/3DGS 写实渲染（S3D-NeRF/FaceTalk/GaussianSpeech/MGGTalk）
  │             → 2D 视频像素（SadTalker/Hallo/EchoMimic/IM-Portrait）
  │
  └─ 评测：顶点误差(LVE)/唇误差/多样性 → 感知指标 + 物理同步指标（CVPR 2025）
            → 用户研究 + 商业工具对比（3DMEAD-ARKit）
```

---

## 2. 2022–2023：奠基期（Transformer、离散先验、扩散、情感）

### 2.1 2022

| 论文 | 会议 | 核心方法 | 数据 |
|---|---|---|---|
| FaceFormer | CVPR 2022 | 首个端到端 Transformer，自回归生成 3D 面部运动，注意力遮蔽与自动回归 | VOCASET |

### 2.2 2023

| 论文 | 会议 | 核心方法 | 数据 | 意义 |
|---|---|---|---|---|
| CodeTalker | CVPR 2023 | 离散运动先验（VQ）+ Transformer，缓解回归平均化 | VOCASET | 离散 token 路线的代表 |
| SelfTalk | CVPR 2023 | 自监督预训练音频，解耦内容/风格 | VOCASET | 引入自监督音频 |
| ProbTalk3D | CVPR 2023 | 离散概率扩散，非确定性 + 情感可控 | MEAD | 概率生成 + 情感控制 |
| SadTalker | CVPR 2023 | 2D 说话人：从音频回归 3DMM 系数（姿态/表情）再 3D-aware 渲染 | 单图 + 音频 | 2D 视频方向标杆之一 |
| DiffPoseTalk | ICCV 2023 | 扩散 + 风格编码器（参考视频提取风格），生成风格化面部动画 + 头部姿态 | TFHP（自建） | 风格化扩散的代表，自建 TFHP 基准 |
| EmoTalk | ICCV 2023 | 情感解纠缠（内容/情感分离），情感可控生成 | MEAD | 情感控制代表 |
| Imitator | ICCV 2023 | 个性化：说话人无关内容 + 说话人特定身份嵌入 | 自定义 + 公共数据 | 个性化代表 |
| Semi-supervised 3D Facial Animation (Cross-modal Encoding) | ICCV 2023 | 跨模态编码的半监督学习 | — | 半监督方向 |
| FaceDiffuser | SIGGRAPH Asia 2023 | 扩散 + HuBERT 音频编码 + GRU 解码 | VOCASET/BIWI 等 | 扩散 3D 面部动画的早期代表（3DMEAD-ARKit 直接重训它） |

**2023 小结**：Transformer/离散先验/扩散三种范式成型；情感与风格控制成为卖点；VOCASET 与 MEAD 成为标准评测集。

---

## 3. 2024：扩散普及 + 基准化 + 渲染升级

| 论文 | 会议 | 核心方法 | 数据 |
|---|---|---|---|
| FaceTalk | CVPR 2024 | 音频驱动运动扩散，神经参数化头部模型 | 大规模自定义 |
| Probabilistic Speech-Driven 3D Facial Motion Synthesis: New Benchmarks, Methods and Applications | CVPR 2024 | 概率生成 + **新基准/新应用**（Stability/Visual Style/Emotional Style/Impulsiveness 四类） | 自建多来源 |
| Towards Variable and Coordinated Holistic Co-Speech Motion Generation | CVPR 2024 | 全身共语音动作的多样性与协调 | — |
| SyncTalk | CVPR 2024 | NeRF 说话人，"魔鬼在同步"，唇同步 + 姿态/表情协调 | 2D 说话人 |
| KMTalk | ECCV 2024 | 关键运动嵌入（key motion embedding） | VOCASET/BIWI |
| S3D-NeRF | ECCV 2024 | 单张图 + 语音 → 高保真 NeRF 说话头 | 2D 说话人 |
| ScanTalk | ECCV 2024 | 未注册扫描网格上的 3D 说话头 | 扫描数据 |
| Loc3Diff | ECCV 2024 | 局部扩散 3D 人头合成/编辑 | — |
| AnimateMe | ECCV 2024 | 4D 面部表情扩散 | — |
| AniPortrait | arXiv 2024 | 音频 → 参考人像 → 写实说话视频 | 2D |
| Hallo / EchoMimic / MuseTalk / V-Express | arXiv 2024（后续进 AAAI/ACM MM 等） | 扩散式 2D 说话人生成（DiT/UNet） | 大规模 2D 视频 |

**2024 小结**：扩散成为主流范式；出现"基准本身"型论文（Probabilistic 3D Facial Motion Synthesis）；NeRF/参数化渲染升级；2D 扩散说话人爆发（Hallo/EchoMimic/MuseTalk 等开源生态）。

---

## 4. 2025：3DGS + 视频 DiT + 可控性 + 评测标准化（竞争最激烈一年）

### 4.1 CVPR 2025

| 论文 | 核心方法 | 意义 |
|---|---|---|
| Wav2Sem | 即插即用音频**语义解耦**：针对 SSL 音频特征（WavLM 类）的近音词耦合 → 唇形平均化问题 | **与本项目 WavLM 路线直接相关**；给出 SSL 特征在本任务中的已知缺陷与修复 |
| Teller | **首个实时流式**自回归音频驱动人像动画 | **与本项目流式方向直接相关**；上了 CVPR 正会 |
| EmotiveTalk | 音频信息解耦 + 情感视频扩散 | 情感 + 扩散 |
| MGGTalk | 可泛化 3DGS 一次性说话头 | 3DGS 泛化 |
| TaoAvatar | 实时全身说话头像（3DGS + AR） | 3DGS + AR 部署 |
| Perceptually Accurate 3D Talking Head | **新定义 + speech-mesh 表示 + 感知/物理同步评测指标** | **评测本身成为贡献**，标志评测标准化趋势 |
| IM-Portrait | 3D-aware 视频扩散，写实说话头 | 扩散 + 写实 |
| Hallo3 | 视频 Diffusion Transformer，高动态人像动画（120 小时数据） | 视频 DiT 代表 |

### 4.2 ICCV / ICLR / NeurIPS / AAAI / IJCAI / ACM MM 2025

| 论文 | 会议 | 核心方法 | 意义 |
|---|---|---|---|
| ACTalker | ICCV 2025 | 视频扩散 + 掩码选择性状态空间（Mamba），多信号/单信号控制 | 状态空间 + 扩散 |
| MemoryTalker | ICCV 2025 | 音频引导风格化 + 个性化（记忆机制） | 个性化风格代表 |
| GaussianSpeech | ICCV 2025 | 音频驱动个性化 3DGS 头像，实时写实 | 3DGS 头像代表 |
| Expressive Talking Human from Single-Image | ICCV 2025 | 用姿态引导视频扩散生成伪标签，单图表情说话人 | 伪标签范式（与本项目 Silver 标签思路呼应） |
| EcoFace | ICLR 2025 | 音视频情感共同解纠缠 | 情感解耦 + 多模态 |
| VASA-3D | NeurIPS 2025 | 单图音频驱动高斯头部头像 | 3DGS + 单图 |
| MegActor-Sigma | AAAI 2025 | 混合模态条件 DiT 人像动画 | DiT 控制 |
| AnyTalk (+AniTalk 数据集) | AAAI 2025 | 多模态多域说话人 + 新数据集 | 数据集贡献 |
| GLDiTalker | IJCAI 2025 | 图增强量化空间学习 + 时空潜扩散 | 唇同步 + 多样性 |
| Ditto | ACM MM 2025 | 运动空间扩散，可控实时说话头 | 实时可控 |
| PTalker | ACM MM 2025 / arXiv 2512.22602 | 风格解耦 + 模态对齐的个性化 3D 说话头 | 个性化 + 对齐 |
| PESTalk | ACM MM 2025 | 语音驱动 3D 面部动画风格 | 风格 |
| MEDTalk | arXiv 2025 | 多模态控制 + 动态情感解耦嵌入 | 多模态控制 |
| EmoRLTalk | ICME 2025 | 离线强化学习控制情感面部动画 | RL 控制（新范式） |
| Model See Model Do | arXiv 2025 | 风格控制，用 RAVDESS 作风格参考、CelebV-Text 作音频 | 跨域风格泛化 |

**2025 小结**：3DGS 与视频 DiT 成为渲染主线；"可控性"（情感/风格/个性化/多模态）是最大卖点；**评测指标创新成为独立贡献点**；实时流式出现正会论文（Teller）；RL 开始进入（EmoRLTalk）。

---

## 5. 2026 最新（arXiv / SIGGRAPH / AAAI）：部署落地 + AV-JEPA

| 论文 | 会议/arXiv | 核心内容 | 与项目关系 |
|---|---|---|---|
| **Deploying Speech-Driven 3D Facial Animation in Unreal Engine for Production-Ready Digital Humans（3DMEAD-ARKit）** | arXiv 2606.10753 / ACM DOI 10.1145/3799825.3818695 / SIGGRAPH 2026 演示 | 用 MediaPipe 把 MEAD 转成 ARKit blendshape 序列，重训 FaceDiffuser 与 ProbTalk3D-X（随机 + 情感可控），开发模块化 UE 插件（Python 后端），用户研究对比 Epic MetaHuman 与 NVIDIA Audio2Face；并承认 MediaPipe 转换有噪声/抖动 | **与 JEPA-ARKit 路线几乎完全重叠，是最直接的对标**（见第 9 节） |
| EchoMimicV3 | AAAI 2026 | 1.3B 参数统一多模态/多任务人体动画 | 大规模统一模型 |
| AV-JEPA | arXiv 2607.15295（2026） | 扩展 LeJEPA 到音视频自监督（SIGReg 驱动共享嵌入空间） | JEPA 音视频方向进展 |
| MJEPA | arXiv 2606.25225（2026） | 单一统一编码器的音视频 JEPA，K400 80.6% | JEPA 音视频方向进展 |
| Audio-JEPA | arXiv 2507.02915 | 音频表示学习的 JEPA | JEPA 音频方向 |

**2026 小结**：**「生产级部署（UE/ARKit + 商业工具对比）」成为新热点**，且已被 3DMEAD-ARKit 占位；JEPA 在音视频表示学习上持续升温（AV-JEPA/MJEPA），但**尚未进入面部动画领域（蓝海）**。

---

## 6. 最有价值、最值得精读的近期论文（按优先级）

1. **3DMEAD-ARKit（arXiv 2606.10753，2026-06）** — 必读。路线与 JEPA-ARKit 几乎相同（MediaPipe→ARKit→UE），读它才能找差异化：它的缺口 = MediaPipe 噪声/抖动、确定性/可审计评测缺失、无实时流式。
2. **Wav2Sem（CVPR 2025，pp.183-192）** — 必读。说明"直接用 WavLM 特征"在本任务存在已知缺陷（近音词耦合 → 唇形平均化），本项目若要走 WavLM 路线必须先解决/利用这一点。
3. **Teller（CVPR 2025）** — 必读。实时流式自回归说话人，与本项目 streaming 协议直接对标；看它如何组织流式评测。
4. **Perceptually Accurate 3D Talking Head（CVPR 2025）** — 推荐。评测指标本身就是贡献的范例，与本项目"确定性评测/可审计"定位呼应。
5. **Probabilistic Speech-Driven 3D Facial Motion Synthesis（CVPR 2024）** — 推荐。"新基准"型论文模板：它重新定义了评测类别（Stability/Style/Emotional/Impulsiveness），正是本项目可效仿的"评测贡献"路径。
6. **MJEPA / AV-JEPA（arXiv 2026）** — 推荐。JEPA 在音视频表示学习的最新范式，做表示学习论证时可直接引用/对比。
7. **D-JEPA（ICLR 2025）** — 参考。生成式 JEPA 的顶会范例（本项目 JEPA 是判别式/预测式，可对比定位）。
8. **Survey: Advancing Talking Head Generation（arXiv 2507.02900）** — 参考。多模态方法/数据集/指标/损失函数的系统综述，含 GitHub 资源清单。

---

## 7. 关键数据集与基准

| 数据集 | 类型 | 规模 | 用途/说明 |
|---|---|---|---|
| VOCASET | 3D 网格（FLAME） | 12 人 480 段 | 事实标准 3D 动画基准，申请制，研究仅用 |
| BIWI B3D(AC)2 | 3D 网格 | 14 人 | 3D 动画辅助基准，官网曾失效 |
| MEAD | 2D 视频 + 表情 | 60+ 人 8 情绪 | 情感说话人脸基准（ECCV 2020） |
| RAVDESS | 2D 视频 + 音频 | 24 人 1,440 段 | 情感音视频，CC BY-NC-SA，研究仅用（本项目主数据） |
| TFHP | 3D 面部 + 头部姿态 | DiffPoseTalk 自建 | 风格化动画 + 头姿基准 |
| HDTF | 2D 视频 | 300+ 视频 | 2D 说话人常用 |
| MMHead | 3D 面部 + 文本 | 35,903 段 | 多模态 3D 动画，门控 |
| VoxCeleb1/2 | 2D 音视频 | 数千小时 | 大规模音视频预训练/鲁棒性 |
| VFHQ / AniTalk | 2D 视频 | 大规模 | 2D 说话人训练 |
| 3DMEAD-ARKit | ARKit blendshape | MEAD 转换 | 2026 新发布，与项目最接近的公开 ARKit 数据 |

---

## 8. 评测指标演进

| 阶段 | 指标 | 说明 |
|---|---|---|
| 传统 | LVE / MVE（顶点误差）、FDD 上/下唇、唇顶点误差、MEE、CE | 几何精度 |
| 同步 | LSE-C / LSE-D、Sync-C / Sync-D | 唇音同步（2D 领域常用） |
| 生成质量 | FID / FVD、多样性（diversity） | 2D 说话人/扩散 |
| 感知 | MOS 用户研究、感知准确率 | 顶会标配 |
| 2025 新 | 物理同步 + 感知定义（Perceptually Accurate 3D Talking Head） | 评测本身成为贡献 |
| 商业对比 | 与 MetaHuman / Audio2Face 用户研究对比（3DMEAD-ARKit） | 生产级定位 |
| 本项目自建 | canonical 曲线 mouth/curve MAE、audio-shift ratio、silence ratio | 与领域标准不可比，需对齐 |

---

## 9. 热点趋势总结 + 对 JEPA-ARKit 的启示

### 9.1 热点（2025–2026 排序）

1. 写实渲染：3DGS 说话头像（GaussianSpeech、MGGTalk、TaoAvatar、VASA-3D）
2. 视频 Diffusion Transformer 说话人（Hallo3、ACTalker、EmotiveTalk、IM-Portrait）
3. 实时流式生成（Teller）
4. 情感/风格/个性化可控与解耦（EcoFace、MemoryTalker、MEDTalk、PTalker、EmoRLTalk）
5. 音频语义解耦 + SSL 特征（Wav2Sem）
6. 评测标准化与感知指标（Perceptually Accurate 3D Talking Head、商业工具对比）
7. 生产级部署（UE/ARKit，3DMEAD-ARKit）
8. AV-JEPA / 表示学习（MJEPA、AV-JEPA，尚未进入面部动画）

### 9.2 对本项目的直接启示

| 本项目方向 | 对标 | 启示 |
|---|---|---|
| MediaPipe→ARKit→UE | 3DMEAD-ARKit（2026-06） | 已被抢先占位；差异化只能在"它的缺口"上做：抗 MediaPipe 噪声的稳健表示、确定性/可审计评测、实时流式、JEPA 表示 |
| WavLM 特征 | Wav2Sem（CVPR 2025） | "直接用 SSL 特征"不是贡献；要么解决近音词耦合，要么把 WavLM 换成更先进的解耦方案 |
| 流式协议 | Teller（CVPR 2025） | 流式要上正会必须有方法贡献（自回归/大规模），单纯工程协议不够 |
| 评测/可审计 | Probabilistic CVPR 2024；Perceptually Accurate CVPR 2025 | 评测/基准可以成为贡献，但需要标准数据集 + 标准指标 + 社区认可 |
| JEPA 表示 | 面部动画领域**无 JEPA 先例**（蓝海）；MJEPA/AV-JEPA 是相邻参考 | 机会真实存在，但 E10/E11 现状（有效秩不达标、纯音频打平）必须先用强证据救活 |

### 9.3 结论

- 以"RAVDESS-only + 自建指标 + 无 SOTA 对比"现状投 CVPR/ICCV/ECCV 正会：不现实。
- 最可能成立的发表路径：**评测/基准 + 可审计确定性协议**（对标 Probabilistic CVPR2024 / Perceptually Accurate CVPR2025），或 **JEPA 表示 + 抗噪稳健性**（填补 3DMEAD-ARKit 的 MediaPipe 噪声缺口），或 **实时流式 + UE 端到端 + 用户研究 vs 商业工具**（但需在 3DMEAD-ARKit 之后提供明显增量）。
- 建议下一步：精读 3DMEAD-ARKit / Wav2Sem / Teller 三篇；申请 VOCASET/MMHead；把评测对齐到领域标准（LVE/FDD/感知用户研究）。

---

## 10. 参考文献（按年份分组）

### 2022
- FaceFormer: Speech-Driven 3D Facial Animation With Transformers — CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Fan_FaceFormer_Speech-Driven_3D_Facial_Animation_With_Transformers_CVPR_2022_paper.html

### 2023
- CodeTalker — CVPR 2023. https://github.com/Doubiiu/CodeTalker
- SelfTalk — CVPR 2023
- ProbTalk3D — CVPR 2023
- SadTalker — CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_CVPR_2023_paper.html
- DiffPoseTalk — ICCV 2023. https://huggingface.co/papers/2310.00434
- EmoTalk — ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.html
- Imitator — ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Thambiraja_Imitator_Personalized_Speech-driven_3D_Facial_Animation_ICCV_2023_paper.html
- Semi-supervised Speech-driven 3D Facial Animation via Cross-modal Encoding — ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Semi-supervised_Speech-driven_3D_Facial_Animation_via_Cross-modal_Encoding_ICCV_2023_paper.html
- FaceDiffuser — SIGGRAPH Asia 2023

### 2024
- FaceTalk — CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Aneja_FaceTalk_Audio-Driven_Motion_Diffusion_for_Neural_Parametric_Head_Models_CVPR_2024_paper.html
- Probabilistic Speech-Driven 3D Facial Motion Synthesis: New Benchmarks, Methods and Applications — CVPR 2024. https://mlanthology.org/cvpr/2024/yang2024cvpr-probabilistic/
- Towards Variable and Coordinated Holistic Co-Speech Motion Generation — CVPR 2024
- SyncTalk — CVPR 2024
- KMTalk — ECCV 2024. https://eccv2024.ecva.net/virtual/2024/poster/1279
- S3D-NeRF — ECCV 2024
- ScanTalk — ECCV 2024
- Loc3Diff — ECCV 2024
- AnimateMe — ECCV 2024
- AniPortrait — arXiv:2403.17694, 2024
- Hallo — arXiv 2024；EchoMimic — arXiv 2024/AAAI 2025；MuseTalk — arXiv 2024

### 2025
- Wav2Sem — CVPR 2025, pp.183-192. https://openaccess.thecvf.com/content/CVPR2025/html/Li_Wav2Sem_Plug-and-Play_Audio_Semantic_Decoupling_for_3D_Speech-Driven_Facial_Animation_CVPR_2025_paper.html
- Teller — CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Zhen_Teller_Real-Time_Streaming_Audio-Driven_Portrait_Animation_with_Autoregressive_Motion_Generation_CVPR_2025_paper.html
- EmotiveTalk — CVPR 2025
- MGGTalk — CVPR 2025
- TaoAvatar — CVPR 2025
- Perceptually Accurate 3D Talking Head Generation — CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Chae-Yeon_Perceptually_Accurate_3D_Talking_Head_Generation_New_Definitions_Speech-Mesh_Representation_CVPR_2025_paper.html
- IM-Portrait — CVPR 2025
- Hallo3 — CVPR 2025
- ACTalker — ICCV 2025
- MemoryTalker — ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Kim_MemoryTalker_Personalized_Speech-Driven_3D_Facial_Animation_via_Audio-Guided_Stylization_ICCV_2025_paper.html
- GaussianSpeech — ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Aneja_GaussianSpeech_Audio-Driven_Personalized_3D_Gaussian_Avatars_ICCV_2025_paper.html
- Expressive Talking Human from Single-Image with Imperfect Priors — ICCV 2025
- EcoFace — ICLR 2025. https://mlanthology.org/iclr/2025/xie2025iclr-ecoface/
- VASA-3D — NeurIPS 2025
- MegActor-Sigma — AAAI 2025
- AnyTalk (+AniTalk) — AAAI 2025
- GLDiTalker — IJCAI 2025. https://www.ijcai.org/proceedings/2025/173
- Ditto — ACM MM 2025
- PTalker — ACM MM 2025 / arXiv:2512.22602
- PESTalk — ACM MM 2025
- MEDTalk — arXiv 2025
- EmoRLTalk — ICME 2025
- Model See Model Do — arXiv:2505.01319, 2025
- Survey: Advancing Talking Head Generation: A Comprehensive Survey of Multi-Modal Methodologies, Datasets, Evaluation Metrics, and Loss Functions — arXiv:2507.02900

### 2026（最新）
- Deploying Speech-Driven 3D Facial Animation in Unreal Engine for Production-Ready Digital Humans（3DMEAD-ARKit）— arXiv:2606.10753 / ACM DOI 10.1145/3799825.3818695（SIGGRAPH 2026 演示）
- EchoMimicV3 — AAAI 2026
- AV-JEPA — arXiv:2607.15295
- MJEPA — arXiv:2606.25225
- Audio-JEPA — arXiv:2507.02915
- D-JEPA — ICLR 2025（生成式 JEPA，供表示学习定位参考）

### 本领域经典基线（更早）
- VOCA — SIGGRAPH Asia 2019；MeshTalk — ICCV 2021；Wav2Lip — ECCV 2020；MEAD — ECCV 2020；VoxCeleb — 2017/2018
