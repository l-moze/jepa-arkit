# JEPA-ARKit 数据集下载指南

> 更新日期：2026-08-13 ｜ 本文档面向需要重新获取本项目数据的人（包括 AI 代理）。
> 所有下载链接均已实测验证（HTTP 200 / 可 clone）。**本机已下载的数据不需要重复下载**，位置见第 7 节。
> 注意：`data/` 目录已被 `.gitignore` 排除，不会进入 GitHub 仓库；GitHub 仓库（https://github.com/l-moze/jepa-arkit）只包含代码与配置。
> 本机数据清单的权威说明见 `data/README.md`；每个数据集的角色/访问方式/阻断项登记在 `configs/data/dataset_catalog.yaml`（14 个候选）；许可证据快照在 `docs/evidence/datasets/`。

---

## 0. 一句话总结

| 要下载的东西 | 从哪里下 | 多大 |
|---|---|---|
| RAVDESS 主数据集（24 个 Video_Speech zip） | Zenodo（DOI 10.5281/zenodo.1188976） | 约 12.4 GiB |
| CREMA-D 辅助数据集（第二来源候选） | GitLab 镜像（git clone + LFS） | 全量约 7.55 GB（本机只拉 WAV+FLV 约 2.9 GiB） |
| UniTalker 候选数据（隔离，仅审计） | Google Drive | 2.44 GiB |
| MediaPipe Face Landmarker 模型 | Google 存储 | 3.6 MB |
| WavLM Base 模型（提取特征用） | Hugging Face | 约 380 MB |
| 其他候选数据集（14 个登记项，见第 6 节） | 各自官网（多数需申请/门控） | 未下载 |

---

## 1. 总览

| 数据集 | 状态 | 本地路径 | 大小 | 许可/合规 |
|---|---|---|---|---|
| RAVDESS 1.0.0（Video_Speech） | 已下载、24/24 MD5 校验通过 | `data/raw/ravdess/archives/` | 12.44 GiB | CC BY-NC-SA 4.0，仅研究轨 |
| RAVDESS 规范 release | 已生成 | `data/raw/ravdess/release_v1/` | 6.34 GiB | 同上（派生数据） |
| CREMA-D | 已下载（GitLab 镜像，LFS） | `data/raw/crema_d/official_gitlab_mirror/` | 2.90 GiB（本地拉取子集） | ODbL 1.0 + Database Contents License |
| UniTalker 候选 | 已下载、CRC 通过、**已隔离** | `data/candidate/unitalker_released_v1/` | 2.44 GiB | 上游来源逐个 blocked → 禁止训练 |
| WavLM Base FP16 特征 v1/v2 | 已提取 | `data/real/features/wavlm_base_fp16_ravdess_v1/`、`_v2/` | 各 0.36 GiB | 派生特征 |
| MediaPipe Face Landmarker | 已下载（SHA-256 已核验） | `data/models/face_landmarker.task` | 3.6 MB | Apache-2.0 |
| 规范音频+动作（prepared） | 已生成 | `data/real/ravdess_prepared_v1/` | 0.20 GiB | 派生数据 |

---

## 2. RAVDESS（主训练/评测数据集）

- 官方记录页：https://zenodo.org/records/1188976 ｜ DOI：`10.5281/zenodo.1188976`
- 记录标题：The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)，version 1.0.0，许可 `cc-by-nc-sa-4.0`
- **该记录共 49 个文件**：24 个 `Video_Speech_Actor_*.zip`（12.44 GiB，本项目使用）+ 23 个 `Video_Song_Actor_*.zip`（11.00 GiB）+ 1 个 `Audio_Speech` 归档（0.19 GiB）+ 1 个 `Audio_Song` 归档（0.21 GiB）。**本项目只下载/使用 24 个 Video_Speech 归档**。
- 文件：`Video_Speech_Actor_01.zip` ~ `Video_Speech_Actor_24.zip`（每个约 500–570 MB），共 24 位演员、1,440 个语音视频片段。
- 下载地址（已验证 HTTP 200）：
  - 网页直链：`https://zenodo.org/records/1188976/files/Video_Speech_Actor_01.zip?download=1`
  - API 直链：`https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_01.zip/content`
  - 每个文件的官方 **MD5** 可通过 API 记录 `https://zenodo.org/api/records/1188976` 的 `files[].checksum` 字段取得（如 `md5:3c42877921cc08cfb5c841a0f2cb94a7`）。

PowerShell 批量下载（断点续传，失败自动重试）：

```powershell
New-Item -ItemType Directory -Force -Path data\raw\ravdess\archives | Out-Null
1..24 | ForEach-Object {
  $a = '{0:D2}' -f $_
  curl.exe -L --fail --retry 5 --retry-delay 5 -C - `
    -o "data\raw\ravdess\archives\Video_Speech_Actor_$a.zip" `
    "https://zenodo.org/records/1188976/files/Video_Speech_Actor_$a.zip?download=1"
}
```

校验 + 解压 + 生成规范 release（`ingest-ravdess` 自动比对官方 MD5，全部通过才解压）：

```powershell
uv run jepa-arkit ingest-ravdess `
  --archives data\raw\ravdess\archives `
  --output data\raw\ravdess\release_v1
```

许可：**CC BY-NC-SA 4.0**。只能进入研究轨，产出的 checkpoint 一律 `research_only`，不能初始化产品轨模型。
本项目的 RAVDESS 权利登记见 `configs/data/ravdess_rights_registry.json`，官方许可证据见 `docs/evidence/datasets/ravdess-license-2026-08-13.md`。

---

## 3. CREMA-D（辅助数据，第二来源候选）

- 官方 GitLab 镜像（已验证可 clone）：https://gitlab.com/cs-cooper-lab/crema-d-mirror.git
- 参考 GitHub 组织页：https://github.com/CheyneyComputerScience/CREMA-D
- 规模：91 位演员、7,442 条真实音视频（完整约 7.55 GB）。
- **许可：ODbL 1.0（数据）+ Database Contents License（单独内容）**。本机已按官方 GitLab 镜像拉取并保留 LFS 指针与对象校验。

需要先安装 [Git LFS](https://git-lfs.com/)，然后：

```powershell
git lfs install
git clone https://gitlab.com/cs-cooper-lab/crema-d-mirror.git data\raw\crema_d\official_gitlab_mirror
cd data\raw\crema_d\official_gitlab_mirror
# 只拉取 WAV 音频和 Flash 视频（跳过 AudioMP3 可省流量）
git lfs pull --include="AudioWAV/**,VideoFlash/**" --exclude="AudioMP3/**"
```

- 本机拉取结果：`AudioWAV` 578 MB、`VideoFlash` 2.3 GiB、`AudioMP3` 很小（未拉），合计 2.90 GiB。
- 用途：作为 RAVDESS 之外的**第二来源候选**，检验更强的 identity-disjoint 泛化；仓库已加入可复现的 CREMA-D 接入/审计代码（文件名解析、音视频配对、WAV 解码抽检、许可与 Git 提交记录、LFS 指针检测），但**不会自动混入现有 RAVDESS 训练 release**。
- 注意：CREMA-D 尚未登记进 `dataset_catalog.yaml`（属补充候选）。

---

## 4. UniTalker 候选数据（已隔离，仅审计）

- 分享页：https://drive.google.com/file/d/1Un7TB0Z5A1CG6bgeqKlhnSOECFN-C6KK/view
- 直链下载（文件 ID：`1Un7TB0Z5A1CG6bgeqKlhnSOECFN-C6KK`）：

```powershell
New-Item -ItemType Directory -Force -Path data\candidate\unitalker_released_v1 | Out-Null
curl.exe -L --fail --retry 5 --retry-delay 5 -C - `
  -o data\candidate\unitalker_released_v1\unitalker_data_release_V1.zip `
  "https://drive.usercontent.google.com/download?id=1Un7TB0Z5A1CG6bgeqKlhnSOECFN-C6KK&export=download&confirm=t"
```

CRC 审计（本机已执行过：8,002 条目，CRC 通过）：

```powershell
uv run jepa-arkit audit-unitalker-candidate `
  --archive data\candidate\unitalker_released_v1\unitalker_data_release_V1.zip `
  --output artifacts\unitalker_candidate_audit.json
```

**归档组成（8 个上游来源，全部 blocked）**，见 `src/jepa_arkit/data/unitalker.py` 的 `SOURCE_RIGHTS`：

| 来源 | 阻断原因 |
|---|---|
| D0 BIWI B3D(AC)2 | 上游条款与再分发权未核实 |
| D1 VOCASET | 研究仅用申请条款，派生再分发权未核实 |
| D2 MeshTalk / Multiface | 上游数据集条款与受试者同意范围未核实 |
| D3 HDTF（经 3DETF） | 源媒体与派生标注权利未核实 |
| D4 RAVDESS（经 3DETF） | 派生标签缺验证流水线与发布谱系 |
| D5 FaceForensics++ | 申请条款与源媒体权利需审查 |
| D6 UniTalker 自有中文语音 | 归档无参与者同意或数据集许可 |
| D7 UniTalker 自有歌曲 | 归档无参与者/录音/音乐权利许可 |

**重要**：该归档混合多个上游来源且无伞形训练许可，属于隔离（quarantine）状态，**不得用于训练**，只做完整性/组成审计。审计报告示例：`docs/evidence/datasets/unitalker-candidate-2026-08-13.md`。

---

## 5. 模型文件

### 5.1 MediaPipe Face Landmarker（生成动作标签用）

- 下载地址（已验证 HTTP 200，3,758,596 字节，与本地文件一致）：
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`

```powershell
curl.exe -L --fail -o data\models\face_landmarker.task `
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

- 本机文件的期望 SHA-256（已核验一致）：`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`（见 `data/README.md`）
- 许可：Apache-2.0。

### 5.2 WavLM Base（提取冻结音频特征用）

- Hugging Face 仓库：https://huggingface.co/microsoft/wavlm-base
- 包含文件：`config.json`、`preprocessor_config.json`、`pytorch_model.bin`
- **固定 revision**（本仓库锁定，等于 HF 主干 SHA）：`efa81aae7ff777e464159e0f877d54eac5b84f81`
- **许可：HF 模型卡未声明 license 字段 → 标记「待核验」**，使用前需以官方条款为准（代码为 MIT，但权重许可未明确）。

自动下载（推荐，transformers 会连同 config 一起下载并缓存）：

```powershell
uv run jepa-arkit extract-wavlm `
  --manifest <release_manifest.json> `
  --output data\real\features\wavlm_base_fp16_ravdess_v2 `
  --model-id microsoft/wavlm-base `
  --revision efa81aae7ff777e464159e0f877d54eac5b84f81 `
  --source-release-id ravdess_prepared_v1 `
  --feature-release-id wavlm_base_fp16_ravdess_v2
```

手动下载主权重：

```powershell
curl.exe -L --fail -o pytorch_model.bin `
  "https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin"
```

---

## 6. 候选数据集（未下载 / 需要申请或门控）

完整登记见 `configs/data/dataset_catalog.yaml`（14 项），许可证据快照见 `docs/evidence/datasets/`。下表按 catalog 顺序列出（RAVDESS、UniTalker 已单独成节，此处不重复）：

| 数据集 | 地址 | 访问方式 | 状态 / 阻断原因 |
|---|---|---|---|
| VOCASET | https://voca.is.tue.mpg.de/ | 官网申请 | 研究仅用（证据已存），申请制 |
| BIWI B3D(AC)2 | https://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz | 官网失效，用镜像 | 许可未知，需核实镜像完整性 |
| 3DMEAD (Face-Diffusion-Model) | https://github.com/wangxuanx/Face-Diffusion-Model | 项目链接（百度网盘） | 许可未核实，源自 MEAD |
| NVIDIA Audio2Face-3D Sample | https://huggingface.co/datasets/nvidia/Audio2Face-3D-Dataset-v1.0.0-claire | HF gated | 仅演示，禁止训练 |
| 3D-CAVFA | https://github.com/X-niper/UniTalker | 项目链接 | 未定位发布物，说话人同意未核实 |
| MMHead | https://huggingface.co/datasets/Human-X/MMHead | HF gated | 门控仓库，研究仅用（证据已存），catalog 仍标 unverified |
| DiffPoseTalk TFHP | https://diffposetalk.github.io/ | 项目链接 | 许可未知，源视频权利待核实 |
| EmoVOCA (Synthetic) | https://voca.is.tue.mpg.de/ | 需先获取 VOCASET + Florence 4D | 派生数据，取决于两个源许可 |
| Florence 4D Facial Expression | （未定位权威来源） | 未知 | 需定位权威来源并保存许可 |
| VoxCeleb 1 & 2 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ | 官网注册 | CC BY 4.0 声明待核实，源媒体条款待核实 |
| FaceForensics++ | https://github.com/ondyari/FaceForensics | 表格/邮件申请 | 许可未知，默认禁止动作训练 |
| AVSpeech | https://looking-to-listen.github.io/avspeech/ | 元数据 + 按 YouTube 条款自取视频 | 源平台条款待核实 |

---

## 7. 本机已下载位置汇总（2026-08-13 实测）

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/raw/ravdess/archives/` | 24 个官方 Video_Speech zip（MD5 已校验） | 12.44 GiB |
| `data/raw/ravdess/release_v1/` | 解压后的视频 + 标签（1,443 文件） | 6.34 GiB |
| `data/raw/crema_d/official_gitlab_mirror/` | CREMA-D GitLab 镜像工作树（WAV+FLV） | 2.90 GiB |
| `data/candidate/unitalker_released_v1/` | UniTalker zip（隔离） | 2.44 GiB |
| `data/real/ravdess_prepared_v1/` | 规范 16 kHz 音频 + 规范 ARKit 动作 | 0.20 GiB |
| `data/real/features/wavlm_base_fp16_ravdess_v1/` | WavLM Base FP16 特征（v1） | 0.36 GiB |
| `data/real/features/wavlm_base_fp16_ravdess_v2/` | WavLM Base FP16 特征（v2，精确中心对齐） | 0.36 GiB |
| `data/pilot/` | prepare 基准与特征基准小样本 | 约 3 MB |
| `data/demo/` | 合成演示数据（manifest + 音频 + 动作） | 约 2 MB |
| `data/models/face_landmarker.task` | MediaPipe 模型（SHA-256 已核验） | 3.6 MB |

---

## 8. 数据流水线（从原始数据到特征）

```
Zenodo zip (data/raw/ravdess/archives)
        │  uv run jepa-arkit ingest-ravdess   （MD5 校验 + 解压 + 来源追踪）
        ▼
data/raw/ravdess/release_v1
        │  uv run jepa-arkit prepare-ravdess --release ... --output ... --model data/models/face_landmarker.task
        │  （MediaPipe 解算规范动作 + 16 kHz 音频 + 缺失曲线策略）
        ▼
data/real/ravdess_prepared_v1
        │  uv run jepa-arkit build-ravdess-release --raw-release ... --prepared ... --output ...
        ▼
release manifest（.jsonl，含 1,440 条样本记录）
        │  uv run jepa-arkit extract-wavlm --manifest ... --revision <固定 revision> ...
        ▼
data/real/features/wavlm_base_fp16_ravdess_v2
```

关键点：

- `ingest-ravdess` 校验官方 MD5（仓库内 `src/jepa_arkit/data/ravdess.py` 保存了官方校验和表），24/24 通过才允许继续。
- `prepare-ravdess` 使用 `data/models/face_landmarker.task`（SHA-256 已核验）生成规范动作；缺失曲线按 `configs/contracts/mediapipe_missing_curve_policy_v1.json` 策略处理。
- 划分：actor-disjoint，840 train（01–14）/ 300 validation（15–19）/ 300 test（20–24）。
- 特征提取把 WavLM Base 锁定在固定 revision `efa81aae7ff777e464159e0f877d54eac5b84f81`，保证可复现。
- v2 特征使用卷积感受野精确中心对齐（首帧 12.46875 ms、步长 20 ms，266,110 帧、768 维 FP16）。
- 若未激活 `uv` 环境，可用 `.venv\Scripts\python.exe -m jepa_arkit.cli <command> ...` 等价调用。

---

## 9. 合规与双轨治理

- 数据按「研究轨 / 产品轨」双轨管理：研究轨允许非商业数据，产物必须带 `research_only=true`；产品轨只接受可商用、有充分同意的数据。
- RAVDESS（CC BY-NC-SA 4.0）、VOCASET、MMHead 均为研究仅用，不能初始化产品 checkpoint。
- UniTalker 归档 8 个上游来源全部 blocked → 隔离，仅审计。
- CREMA-D：ODbL 1.0 + Database Contents License。
- MediaPipe Face Landmarker（Apache-2.0）许可明确；**WavLM Base 许可未在 HF 模型卡声明（待核验）**。
- 删除/撤回：通过 `withdrawal_key` 可定位原始样本、派生标签、特征缓存与受影响 checkpoint（详见研究计划 §4.6）。
- 许可证据统一存放于 `docs/evidence/datasets/`，每条记录带 SHA-256 与日期。

---

## 10. 参考

- GitHub 仓库：https://github.com/l-moze/jepa-arkit
- 研究计划：`docs/jepa-arkit-research-plan.md`
- 实施状态：`docs/implementation-status.md`
- 数据布局说明：`data/README.md`
- 数据集目录登记（14 项）：`configs/data/dataset_catalog.yaml`
- RAVDESS 权利登记：`configs/data/ravdess_rights_registry.json`
- 许可证据快照：`docs/evidence/datasets/`
- 官方校验和：`src/jepa_arkit/data/ravdess.py` ｜ UniTalker 来源权利：`src/jepa_arkit/data/unitalker.py`
- Codex 会话记录（数据下载与处理的完整过程）：
  - 主下载/审计会话：`C:\Users\24787\.codex\sessions\2026\08\11\rollout-2026-08-11T20-32-35-019ff0cf-74c4-77e0-a426-7a76335337bc.jsonl`
  - 数据处理/特征会话：`C:\Users\24787\.codex\sessions\2026\08\11\rollout-2026-08-11T15-26-18-019fefb7-09d0-7943-8d88-e188b7b440d3.jsonl`
