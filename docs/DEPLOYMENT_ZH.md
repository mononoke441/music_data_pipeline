# Music-Data-Pipeline 部署指南

本文给出一套可以复制执行的 Linux/NVIDIA 部署流程。主流程入口始终是：

```bash
bash run_pipeline.sh INPUT_DIR RESULT_DIR
```

以下命令基于 Ubuntu 22.04、Python 3.10 和 NVIDIA CUDA 12 运行时。项目已在单张 H100 80GB 上完成全流程验证；软件版本以本文列出的已验证组合为准。

## 1. 先选部署方式

| 方式 | Qwen3-Omni | 适用场景 | 状态 |
|---|---|---|---|
| 单机单卡 | 1×80GB，BF16 | 最简单的完整 Pipeline | 已验证、推荐 |
| Pipeline 与 Omni 分离 | Omni 服务使用 2×48GB、TP=2 | 已有独立推理机 | 可部署，但需在目标卡型实测 |
| 单张 48GB | 不加载原始 BF16 Omni | 只跑门控、Discogs、CPU MIR、SongFormer | 可用 |

原始 `Qwen3-Omni-30B-A3B-Instruct` 权重约 70.5GB。单张 48GB 无法运行本项目使用的 BF16 版本。当前 `ALM_MODE=local` 固定按单卡 TP=1 启动，因此 2×48GB 不能只靠设置 `CUDA_VISIBLE_DEVICES=0,1` 自动切分；应把 Omni 作为独立 TP=2 服务部署，并通过 `ALM_MODE=external` 连接。

如果 TP=2 Omni 与 Pipeline 位于同一台只有两张 48GB 卡的机器，Omni 会占用两张卡，而外部模式会让它与 SongFormer/ASR 重叠，不适合直接全流程共存。完整部署请使用以下任一方案：

- 单张 80GB 卡，使用 `ALM_MODE=local`；
- Pipeline 机器和 2×48GB Omni 服务机器分离；
- 先只部署前半段，不启用 ALM、Section Caption 和 ASR。

建议资源：16 个以上 CPU 核、64GB 以上内存，以及除输入/输出数据外至少 150GB 可用磁盘。模型、缓存和结果目录应放在本地 SSD 或高吞吐共享存储上。

## 2. 获取代码和系统依赖

```bash
export REPO_ROOT=/opt/Music-Data-Pipeline
export ENV_ROOT=/opt/conda-envs
export PIPELINE_MODEL_ROOT=/opt/models/Music-Data-Pipeline

git clone git@github.com:mononoke441/music_data_pipeline.git "$REPO_ROOT"
cd "$REPO_ROOT"

sudo apt-get update
sudo apt-get install -y \
  build-essential curl ffmpeg git libexpat1 libsndfile1 pkg-config

nvidia-smi
ffmpeg -version
```

若仓库已经存在，只需确认当前位于项目根目录。`ffmpeg`、`ffprobe`、`nvidia-smi` 和 `curl` 都必须可执行。

## 3. 创建三个隔离环境

项目运行时有三个职责不同的 Python 环境：

1. `moss-music-pipeline`：门控、Discogs、CPU MIR、MuQ、MusicFM、SongFormer；
2. `qwen3-vllm`：Qwen3-ASR 与 ForcedAligner；
3. `omni-vllm`：仅用于本地 Qwen3-Omni OpenAI-compatible 服务。

第三个环境只有使用 `ALM_MODE=local`，或在本机单独启动 Omni 服务时才需要。

### 3.1 音乐分析环境

```bash
conda create -y -p "$ENV_ROOT/moss-music-pipeline" python=3.10

"$ENV_ROOT/moss-music-pipeline/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
"$ENV_ROOT/moss-music-pipeline/bin/python" -m pip install -r SongFormer/requirements.txt
"$ENV_ROOT/moss-music-pipeline/bin/python" -m pip install -r requirements.txt
"$ENV_ROOT/moss-music-pipeline/bin/python" -m pip install \
  'ray==2.56.1' \
  'essentia-tensorflow==2.1b6.dev1389' \
  'BeatNet==1.1.1' \
  'chord-extractor==0.1.3' \
  'requests>=2.31'
```

已验证的关键版本是 Python 3.10、PyTorch/torchaudio 2.4.0、ONNX Runtime GPU 1.20.2、Ray 2.56.1、Transformers 4.51.1、NumPy 1.25.0。

### 3.2 ASR 环境

```bash
conda create -y -p "$ENV_ROOT/qwen3-vllm" python=3.10

"$ENV_ROOT/qwen3-vllm/bin/python" -m pip install --upgrade pip wheel setuptools
"$ENV_ROOT/qwen3-vllm/bin/python" -m pip install \
  'qwen-asr==0.0.6' \
  'vllm==0.14.0' \
  'qwen-omni-utils==0.0.9' \
  'modelscope==1.39.1'
```

### 3.3 本地 Omni 环境

```bash
conda create -y -p "$ENV_ROOT/omni-vllm" python=3.10

"$ENV_ROOT/omni-vllm/bin/python" -m pip install --upgrade pip wheel setuptools
"$ENV_ROOT/omni-vllm/bin/python" -m pip install \
  'vllm==0.13.0' \
  'qwen-omni-utils==0.0.9' \
  'modelscope==1.36.0'
"$ENV_ROOT/omni-vllm/bin/python" -m pip install \
  'flash-attn==2.8.3' --no-build-isolation
```

以上是当前项目实测环境组合。不要把 ASR 的 vLLM 0.14 环境直接替代已验证的 Omni vLLM 0.13 环境。

## 4. 下载权重

先让下载脚本使用刚创建的环境和统一模型目录：

```bash
cd "$REPO_ROOT"

export PY_PIPELINE="$ENV_ROOT/moss-music-pipeline/bin/python"
export PY_QWEN="$ENV_ROOT/qwen3-vllm/bin/python"
export HF_BIN="$ENV_ROOT/qwen3-vllm/bin/hf"
export MODELSCOPE_BIN="$ENV_ROOT/qwen3-vllm/bin/modelscope"
export PIPELINE_MODEL_ROOT

mkdir -p "$PIPELINE_MODEL_ROOT"

bash scripts/download_gate_assets.sh all
bash scripts/download_weights.sh muq wav2vec discogs omni asr
```

再下载 MusicFM 与 SongFormer：

```bash
cd "$REPO_ROOT/SongFormer"
"$PY_PIPELINE" -c \
  'from utils.fetch_pretrained import download_all; download_all(use_mirror=True)'
cd "$REPO_ROOT"
```

`use_mirror=True` 使用 `hf-mirror.com`；能直接访问 Hugging Face 时可改为 `False`。不要使用 `download_weights.sh all` 做生产部署：`all` 还会下载主 Pipeline 不使用的 235B LLM，浪费大量磁盘。

下载完成后应存在：

```text
MusicToolsPipeline/checkpoints/fast_gate/MobileNetV1_mAP=0.389.pth
MusicToolsPipeline/checkpoints/fast_gate_config.json
MusicToolsPipeline/discogs_onnx/*.onnx
SongFormer/ckpts/MuQ-large-msd-iter/model.safetensors
SongFormer/ckpts/MusicFM/pretrained_msd.pt
SongFormer/ckpts/MusicFM/msd_stats.json
SongFormer/ckpts/SongFormer.safetensors
SongFormer/ckpts/wav2vec2-conformer-rope-large-960h-ft/config.json
$PIPELINE_MODEL_ROOT/Qwen3-Omni-30B-A3B-Instruct/
$PIPELINE_MODEL_ROOT/Qwen3-ASR-1.7B/
$PIPELINE_MODEL_ROOT/Qwen3-ForcedAligner-0.6B/
```

门控权重可以再次离线校验：

```bash
bash scripts/download_gate_assets.sh --verify all
```

## 5. 设置运行变量

每次运行前设置以下变量。不要依赖 `run_pipeline.sh` 中为现有集群保留的默认绝对路径。

```bash
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0
export PY_PIPELINE="$ENV_ROOT/moss-music-pipeline/bin/python"
export PY_QWEN="$ENV_ROOT/qwen3-vllm/bin/python"
export OMNI_VLLM_BIN="$ENV_ROOT/omni-vllm/bin/vllm"
export OMNI_CPATH="$ENV_ROOT/omni-vllm/include/python3.10"
export PIPELINE_EXPAT_LIB="$("$PY_PIPELINE" -c 'import sys; print(sys.base_prefix)')/lib/libexpat.so.1"

export OMNI_MODEL_PATH="$PIPELINE_MODEL_ROOT/Qwen3-Omni-30B-A3B-Instruct"
export QWEN3_ASR_MODEL_PATH="$PIPELINE_MODEL_ROOT/Qwen3-ASR-1.7B"
export QWEN3_ALIGNER_MODEL_PATH="$PIPELINE_MODEL_ROOT/Qwen3-ForcedAligner-0.6B"
```

## 6. 部署前检查

```bash
bash -n run_pipeline.sh
bash -n scripts/download_weights.sh
bash -n scripts/download_gate_assets.sh

LD_PRELOAD="$PIPELINE_EXPAT_LIB" "$PY_PIPELINE" - <<'PY'
import onnxruntime as ort
import ray
import torch
import torchaudio
from BeatNet.BeatNet import BeatNet
from chord_extractor.extractors import Chordino
import essentia.standard
import muq

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("onnxruntime providers", ort.get_available_providers())
assert torch.cuda.is_available()
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY

"$PY_QWEN" - <<'PY'
from qwen_asr import Qwen3ASRModel
import vllm
print("ASR imports OK", vllm.__version__)
PY

"$OMNI_VLLM_BIN" --version
```

如果 `chord-extractor` 的导入路径因发行版本不同而变化，可用下面的项目真实调用路径确认安装；正式运行以此为准：

```bash
(
  cd "$REPO_ROOT/MusicToolsPipeline"
  LD_PRELOAD="$PIPELINE_EXPAT_LIB" "$PY_PIPELINE" -c \
    'from sub_models.chordino_model import ChordinoModel; print("Chordino import OK")'
)
```

## 7. 单张 80GB：完整运行

这是最简单、项目已经验证过的部署方式。runner 会在 SongFormer 结束后临时启动 Omni，在 Section Caption 结束后关闭 Omni，再启动 ASR/ForcedAligner，避免三类模型同时占用同一张 GPU。

```bash
export CUDA_VISIBLE_DEVICES=0
export ALM_MODE=local
export OMNI_GPU_MEMORY_UTILIZATION=0.90
export OMNI_MAX_MODEL_LEN=16384
export OMNI_MAX_NUM_SEQS=2
export VLLM_GPU_HEADROOM_GIB=4

bash run_pipeline.sh /data/raw_media /data/results/run_001
```

结果写入：

```text
/data/results/run_001/final/annotations/
/data/results/run_001/final/review.jsonl
/data/results/run_001/final/rejected.jsonl
/data/results/run_001/final/retry.jsonl
/data/results/run_001/intermediate/logs/
```

## 8. 2×48GB：部署独立 Omni 服务

在独立 Omni 服务机器上执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMNI_MODEL_PATH="$PIPELINE_MODEL_ROOT/Qwen3-Omni-30B-A3B-Instruct"
export OMNI_VLLM_BIN="$ENV_ROOT/omni-vllm/bin/vllm"

"$OMNI_VLLM_BIN" serve "$OMNI_MODEL_PATH" \
  --host 0.0.0.0 \
  --port 10008 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"audio":1,"image":1,"video":1}' \
  --mm-encoder-attn-backend TORCH_SDPA \
  --allowed-local-media-path / \
  --served-model-name Qwen3-Omni-30B-A3B-Instruct
```

首次在某种 48GB 卡型上启动时，建议先把 `--max-num-seqs` 调成 `1`。若仍 OOM，再把 `--max-model-len` 调成 `8192`。TP=2 方案需要在具体卡型、驱动和 vLLM 版本上完成实际验证，本文不把它标记为项目已验证配置。

在 Pipeline 机器上确认服务：

```bash
export ALM_SERVER=http://OMNI_HOST:10008
curl -fsS "$ALM_SERVER/v1/models"
```

然后运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export ALM_MODE=external
export ALM_MODEL=Qwen3-Omni-30B-A3B-Instruct
export PARALLEL_ASR_WITH_EXTERNAL_ALM=1

bash run_pipeline.sh /data/raw_media /data/results/run_001
```

`ALM_SERVER` 必须能读取请求中使用的音频 URI。最稳妥的方式是让两台机器以相同绝对路径挂载输入数据；否则需要在服务端与 Pipeline 端验证文件 URI 的可达性。服务端口不要直接暴露到公网，应通过内网、防火墙或反向代理限制访问。

## 9. 单张 48GB：先运行不依赖 Qwen 的前半段

```bash
export CUDA_VISIBLE_DEVICES=0

RUN_ALM=0 \
RUN_SECTION_CAPTION=0 \
RUN_ASR=0 \
  bash run_pipeline.sh /data/raw_media /data/results/run_001
```

该方式仍会运行资产登记、PANNs 门控、Discogs、CPU MIR、MuQ/MusicFM、SongFormer/Instrumental 结构和 Section Key。Qwen 相关字段会按统一 schema 标记为 `not_run`。

## 10. 小样本验收与断点恢复

正式批量运行前，准备一个只包含 1–3 首音频的小目录，并使用独立结果目录：

```bash
bash run_pipeline.sh /data/smoke_audio /data/results/smoke_001

find /data/results/smoke_001/final -maxdepth 4 -type f -print
tail -n 50 /data/results/smoke_001/intermediate/logs/pipeline.log
cat /data/results/smoke_001/intermediate/logs/pipeline_runtime.json
cat /data/results/smoke_001/intermediate/logs/stage_timings.jsonl
```

断点恢复时使用完全相同的 `INPUT_DIR` 和 `RESULT_DIR` 重跑同一条命令。不要删除 `intermediate/`；缓存命中的阶段会直接复用。模型版本字段仅作 provenance，不会自动使旧缓存失效。主动更换模型后，应使用新的结果目录，或明确清理对应阶段缓存。

## 11. 常见故障

### Omni 启动即 OOM

- 原始 BF16 模型不能放入单张 48GB；使用单张 80GB 或独立 2×48GB TP=2 服务。
- 确认 GPU 没有其他进程：`nvidia-smi`。
- 80GB 卡可先把 `OMNI_MAX_NUM_SEQS` 降为 `1`，再把 `OMNI_MAX_MODEL_LEN` 降为 `8192`。

### ONNX Runtime 没有 CUDA provider

确认安装的是 `onnxruntime-gpu==1.20.2`，且环境内存在 cuDNN 9。Pipeline 会拒绝 Discogs 静默回退到 CPU。

### `ALM_MODE=local` 报端口占用

本地模式要求 `OMNI_HOST:OMNI_PORT` 未被占用。已有独立服务时使用 `ALM_MODE=external`，不要让 runner 接管同一端口。

### 外部 Omni 能健康检查，但读不到音频

外部服务需要访问请求中的本地文件 URI。让服务机挂载相同路径，或把 Omni 与输入存储部署在同一网络文件系统中。

### 结果中存在 `retry.jsonl`

单条失败不会使整批退出失败。先查看 `final/retry.jsonl` 与 `intermediate/logs/pipeline.log`，修复服务或媒体问题后，用原命令原结果目录继续运行。

## 12. 当前已验证基线

截至 2026-08-13，项目验证基线为：

- Ubuntu 22.04、NVIDIA H100 80GB、驱动 550.54.15；
- 主环境：Python 3.10.20、PyTorch 2.4.0、ONNX Runtime GPU 1.20.2、Ray 2.56.1；
- ASR 环境：Qwen ASR 0.0.6、vLLM 0.14.0、Qwen Omni Utils 0.0.9；
- Omni 环境：vLLM 0.13.0、Qwen Omni Utils 0.0.9；
- FFmpeg 4.4.2；
- 180 个远程测试和 Ruff F 检查通过；
- 完整 Song/Instrumental、全曲/分段 Caption、Song ASR 与严格最终校验均已通过。

生产部署前仍应在目标 GPU、驱动和存储环境中执行第 10 节的小样本验收。
