# 常驻模型服务部署指南

## 1. 部署边界

Pipeline 分为两类进程：

- runner：`run_pipeline.sh` 或 `run_stream_pipeline.sh`，只做无模型编排。
- 常驻服务：Fast Gate、Discogs、CPU MIR、SongFormer、Section ASR、Omni proxy。

runner 与服务必须能访问相同的音频绝对路径。多机部署时建议使用相同挂载点；不通过 HTTP 传输音频字节。

## 2. Python 环境

音乐分析服务使用：

```text
/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/moss-music-pipeline/bin/python
```

Section ASR 使用：

```text
/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/qwen3-vllm/bin/python
```

Omni 上游使用已验证的 vLLM 环境并独立启动。需要的系统组件包括 NVIDIA 驱动、CUDA、ffmpeg/ffprobe、VAMP/Chordino 依赖及 libsndfile。

## 3. 资源 profile

supervisor 支持：

```bash
bash manage_model_services.sh status all --profile 24
bash manage_model_services.sh status all --profile 48
bash manage_model_services.sh status all --profile 80
```

profile 是部署上限，不代表所有服务必须位于同一张卡。通过以下变量分配设备和配额：

```bash
export FAST_GATE_SERVICE_GPU=0
export DISCOGS_SERVICE_GPU=0
export SONGFORMER_SERVICE_GPU=1
export SECTION_ASR_SERVICE_GPU=2

export FAST_GATE_SERVICE_MEMORY_GIB=3
export DISCOGS_SERVICE_MEMORY_GIB=4
export SONGFORMER_SERVICE_MEMORY_GIB=10
export SECTION_ASR_SERVICE_MEMORY_GIB=24
```

生产配额应以目标机器实测峰值乘 1.2 为基准，并在每张 GPU 上额外保留至少 4 GiB。首次部署必须用代表性音频测量 idle/peak；profile 默认值只是安全配置入口，不能替代实测。

24 GiB 卡通常需要把 SongFormer、ASR 和 Omni 分到不同卡或不同节点。48 GiB 仍不能直接容纳项目使用的原始 BF16 Omni 30B；建议独立 2×48 GiB TP=2。80 GiB 可承载更大的单服务配额，但同卡常驻组合仍要按实际峰值验收。

## 4. 启动 Omni 上游与 proxy

在独立 Omni 节点启动 OpenAI-compatible vLLM，例如：

```bash
vllm serve /models/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 10008 \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --max-num-seqs 2 \
  --allowed-local-media-path /
```

在运行 pipeline API proxy 的节点：

```bash
export OMNI_UPSTREAM_SERVER=http://omni-node:10008
bash manage_model_services.sh start omni --profile 80
```

proxy 启动后仍会检查上游 `/v1/models`。上游不可用时 `/healthz` 为 503，runner 不会回退到本地加载 Omni。

## 5. 启停与状态

```bash
bash manage_model_services.sh start all --profile 80
bash manage_model_services.sh status all --profile 80 --json
bash manage_model_services.sh restart section-asr --profile 80
bash manage_model_services.sh stop all --profile 80
```

状态文件、PID 和默认日志位于 `/tmp/music-data-pipeline-model-services-$UID/`。可以用 `MODEL_SERVICE_STATE_DIR` 改到持久目录。PID 校验同时检查进程启动 ticks 与精确脚本 cmdline，避免误杀复用 PID 的其他进程。

每个服务只有在模型完全加载后才 ready。修改模型路径、版本、GPU 映射或关键 batching 配置后，应显式 `restart` 对应服务。

## 6. 服务 URL

本机默认端口：

```text
Fast Gate       127.0.0.1:18101
Discogs         127.0.0.1:18102
CPU MIR         127.0.0.1:18103
SongFormer      127.0.0.1:10101
Section ASR     127.0.0.1:10102
Omni proxy      127.0.0.1:10103
```

跨机器 runner 使用：

```bash
export FAST_GATE_SERVICE_URL=http://gate-host:18101
export DISCOGS_MIR_SERVICE_URL=http://discogs-host:18102
export MUSIC_CPU_SERVICE_URL=http://cpu-host:18103
export STRUCTURE_RAW_SERVICE_URL=http://songformer-host:10101
export SECTION_ASR_SERVICE_URL=http://asr-host:10102
export ALM_SERVICE_URL=http://omni-proxy-host:10103
```

## 7. 运行与恢复

```bash
bash run_pipeline.sh /data/raw /data/results/batch_001
bash run_stream_pipeline.sh /data/raw /data/results/stream_001
```

Batch runner 在开始时健康检查所有启用服务，并保留阶段屏障。Streaming runner 逐条推进，SQLite 位于 `intermediate/stream/state.sqlite3`。两者都不负责启动或停止模型服务。

同一输入和结果目录可直接重跑。成功且输入指纹匹配的阶段会复用；网络中断使用同一 request ID 重试。主动更换模型不会仅因 provenance 字段变化自动使缓存失效，需要显式清理对应 stage cache 或使用新结果目录。

## 8. 验收清单

部署后至少验证：

1. 每项 `/healthz` 为 ready，并记录 PID、模型指纹、加载时间、内存和队列深度。
2. 连续两批请求间 PID 和模型加载时间不变。
3. 队列满返回 429；相同 request ID 重试不重复推理；不同 payload 冲突返回 409。
4. GPU 服务没有静默 CPU fallback。
5. Streaming 的首条 annotation 在整批结束前出现；同一音频的 CPU MIR、SongFormer、Omni 有真实重叠。
6. Instrumental 从不调用 Section ASR；Song 只等待自己的 ASR。
7. 最终四分区完整互斥，annotation schema 为 v2，且没有已删除的 section 元数据。
8. 用代表性 100 条对比 service-backed batch 与 streaming；预热后 streaming 总耗时不比 batch 回退超过 20%。
