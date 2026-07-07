# 运行指南：vLLM & RTP-LLM 端到端 Bench

## 环境说明

### 机器 A：B300 8×L20D (SM 10.3, CUDA 13.x)

两个引擎共享同一台机器，共用 `/opt/conda310` Python 环境。

- **vLLM**：`pip install -e .` 安装，直接用系统 Python
- **RTP-LLM**：必须通过 `bazel test` 运行，bazel sandbox 自带依赖隔离，不受系统 pip 影响

**不要**用 `bazel run` 或手动 `python batch_decode_test.py` 跑 RTP-LLM，会加载系统环境里被 vLLM 改过的包导致失败。

### 机器 B：H20 (SM 9.0, CUDA 12.9)

Docker 镜像 `rtp_llm_dev_gpu_cuda12_9`，Python 在 `/opt/conda310/bin/python3`，torch 2.8.0+cu129 预装。

vLLM 安装步骤：
```bash
# 安装构建依赖
/opt/conda310/bin/pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" ninja packaging jinja2

# 安装 vLLM（editable mode，用镜像自带的 torch）
VLLM_USE_PRECOMPILED=1 /opt/conda310/bin/pip install -e . --no-build-isolation
```

H20 上的已知问题及 workaround：
- **FlashInfer GDN prefill JIT 编译失败**：镜像内 GCC 4.8.5 太旧，nvcc 编译 sm_90a CUDA 代码时 segfault。Workaround：在 harness 中设置 `additional_config={"gdn_prefill_backend": "triton"}`
- **minimax_m3 triton kernel 不兼容**：vLLM 拉入 triton 3.6.0 与 minimax_m3 代码不兼容。Workaround：`kernel_warmup.py` 中 minimax import 加 try/except
- **nvcc 不在 PATH**：EngineCore 子进程找不到 nvcc 导致 FlashInfer DeepGEMM cubin 编译失败。Workaround：`export PATH=/usr/local/cuda-12.9/bin:$PATH`
- **engine_core 属性变更**：V1 多进程模式下 `SyncMPClient` 没有 `engine_core` 属性。Workaround：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 使用 InprocClient

---

## vLLM Bench

### B300 (CUDA 13.x)

```bash
cd /tmp   # 不能在 /data2/liusongyue.lsy/ 或 vllm/ 下运行（sys.path 冲突）

export PATH=/opt/conda310/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # 走 InprocClient，直接访问 EngineCore

# Prefill
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /mnt/nas1/hf/Qwen3-8B \
  --mode prefill \
  --batch-sizes 1,4,16 \
  --seq-lens 128,512,1024 \
  --num-iters 5 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.1

# Decode
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /mnt/nas1/hf/Qwen3-8B \
  --mode decode \
  --batch-sizes 1,4,16 \
  --seq-lens 128,512,1024 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.1
```

需要的环境变量（编译时也需要，运行时可选）：
```bash
export CC=/data2/liusongyue.lsy/local/gcc12/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data2/liusongyue.lsy/local/gcc12/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=/data2/liusongyue.lsy/local/gcc12/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="10.3"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### H20 (CUDA 12.9)

```bash
export PATH=/usr/local/cuda-12.9/bin:/opt/conda310/bin:$PATH
export CUDA_VISIBLE_DEVICES=1              # GPU 0 可能被其他任务占用
export VLLM_ENABLE_V1_MULTIPROCESSING=0    # 走 InprocClient，直接访问 EngineCore

# Decode (Qwen3.5-35B-A3B-FP8, MoE 模型)
/opt/conda310/bin/python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3.5-35B-A3B-FP8/ \
  --mode decode \
  --batch-sizes 1,4,16 \
  --seq-lens 128,512,1024 \
  --num-iters 5 --num-decode-steps 20 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --tp-size 1
```

---

## RTP-LLM Bench

```bash
export CUDA_VISIBLE_DEVICES=0

BAZEL=/home/liusongyue.lsy/.cache/bazelisk/downloads/sha256/79e4f370efa6e31717b486af5d9efd95864d0ef13da138582224ac9b2a1bad86/bin/bazel

cd /data2/liusongyue.lsy/RTP-LLM/github-opensource

$BAZEL --output_user_root=~/.cache/bazel_cuda13_cache \
  test //rtp_llm/test/perf_test:grid_perf_test \
  --config=cuda13 --jobs=200 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_timeout=3600 --test_output=all
```

测试参数在 `rtp_llm/test/perf_test/BUILD` 的 `grid_perf_test` target 里改：
```python
args = [
    "--model_type", "qwen_3",           # qwen_3 / qwen_3_moe / deepseek_v32
    "--checkpoint_path", "/mnt/nas1/hf/Qwen3-8B",
    "--batch_size", "1",
    "--input_len", "128",
    "--partial", "0",                    # 0=both, 1=decode, 2=prefill
    "--decode_test_length", "8",
    "--seq_size_per_block", "64",        # 必须 64
    "--tp_size", "1",
    "--dp_size", "1",
]
```

Timeline 输出在 bazel testlogs 的 `test.outputs/timelines/` 下。

---

## GPU Profiling（nsys / torch.profiler）

### 方式 A：nsys + cudaProfilerApi（推荐，最完整）

用 nsys 包裹进程，`--profile` 开关控制 `cudaProfilerStart/Stop` 窗口，只采集 warmup 后的 N 步：

```bash
cd /tmp
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# 可选：开启 NVTX 标注看 vLLM 内部函数边界
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --output=/tmp/vllm_decode_bs1_seq128 \
  --force-overwrite=true \
  python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
    --model /mnt/nas1/hf/Qwen3-8B \
    --mode decode \
    --batch-sizes 1 --seq-lens 128 \
    --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
    --enforce-eager --max-model-len 2048 --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --profile --profile-steps 5
```

输出 `/tmp/vllm_decode_bs1_seq128.nsys-rep`，可用 `nsys stats` 查看或拉到本地用 Nsight Systems GUI 打开。

**注意事项**：
- `--trace=cuda,nvtx` 必须加，否则默认不采集 GPU kernel 数据
- `--cuda-graph-trace=node` 必须加，否则 CUDA Graph 内部 kernel（GEMM、attention、RMSNorm 等）不可见，只能看到 graph 外的 sampling kernel
- nsys 的 CUDA tracing 有额外显存开销，`--gpu-memory-utilization` 需要 ≥0.3（0.1 会 OOM）
- `--profile-steps N` 控制采集窗口大小：warmup 后的前 N 个 step 被 profiler 覆盖
- 查看 kernel 统计：`nsys stats xxx.nsys-rep --report cuda_gpu_kern_sum`
- 不加 `--enforce-eager` 时 vLLM 会自动使用 CUDA Graph（decode 延迟 ~4ms vs eager ~12ms）

### 方式 B：torch.profiler Kineto trace（无需 nsys）

直接生成 Chrome Trace JSON，和 RTP-LLM 的 `gen_timeline` 输出格式兼容：

```bash
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /mnt/nas1/hf/Qwen3-8B \
  --mode decode \
  --batch-sizes 1 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.1 \
  --profile --profile-mode torch --profile-output /tmp/traces
```

输出 `/tmp/traces/vllm_decode_bs1_seq128_steps10.json`，用 `chrome://tracing` 或 Perfetto 查看。
trace 名里编码了 `mode/bs/seq/steps`，分析器据此还原每步平均耗时。

### 方式 C：逐组件耗时分解 & 与 RTP 对齐（perf_test_timeline）

`--profile --profile-mode torch` 会走一次**专用 profiling pass**（warmup → 只抓 decode
步，prefill 在窗口外），产出干净的 chrome trace。加 `--analyze` 直接打印**按组件分类的
GPU 耗时**（Attention / MoE GEMM / Dense GEMM / Norm / RoPE / Activation / Sampling /
Comm …，分类法与 RTP `analyze_timeline.py` 对齐）。

```bash
# CUDA graph 下的逐组件 GPU 耗时（分类靠 kernel 名，CUDA graph 也准）
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-8B/ --mode decode \
  --batch-sizes 4 --seq-lens 128 --num-iters 3 --num-decode-steps 10 \
  --max-model-len 4096 --dtype bfloat16 --gpu-memory-utilization 0.6 \
  --profile --profile-mode torch --profile-output /tmp/traces --analyze
```

**VL 模型（如 Qwen3.5-35B-A3B-FP8）需加 `--disable-mm`**：把多模态槽位清零,让引擎跳过
视觉塔的显存 profiling,只跑语言模型 decode（和 RTP 的 text 基准对齐）。否则会在 vision
dummy batch 处崩。

**单独分析 / 两引擎对比**（analyzer 可独立跑）：

```bash
# 单个 trace 的分类分解
python -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  /tmp/traces/vllm_decode_bs4_seq128_steps10.json

# vLLM ↔ RTP 逐组件每步 diff（RTP trace 见下方 RTP Profiling 对照）
python -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  --compare vllm.json rtp.json --labels vLLM RTP-LLM
# 或在跑 vLLM 时直接对比：给 runner 加 --rtp-trace rtp.json
```

说明：CUDA graph + torch.compile 会把 RoPE/Norm/残差融进匿名 `triton_*_fused` kernel，
归到 **Fused (compile)** 桶；要看清 RoPE/Norm/Activation 用 `--enforce-eager`（禁融合，
kernel 名恢复语义）。

### RTP-LLM Profiling 对照

RTP-LLM 内建 Kineto profiler，perf test 自动走 3 轮（warmup → measure → profile），timeline 输出在 `TEST_UNDECLARED_OUTPUTS_DIR/timelines/`。用 BUILD 文件的 env 控制：
- `GEN_TIMELINE_SYNC=1`：同步 timeline
- `PERF_PREARM_PROFILE=1`：预配置 profiler
- `PERF_PROFILE_NUM_STEPS=4`：采集步数

---

## 多 DP Bench

harness 支持 `--dp-size N`，自动启动 N 个进程，每个 rank 分到 `batch_size // dp_size` 个请求。

- **Dense 模型**：各 rank 通过 `CUDA_VISIBLE_DEVICES` 隔离 GPU，完全独立运行
- **MoE 模型（不开 EP）**：同 Dense，各 rank 独立加载全部 experts（TP 拆分）
- **MoE 模型（开 EP）**：目前不支持跨 DP rank 的 EP（需要共享 distributed world），`--enable-expert-parallel` 仅在单 rank 或同 world 内生效

### H20 DP=2 示例 (Qwen3-235B-A22B, TP=4)

```bash
cd /tmp
export PATH=/usr/local/cuda-12.9/bin:/opt/conda310/bin:$PATH

# BS=128, decode 30 步
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/muxue.xy/Qwen3-235B-A22B-Instruct-2507-FP8/ \
  --mode decode \
  --batch-sizes 128 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 30 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 512 --dtype auto \
  --gpu-memory-utilization 0.9 \
  --tp-size 4 --dp-size 2

# BS=1024, decode 30 步（需降低 max-model-len 避免 profile OOM）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/muxue.xy/Qwen3-235B-A22B-Instruct-2507-FP8/ \
  --mode decode \
  --batch-sizes 1024 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 30 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 256 --dtype auto \
  --gpu-memory-utilization 0.9 \
  --tp-size 4 --dp-size 2
```

**注意事项**：
- `batch_size` 必须 ≥ `dp_size` 且能整除
- TP=4 DP=2 需要 8 张 GPU，通过 `CUDA_VISIBLE_DEVICES` 指定
- 不需要设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，harness 内部自动设置
- 235B 模型至少需要 TP=4 才能放进单个 DP rank 的显存（FP8 ~60GB/GPU）
- `max_model_len` 影响 profile_run 的显存分配（`max_num_batched_tokens = local_bs × max_model_len`），大 BS 时需要降低

### 已知限制

- **不支持跨 DP rank EP**：每个 DP rank 进程有独立的 torch.distributed world，无法形成跨 rank 的 NCCL EP group。MoE 大模型需要通过 TP 拆分 experts，而非 EP
- **TP > 1 + DP + EP**：MultiProcExecutor 为每个 DP rank 创建独立的 distributed init，无法支持跨 DP 的 all-to-all

---

## 已验证的对比结果

### Qwen3-8B, BF16, single L20D (B300)

| 引擎 | 模式 | BS | SeqLen | 延迟 (ms) |
|---|---|---|---|---|
| RTP-LLM | decode | 1 | 128 | 13.78 |
| vLLM | decode | 1 | 128 | 13.73 (p50) |
| vLLM | prefill | 1 | 128 | 14.53 (p50) |

### Qwen3.5-35B-A3B-FP8, single H20

| 模式 | BS | SeqLen | step p50(ms) | decode/tok(ms) | prefill(ms) |
|---|---|---|---|---|---|
| decode | 1 | 128 | 85.68 | 85.90 | 117.00 |
| decode | 1 | 512 | 85.38 | 86.44 | 110.94 |
| decode | 1 | 1024 | 84.96 | 85.03 | 110.43 |
| decode | 4 | 128 | 87.04 | 87.08 | 111.26 |
| decode | 4 | 512 | 86.91 | 86.94 | 110.92 |
| decode | 4 | 1024 | 86.93 | 86.95 | 173.15 |
| decode | 16 | 128 | 88.23 | 88.33 | 111.49 |
| decode | 16 | 512 | 88.06 | 88.19 | 325.07 |
| decode | 16 | 1024 | 87.90 | 88.06 | 626.79 |

### Qwen3-235B-A22B-FP8, DP=2 TP=4, H20 8×GPU

| BS (global) | SeqLen | Rank 0 step p50(ms) | Rank 1 step p50(ms) |
|---|---|---|---|
| 128 | 128 | 123.75 | 126.19 |
| 1024 | 128 | 127.74 | 126.45 |

各 rank per-step 延迟接近（差异 <3%）。
