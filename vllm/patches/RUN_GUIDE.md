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

## GPU Profiling（torch.profiler / WorkerProfiler）

三种抓 timeline 的方式，按场景选：
- **方式 A — torch.profiler**：进程内自采集，零引擎配置、依赖最稳定（只用 `torch.profiler`）。
  TP=1 本地快速分析首选。
- **方式 B — WorkerProfiler**：vLLM 自带，经 `collective_rpc` 分发到每个 rank，各自出 trace。
  **TP>1 / DP / EP 场景唯一可用**。
- **方式 C — perf_test_timeline**：分析器，把 A/B 产出的 chrome trace 按组件分类、与 RTP 对比。

> 注：旧的 nsys（`--profile-mode nsys` + `cudaProfilerStart`）后端已删除——它和 WorkerProfiler
> 的 cuda 后端底层相同（`cudaProfilerStart/Stop`），后者是超集（多 rank + NVTX）。要 nsys
> 系统级 timeline，直接用 `nsys profile --capture-range=none` 全程采集即可。

### 方式 A：torch.profiler（轻量，TP=1）

进程内自采集，无需外部 wrapper，零引擎配置，直接生成 Chrome Trace JSON（和 RTP-LLM
`gen_timeline` 格式兼容）：

```bash
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /mnt/nas1/hf/Qwen3-8B \
  --mode decode \
  --batch-sizes 1 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.1 \
  --profile --profile-output /tmp/traces
```

输出 `/tmp/traces/vllm_decode_bs1_seq128_steps10.json`，用 `chrome://tracing` 或 Perfetto 查看。
trace 名里编码了 `mode/bs/seq/steps`，`perf_test_timeline` 据此还原每步平均耗时。

**限制**：单进程,只能抓 driver 进程的前向。**TP>1 时前向在 worker 子进程,抓不到**——用方式 B。

### 方式 B：WorkerProfiler（多 rank / TP>1 / DP+EP）

vLLM 自带的 profiler，经 `collective_rpc` 分发到**每个 TP/DP rank**，各自 dump 一份 trace。
**TP>1 / DP / EP 场景唯一可用**（方式 A 的单进程 torch.profiler 看不到 worker 子进程）。
`--worker-profile-dir` 触发：harness 在第一个被测轮的 decode 循环前后调
`llm.start_profile()` / `stop_profile()`，采集窗口正好是 `num-decode-steps` 步。

底层就是 `ProfilerConfig(profiler="torch")` + `LLM.start_profile/stop_profile`，和方式 A 同一个
`torch.profiler` 内核，只是多了 per-rank fanout 和每步 NVTX 标注（trace 里的 `execute_context_N`）。

```bash
# Qwen3-235B-A22B-FP8, TP=2 DP=2 EP, decode 30 步 (H20 × 4)
export PATH=/usr/local/cuda-12.9/bin:/opt/conda310/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda310/bin/python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/muxue.xy/Qwen3-235B-A22B-Instruct-2507-FP8/ \
  --mode decode --batch-sizes 4 --seq-lens 128 \
  --num-iters 2 --num-decode-steps 30 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 512 --dtype auto \
  --gpu-memory-utilization 0.9 \
  --tp-size 2 --dp-size 2 --enable-expert-parallel \
  --worker-profile-dir /tmp/wp_trace
```

产出每 rank 一份：`/tmp/wp_trace/dp{0,1}_tp{0,1}_ep{0..3}_rank*.pt.trace.json.gz`
（外加每 rank 一份 `profiler_out_N.txt` 的 CUDA-time kernel 表）。

**坑点**：
- 输出是 **gzip** 且文件名是 rank 后缀（不是方式 A 的 `vllm_..._steps{N}.json` 约定）。
  要喂 `perf_test_timeline` 得先转：
  `zcat xxx.pt.trace.json.gz > vllm_decode_bs4_seq128_steps30.json`，
  再 `perf_test_timeline vllm_decode_bs4_seq128_steps30.json --steps 30`。
- 单份 trace 很大（30 步 235B ≈ 100MB/rank）。
- TP=2 不开 EP 会 OOM（235GB / 2 > 单卡 97GB）；开 `--enable-expert-parallel` 把 experts 按
  EP=TP×DP 切开才装得下（实测每卡 ~61GB）。
- 需在构造引擎时就带 `profiler_config`（harness 靠 `--worker-profile-dir` 自动完成；直接调
  `worker.profile()` 而没配 `profiler_config` 会抛 `RuntimeError`）。

### 方式 C：逐组件耗时分解 & 与 RTP 对齐（perf_test_timeline）

`--profile` 会走一次**专用 profiling pass**（warmup → 只抓 decode
步，prefill 在窗口外），产出干净的 chrome trace。加 `--analyze` 直接打印**按组件分类的
GPU 耗时**（Attention / MoE GEMM / Dense GEMM / Norm / RoPE / Activation / Sampling /
Comm …，分类法与 RTP `analyze_timeline.py` 对齐）。

```bash
# CUDA graph 下的逐组件 GPU 耗时（分类靠 kernel 名，CUDA graph 也准）
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-8B/ --mode decode \
  --batch-sizes 4 --seq-lens 128 --num-iters 3 --num-decode-steps 10 \
  --max-model-len 4096 --dtype bfloat16 --gpu-memory-utilization 0.6 \
  --profile --profile-output /tmp/traces --analyze
```

**VL 模型（如 Qwen3.5-35B-A3B-FP8）需加 `--disable-mm`**：把多模态槽位清零,让引擎跳过
视觉塔的显存 profiling,只跑语言模型 decode（和 RTP 的 text 基准对齐）。否则会在 vision
dummy batch 处崩。

**vLLM 自带引擎阶段 scope（对齐 RTP 的 executor.model_forward / sampler_forward）**：加
`--vllm-scopes`（需配 `--enforce-eager`）。它会在 torch trace 里输出
`gpu_model_runner: forward / preprocess / sample / postprocess / bookkeep` 和
`schedule: ...` 这些 `user_annotation`，`--analyze` 的 "Semantic scopes" 表会列出来。

坑点（务必知道）：`gpu_model_runner:` 这些 scope **只存在于 vLLM 的 legacy V1 model
runner**（`vllm/v1/worker/gpu_model_runner.py`）。新的 **V2 runner**
（`vllm/v1/worker/gpu/model_runner.py`，对 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` 里的
Qwen3/Llama/Mistral/DeepseekV2/... 是默认）**完全没有 record_function scope**——所以直接开
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 对这些模型只会看到 `schedule:` scope，看不到前向。
`--vllm-scopes` 因此**同时设 `VLLM_USE_V2_MODEL_RUNNER=0`** 强制走 V1 scoped runner。
注意这会切换到 legacy 执行路径（时延不代表 V2 部署路径,仅用于 scope 语义对齐）。
scope 只在 eager 命中,CUDA graph replay 下不触发,时长是 CPU wall（GPU 归因看 kernel 分类表）。

```bash
python -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/muxue.xy/Qwen3-32B --mode decode \
  --batch-sizes 128 --seq-lens 128 --num-iters 3 --num-decode-steps 30 \
  --max-model-len 256 --gpu-memory-utilization 0.8 \
  --tp-size 1 --enforce-eager --vllm-scopes \
  --profile --profile-output /tmp/traces --analyze
```
（`--vllm-scopes` 需要模型在 **单进程**内可见,即 TP=1；TP>1 时前向在 worker 子进程,
harness 主进程的 torch.profiler 抓不到——改用**方式 B 的 `--worker-profile-dir`**,它经
`collective_rpc` 到每个 rank 各自抓 trace。）

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

### Scope 三层观测能力（重要参考）

推理性能观测分三层，从粗到细、从 CPU 到 GPU。**层一按引擎手埋；层二 / 层三是
PyTorch / CUDA 自动，RTP 与 vLLM 机制一样。**

#### 层一：引擎阶段 scope（手埋）

- vLLM 用 `record_function_or_nullcontext("...")` 埋，由 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`
  开；`--vllm-scopes` 会自动设它（并强制 V1，见下）。
- 覆盖：`gpu_model_runner: forward/preprocess/postprocess/sample/bookkeep/eplb/draft/...`、
  `schedule: allocate_slots/...`、`llm_engine step: ...`、`ngram_proposer_gpu: kernel`。
- 性质：**CPU 墙钟、引擎阶段级**（forward 是一整块，不下沉到 attn/moe/gemm）。
- 与 RTP 对齐（引擎阶段级）：

  | 阶段 | vLLM V1 | RTP-LLM |
  |---|---|---|
  | 前向 | `gpu_model_runner: forward` | `executor.model_forward`(=py_model.forward) |
  | 采样 | `gpu_model_runner: sample` | `executor.sampler_forward` |
  | 输入准备 | `gpu_model_runner: preprocess` | `executor.gather_model_input` |
  | 输出 | `postprocess`/`ModelRunnerOutput` | `executor.dispatch_output` |
  | 调度 | `schedule: allocate_slots` | 埋在 `engine.normal.execute` 外层 |

- **失效 / 降级条件**：
  - **V2 runner**（`gpu/model_runner.py`，Qwen3/Llama/Mistral/DeepseekV2/Qwen2Moe 等默认，
    见 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`）**没埋任何 scope** → `--vllm-scopes` 会强制
    `VLLM_USE_V2_MODEL_RUNNER=0` 回到有 scope 的 V1（注意 V1 是 legacy 路径，时延不代表 V2 部署）。
  - **TP>1**：前向在 worker 子进程，harness 主进程 profiler 抓不到。
  - **CUDA graph**：引擎 scope 仍触发，但 `forward` 只量到"启动 graph"的 ~1–2ms
    （实测 121ms→1.9ms），失去前向计算意义（不是消失，是降级）。
  - **DP + EP（TP=1）+ eager 可用**：每个 DP rank 是独立进程、TP=1 → 模型在进程内可见；
    需给 trace 加 rank 后缀避免同名覆盖。⚠️ 尚未实测验证。

#### 层二：算子框架 scope（cpu_op，PyTorch/Kineto 自动）

- `torch.profiler` 采 CPU activity 时，RecordFunction 在**每次算子 dispatch** 自动记一条
  `cat="cpu_op"` 的 range（含 dispatch + launch 的 **CPU** 时间，不是 GPU 时间）。
- name 三种来源：
  - aten 内置：`aten::linear/matmul/mm`（GEMM 三层，**嵌套**，`linear⊃matmul⊃mm` 常同次数，
    直接相加会重复计数）、`aten::copy_/to/reshape/slice/empty`（拷贝/视图/分配）；
  - custom op：**vLLM 注册成功能名**（`unified_attention_with_output`/`moe_forward_shared`/
    `fused_add_rms_norm`/`silu_and_mul`/`unified_kv_cache_update`）；**RTP 多为通用运算名，
    或缺席**（手写 C++ 融合不走 dispatcher）；
  - autograd Function 名（如 RTP GDN 的 `FusedRecurrentFunction`）。
- 进阶：`ac2g`（correlation id）关联到 GPU kernel 时间；`record_shapes` 拿算子尺寸算带宽利用。
- 跨引擎：**裸 aten / GEMM 可比**；但 **vLLM 组件有功能名可归因，RTP 的 attn/norm/激活是手写
  C++ 融合、在 cpu_op 里隐身 → 组件归因对不齐**（RTP 组件归因要退回层三 kernel 名分类）。
- **失效条件**：
  - **RTP C++ 融合算子**：cpu_op 里没有对应条目（隐身）；
  - **TP>1**：同层一，worker 子进程抓不到；
  - **CUDA graph**：图内算子 replay 不再 dispatch、没有 per-op 启动开销 → **cpu_op 消失**。
    （这正是 CG 的目的——消掉 per-op 启动开销，所以 cpu_op 无东西可记。）

#### 层三：算子 GPU 时间（kernel，CUPTI 自动）

- `torch.profiler` 加 `ProfilerActivity.CUDA` → Kineto 用 CUPTI 在 `cudaLaunchKernel` 处建立
  关联，记录 kernel **稍后在 GPU 上执行的真实时间**，`cat="kernel"`。RTP 与 vLLM 完全一致。
- **穿透 CUDA graph**：CG 下层一降级、层二消失，只有它还在。
- **TP>1** 靠各引擎自己的 per-rank profiling（RTP `gen_timeline` / nsys / vLLM 原生 profiler）仍可拿到。
- **谁发的 kernel 都抓**（含 RTP 手写 kernel）→ **跨引擎组件归因的唯一可靠层**，靠 kernel 名分类
  （见方式 C 的 `perf_test_timeline`）。
- 需手动精确计时某段可用 `torch.cuda.Event`。

#### 一句话

越往下越细、越接近真实 GPU 成本、越跨引擎可比：**层一/层二是 CPU 墙钟，CG 下失效或降级；
部署路径（TP>1 + CUDA graph）下只有层三（kernel 名分类）可靠。**

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

# BS=1024, decode 30 步（profile 激活 ~ local_bs×seq = 512×128；大到吃紧就减 batch/seq）
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
- profile_run 的激活显存由 `max_num_batched_tokens = max(local_bs × max(seq_lens), max_model_len)` 决定（harness 用 grid 里最大的 seq_len 而非 max_model_len 作上界，避免过度预留挤占 KV / profile OOM）。所以现在 `max_model_len` 只需 ≥ `seq_len + num_decode_steps`，不必再为了 profile 显存去压它；真正撑爆 profile 的是 `local_bs × max(seq_lens)`，大 BS + 长 seq 时才需要减小 batch 或 seq

### 已知限制

- **不支持跨 DP rank EP**：每个 DP rank 进程有独立的 torch.distributed world，无法形成跨 rank 的 NCCL EP group。MoE 大模型需要通过 TP 拆分 experts，而非 EP
- **TP > 1 + DP + EP**：MultiProcExecutor 为每个 DP rank 创建独立的 distributed init，无法支持跨 DP 的 all-to-all

---

## 已验证的对比结果

> 所有 vLLM 延迟均为 **trimmed mean**（按轮排序、丢掉最小和最大、其余取平均，见
> `_trimmed_mean`），对齐 RTP-LLM `batch_perf_impl.run` 的 `measurements[1:-1]`；
> 不是 p50。

### Qwen3-8B, BF16, single L20D (B300)

| 引擎 | 模式 | BS | SeqLen | 延迟 (ms) |
|---|---|---|---|---|
| RTP-LLM | decode | 1 | 128 | 13.78 |
| vLLM | decode | 1 | 128 | 13.73 (trimmed mean) |
| vLLM | prefill | 1 | 128 | 14.53 (trimmed mean) |

### Qwen3.5-35B-A3B-FP8, single H20

| 模式 | BS | SeqLen | step trimmed(ms) | decode/tok(ms) | prefill(ms) |
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

| BS (global) | SeqLen | Rank 0 step trimmed(ms) | Rank 1 step trimmed(ms) |
|---|---|---|---|
| 128 | 128 | 123.75 | 126.19 |
| 1024 | 128 | 127.74 | 126.45 |

各 rank per-step 延迟接近（差异 <3%）。
