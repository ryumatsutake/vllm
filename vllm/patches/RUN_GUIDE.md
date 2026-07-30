# vLLM Perf Patch 使用指南

`vllm/patches/batch_decode_scheduler/` 是一套 GPU 端到端性能基准工具，用于把 vLLM 的
prefill / decode 延迟与 RTP-LLM `grid_perf_test` 做**同口径对比**。包含三个模块：

| 模块 | 作用 |
|---|---|
| `perf_test_harness.py` | 包装 EngineCore，按 schedule → execute → update 驱动单步执行并计时；提供 fake-KV、scope 注入等能力 |
| `perf_test_runner.py` | CLI 入口：跑 batch_size × seq_len 网格，聚合输出表格 / CSV；支持 TP / PP / DP / EP |
| `perf_test_timeline.py` | Chrome trace 分析器：按组件（Attention / MoE / GEMM / …）归类 GPU kernel 耗时，可与 RTP trace 逐组件对比 |

**计时口径**（与 RTP-LLM 对齐，全文适用）：

- 所有延迟为 **trimmed mean**：按轮排序、丢掉最小和最大、其余取平均（≥3 轮时），
  对齐 RTP `batch_perf_impl.run` 的 `measurements[1:-1]`，不是 p50
- decode 的 `per_token = (cost − prefill) / num_decode_steps`，对齐 RTP `decode_time_per_token`
- prefill 网格里 `seq_len` 是输入长度；decode 网格里 `seq_len` 是 KV 长度（kv_len）
- 每步带相位断言：batch 被拆分 / prefill 被截断会**直接报错**而非产出错误数据

---

## 1. 环境准备

### 机器 A：B300 8×L20D（SM 10.3, CUDA 13.x）

vLLM 与 RTP-LLM 共享 `/opt/conda310` Python 环境。

- vLLM：`pip install -e .`，直接用系统 Python
- RTP-LLM：必须 `bazel test` 运行（sandbox 隔离依赖）。**不要**用 `bazel run` 或手动
  `python batch_decode_test.py`，会加载被 vLLM 改过的系统包导致失败

编译 / 运行需要的环境变量：

```bash
export CC=/data2/liusongyue.lsy/local/gcc12/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data2/liusongyue.lsy/local/gcc12/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=/data2/liusongyue.lsy/local/gcc12/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="10.3"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 机器 B：H20（SM 9.0, CUDA 12.9）

Docker 镜像 `rtp_llm_dev_gpu_cuda12_9`，Python 在 `/opt/conda310/bin/python3`，
torch 2.8.0+cu129 预装。本地模型在 `/home/models/` 下。

```bash
# 安装
/opt/conda310/bin/pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
  "setuptools-rust>=1.9.0" ninja packaging jinja2
VLLM_USE_PRECOMPILED=1 /opt/conda310/bin/pip install -e . --no-build-isolation

# 每次运行前
export PATH=/usr/local/cuda-12.9/bin:/opt/conda310/bin:$PATH
```

H20 已知问题与 workaround（harness 已内置处理的标注为「自动」）：

| 问题 | 处理 |
|---|---|
| FlashInfer GDN prefill JIT 失败（GCC 4.8.5 太旧） | 自动：harness 设 `additional_config={"gdn_prefill_backend": "triton"}` |
| `SyncMPClient` 无 `engine_core` 属性 | 自动：runner 设 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 走 InprocClient |
| EngineCore 子进程找不到 nvcc | 手动：`export PATH=/usr/local/cuda-12.9/bin:$PATH` |
| minimax_m3 triton kernel 不兼容 | 已在 `kernel_warmup.py` 加 try/except |

### 通用注意

- 在 `/tmp` 等目录下运行，**不要**在 vllm 源码目录下运行（sys.path 冲突）
- 模型必须传本地有效路径（如 `/home/models/Qwen3-8B`）；路径不存在时 transformers
  会把它当 HF repo id 报 `HFValidationError`

---

## 2. 快速开始

```bash
cd /tmp

# Prefill：测首 token 延迟（--partial 2）
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-8B \
  --partial 2 \
  --batch-sizes 1,4,16 --seq-lens 128,512,1024 \
  --num-iters 5 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.6

# Decode：PD 路径（--partial 0，真实 prefill 铺 KV 后跑 N 步 decode，两个指标都出）；
# 严格对齐 RTP 的 decode-only 用 --partial 1（fake-KV，见 §5）
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-8B \
  --partial 0 \
  --batch-sizes 1,4,16 --seq-lens 128,512,1024 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 2048 --dtype bfloat16 \
  --gpu-memory-utilization 0.6
```

输出为表格；加 `--output result.csv` 同时写 CSV。

## 3. 常用参数

| 参数 | 说明 |
|---|---|
| `--partial {0,1,2}` | RTP 同语义模式开关：0=PD（真实 prefill+decode），1=只 decode（fake-KV，默认，见 §5），2=只 prefill |
| `--batch-sizes` / `--seq-lens` | 逗号分隔的网格；decode 时 seq-lens 是 kv_len；DP 模式下 batch-sizes 是每 rank 口径（见 §4.2） |
| `--num-iters` / `--num-warmup-iters` | 每格测量轮数 / 预热轮数（预热不计入） |
| `--num-decode-steps` | decode 模式每轮的 decode 步数 |
| `--max-model-len` | 需 ≥ `max(seq_lens) + num_decode_steps`；不必为省显存刻意压小（见下） |
| `--gpu-memory-utilization` | KV + 激活显存上限比例 |
| `--enforce-eager` | 关 CUDA graph / torch.compile；要看清 kernel 语义名或用 scope 时必开 |
| `--tp-size` / `--pp-size` / `--dp-size` | 并行配置（见 §4） |
| `--enable-expert-parallel` | MoE 开 EP（配合 `--dp-size` 见 §4.2） |
| `--disable-fake-balance-expert` | 关闭默认的 RTP 对齐 fake-balance 路由，使用真实 gating；用于 A/B 对照 |
| `--disable-mm` | VL 模型只测语言部分：清零多模态槽位，跳过视觉塔显存 profiling |
| `--output` | CSV 输出路径 |

**显存要点**：token 预算 `max_num_batched_tokens = max(max_bs × max(seq_lens), max_model_len)`。
profile 阶段的激活峰值由它决定，所以真正撑爆显存的是 `bs × max(seq_lens)`，大 batch +
长 seq 时减 batch 或 seq，而不是压 `max_model_len`。

---

## 4. 并行模式

### 4.1 TP / PP

`--tp-size N` 即可；TP>1 时 vLLM 用 MultiprocExecutor（worker 在子进程），harness 的
fake-KV、scope 注入均通过 `collective_rpc` 广播到每个 rank，功能不受影响。唯一区别是
profiling 要用 `--worker-profile-dir`（见 §6）。

### 4.2 DP（`--dp-size N`）

runner 启动 N 个进程。`--batch-sizes` 是**每 DP rank** 的 batch size（与 RTP-LLM
GridRunner 同语义，总请求数 = bs × dp_size），每个 rank 直接跑 bs 个请求。
两种模式自动选择：

- **独立 DP**（Dense 模型，或 MoE 不开 EP）：各 rank 用 `CUDA_VISIBLE_DEVICES` 隔离
  GPU，完全独立运行
- **跨 DP EP**（MoE + `--enable-expert-parallel`）：通过 `VLLM_DP_RANK/SIZE/MASTER_*`
  环境变量组成共享 world，experts 按 EP=TP×DP 切分、all-to-all 跨 rank 通信；engine
  init 各阶段有 barrier 同步。已实测 Qwen3-235B TP4/DP2、122B TP2/DP2 与 TP4/DP2，
  双 rank 结果 spread < 1%

```bash
# Qwen3-235B-A22B-FP8, TP=4 × DP=2 EP（8×H20）；每 rank 64 请求，全局 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --partial 0 --batch-sizes 64 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 30 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 512 --gpu-memory-utilization 0.9 \
  --tp-size 4 --dp-size 2 --enable-expert-parallel
```

注意：

- 与 RTP 对比时 batch_size 可直接对齐（两边都是 per-rank 口径）
- 表格 / CSV 只输出 rank 0（RTP 是全 rank 平均）；runner 会对 rank 间 >10% 的差异打
  WARNING，任一 rank 失败 / 缺结果 / 退出码非零则整体报错退出
- 235B FP8 至少 TP=4 才能放进单 DP rank（~60GB/GPU）；TP=2 需开 EP 才装得下

## 5. fake-KV decode（`--partial 1`，对齐 RTP，实验功能）

跳过 prefill forward：KV block 照常分配、请求经 `collective_rpc` 注册到**每个 TP rank**
的 model_runner（并写入伪 token，V1 runner 写 `token_ids_cpu`，V2 写
`last_sampled_tokens`），但 KV 内容为全零，对齐 RTP `setIsContextStream(false)`。
**TP=1 和 TP>1 均可用**，适用于真实 prefill 会 OOM 或太慢的大 BS × 长 kv_len 场景。

```bash
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen2.5-0.5B-Instruct --partial 1 \
  --batch-sizes 4 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --enforce-eager --max-model-len 512 --gpu-memory-utilization 0.5 \
  --tp-size 2
```

限制：attention 读全零 KV，数值与真实 prefill 不同——**仅用于 kernel 计时对齐**，不用于
精度相关对比。（已用 Qwen2.5-0.5B 在 TP=1/TP=2 验证，per_token 与真实 prefill 同量级。）

---

## 6. Profiling

三种抓 timeline 的方式：

| 方式 | 开关 | 适用 | 说明 |
|---|---|---|---|
| A. torch.profiler | `--profile [--profile-output DIR]` | TP=1 | driver 进程内自采集，专用 profiling pass（prefill 在窗口外，trace 恰好 N 步 decode），输出 `vllm_<mode>_bs<B>_seq<S>_steps<N>.json` |
| B. WorkerProfiler | `--worker-profile-dir DIR` | **TP>1 / DP / EP 唯一可用** | 经 `collective_rpc` 每个 rank 各 dump 一份；采集窗口是第一个测量轮的 decode 循环 |
| C. timeline 分析器 | `--analyze` / 独立运行 | — | 把 A/B 的 chrome trace 按组件分类、与 RTP 对比（见 §7） |

方式 B 注意：输出是 **gzip**、文件名带 rank 后缀，喂分析器前先转换：
`zcat xxx.pt.trace.json.gz > vllm_decode_bs4_seq128_steps30.json`（或用 `--steps` 指定步数）。
单份 trace 较大（30 步 235B ≈ 100MB/rank）。

### 引擎阶段 scope（可选，配合 A/B）

- `--vllm-scopes`：在 trace 里输出 `gpu_model_runner: preprocess / sample /
  postprocess / ModelRunnerOutput` 等 `user_annotation`，与 RTP 的
  `executor.model_forward / sampler_forward / gather_model_input / dispatch_output`
  对位。V2 runner 原生无 scope，harness 经 `collective_rpc` 注入到每个 worker
  （保持真实 V2 路径，TP>1 每 rank 都有）
- `--scope-forward`：额外套 `gpu_model_runner: forward`。CUDA graph 下只量到 host
  启动时间，要有意义的 forward 墙钟配 `--enforce-eager`
- `--vllm-scopes-v1`：逃生开关，强制回退 legacy V1 runner 用其原生 scope

scope 是 **CPU 墙钟**，用于阶段级语义对照；GPU 归因始终以 kernel 分类表为准
（kernel 分类穿透 CUDA graph，是跨引擎对比唯一可靠的层）。V1/V2 的
`preprocess/sample/postprocess` 名字对齐但 span 不同，绝对值不可直接比。

```bash
# TP=1：driver 抓 trace + 打印分类表和 scope 表
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3-8B --partial 0 \
  --batch-sizes 4 --seq-lens 128 --num-iters 2 --num-decode-steps 10 \
  --enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.6 \
  --vllm-scopes --scope-forward \
  --profile --profile-output /tmp/traces --analyze

# TP>1：scope 注入到 worker，用 WorkerProfiler 每 rank 各抓一份
python3 -u -m vllm.patches.batch_decode_scheduler.perf_test_runner \
  --model /home/models/Qwen3.5-35B-A3B --partial 0 --disable-mm \
  --batch-sizes 1 --seq-lens 128 --num-iters 2 --num-decode-steps 30 \
  --max-model-len 2048 --gpu-memory-utilization 0.9 \
  --tp-size 2 --vllm-scopes --worker-profile-dir /tmp/wp_trace
```

---

## 7. Timeline 分析与 RTP 对比

```bash
# 单个 trace 分类分解（steps 从 vllm_* 文件名自动解析，可 --steps 覆盖）
python3 -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  /tmp/traces/vllm_decode_bs4_seq128_steps10.json

# vLLM ↔ RTP 逐组件每步 diff。
# ⚠️ RTP trace 文件名不符合 vllm_* 命名，必须 --steps-b 指定其 decode 步数，
# 否则回落为 1、RTP 侧 per-step 值被放大真实步数倍（会打 WARNING）。
# RTP 的 profiling 轮只采 min(decode_test_length, 3) 步（batch_perf_impl.py
# 的 profile_step），所以 decode trace 通常是 --steps-b 3，prefill 是 1
python3 -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  --compare vllm.json rtp.json --labels vLLM RTP-LLM --steps 10 --steps-b 3

# 或在跑 vLLM 时直接对比：
#   runner 加 --rtp-trace rtp.json --rtp-trace-steps 3
```

说明：

- 分类法与 RTP `analyze_timeline.py` 对齐（Attention / MLA / MoE GEMM / MoE Routing /
  MoE Communication / Dense GEMM / Norm / RoPE / Sampling / Communication / …）
- CUDA graph + torch.compile 会把 RoPE/Norm/残差融进匿名 `triton_*_fused` kernel
  （归入 **Fused (compile)** 桶）；要看清各组件用 `--enforce-eager`
- `Other` 占比 >5% 会提示补充分类模式

### RTP-LLM 侧的跑法

```bash
BAZEL=/home/liusongyue.lsy/.cache/bazelisk/downloads/sha256/79e4f370efa6e31717b486af5d9efd95864d0ef13da138582224ac9b2a1bad86/bin/bazel
cd /data2/liusongyue.lsy/RTP-LLM/github-opensource

$BAZEL --output_user_root=~/.cache/bazel_cuda13_cache \
  test //rtp_llm/test/perf_test:grid_perf_test \
  --config=cuda13 --jobs=200 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_timeout=3600 --test_output=all
```

参数在 `rtp_llm/test/perf_test/BUILD` 的 `grid_perf_test` target 里改
（`--model_type / --checkpoint_path / --batch_size / --input_len / --partial /
--decode_test_length / --tp_size / --dp_size`；`--seq_size_per_block` 必须 64）。
timeline 在 bazel testlogs 的 `test.outputs/timelines/` 下；BUILD env 里
`GEN_TIMELINE_SYNC=1` / `PERF_PREARM_PROFILE=1` / `PERF_PROFILE_NUM_STEPS=N`
控制采集。

---

## 8. 已知限制

- **DP 汇总只输出 rank 0**（RTP 是全 rank 平均）。已验证 case 里 rank 间 spread <1%
  可互换；若观察到差异变大，先改为输出全 rank 及均值再下结论
- **fake-KV 为实验功能**：KV 全零、数值与真实 prefill 不同，仅用于 kernel 计时对齐
- **MoE fake balance 已对齐 RTP**：harness 无条件设置 `FAKE_BALANCE_EXPERT=1`，
  真实 top-k 后按 RTP 的 EP/local-expert 两级 round-robin 覆写 expert IDs，并把权重置
  为 1。vLLM 热路径是 `copy_ + fill_` 两个原地 CUDA kernel，RTP 是单个专用 kernel，
  因此负载分布一致，但路由开销不应视为严格相同。该模式要求 CUDA modular MoE、默认
  `linear` expert placement，且不能同时启用 EPLB
- **与 RTP 对比 MoE 组件归因只能靠 kernel 名分类**：RTP 的 attn/norm/激活是手写 C++
  融合算子，在 cpu_op 层隐身；vLLM fake balance 的通用 `copy_`/`fill_` kernel 也可能落入
  `Other`，但端到端延迟仍完整包含这两次 launch

---

## 9. 已验证的参考结果

### Qwen3-8B, BF16, single L20D (B300)

| 引擎 | 模式 | BS | SeqLen | 延迟 (ms) |
|---|---|---|---|---|
| RTP-LLM | decode | 1 | 128 | 13.78 |
| vLLM | decode | 1 | 128 | 13.73 |
| vLLM | prefill | 1 | 128 | 14.53 |

### Qwen3.5-35B-A3B-FP8, single H20

| 模式 | BS | SeqLen | decode/tok(ms) | prefill(ms) |
|---|---|---|---|---|
| decode | 1 | 128 | 85.90 | 117.00 |
| decode | 1 | 1024 | 85.03 | 110.43 |
| decode | 4 | 128 | 87.08 | 111.26 |
| decode | 4 | 1024 | 86.95 | 173.15 |
| decode | 16 | 128 | 88.33 | 111.49 |
| decode | 16 | 1024 | 88.06 | 626.79 |

### Qwen3-235B-A22B-FP8, TP=4 × DP=2 EP, 8×H20

| BS (per rank) | SeqLen | Rank 0 (ms) | Rank 1 (ms) |
|---|---|---|---|
| 64 | 128 | 123.75 | 126.19 |
| 512 | 128 | 127.74 | 126.45 |
