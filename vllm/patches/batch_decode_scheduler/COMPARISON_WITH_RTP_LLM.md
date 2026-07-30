# vLLM 性能测试工具 vs RTP-LLM `grid_perf_test` 对比

本文档说明 `vllm/patches/batch_decode_scheduler/` 下这套性能测试工具与
RTP-LLM `rtp_llm/test/perf_test/grid_perf_test` 的对应关系、设计差异，以及为什么
两边会有这些差异。

目标读者：需要理解「vLLM 测出来的数字为什么能和 RTP-LLM 对比」、或需要维护/扩展
这套工具的人。

---

## 0. TL;DR（一句话概括）

- **RTP-LLM**：黑盒压测。起一个**真实推理服务**，客户端并发发 HTTP 请求，从服务
  返回的 `aux_info` 里读时间。定长批由一个 C++ `BatchDecodeScheduler` 实现。
- **vLLM 版**：白盒压测。用 `LLM(...)` 起引擎后，在进程内直驱 `EngineCore` 的
  `schedule → execute → update` 循环，不起 HTTP 服务。为了复现 RTP-LLM 那个定制
  scheduler 的行为（定长批、跳过 prefill、跨 DP EP），用了一系列进程内 hack。
  `BenchHarness` ≠ 完整 `vllm serve`（强制 Inproc、无 API / busy loop）。

两者的**指标定义和聚合方式刻意保持一致**，所以数字可以逐项对比。差异主要在
「怎么把引擎逼进被测状态」和「怎么取时间」。

> 工具分层（避免用一套工具扛两个互斥目标）：
>
> | 层 | 做法 | 测什么 |
> |---|---|---|
> | L1 黑盒全栈 | `LLM.generate` / `vllm serve` + bench | 用户感知 / 生产栈 e2e |
> | L2 白盒定长批（**本文档主体**） | 现 harness：`LLM` 起机 + 手 step | 定长批引擎步（含 schedule） |
> | ~~L3-b~~ | worker 经 `collective_rpc` 直驱 runner、绕过 scheduler | **不做**：口径变成 runner 微基准，见 §10.2c |

---

## 1. 整体架构对比

两套工具最根本的差异是**测试代码相对于引擎所处的位置**：RTP-LLM 的测试逻辑站在
引擎**外部、另一个进程**里，通过 HTTP 打服务；vLLM 版的测试逻辑钻进引擎**内部**，
直接驱动调度器（默认与 EngineCore 同进程；`dp_size>1` 时每 DP rank 各一进程）。

### RTP-LLM：client-server HTTP 压测（测试与引擎跨进程）

`GridRunner` / `BatchPerfImpl` 是**纯客户端概念**——它们不是引擎分层里的一层，而是
站在引擎之外的压测脚本，和引擎分处两个进程，只通过 HTTP 通信：

```
┌──────────────────────────────────────────────┐
│ 进程 A：压测客户端 (batch_decode_test.py)       │
│                                                │
│  main()                                        │
│   ├─ EngineServer.start()  ← 拉起/管理进程 B    │
│   └─ GridRunner            ← 网格编排(客户端)   │
│       └─ BatchPerfImpl     ← 单网格点执行        │
│           ├─ POST /update_scheduler_info (切BS/模式)
│           └─ ProcessPool+ThreadPool 并发 POST / │
│                 └─ 读 response["aux_info"].cost_time
└───────────────────────┬────────────────────────┘
                        │  HTTP  127.0.0.1:port
                        ▼
┌──────────────────────────────────────────────┐
│ 进程 B：引擎服务 (MagaServerManager 拉起)        │
│                                                │
│  frontend_app (HTTP 前端)   ← 服务层            │
│   └─ gRPC                                       │
│       └─ NormalEngine       ← 引擎层            │
│           └─ BatchDecodeScheduler ← 调度层      │
│               └─ Executor / model ← 计算层      │
└──────────────────────────────────────────────┘
```
- **两进程唯一的耦合点**是 `EngineServer`（`server.py`）：它在客户端进程里负责
  拉起/关闭引擎子进程，并注入 `USE_BATCH_DECODE_SCHEDULER=1`、`FAKE_BALANCE_EXPERT=1`
  等环境变量。但它本身仍是客户端侧的"服务生命周期管理器"，不属于引擎。
- **请求路径**：客户端 → HTTP → 前端 → tokenizer → gRPC → 引擎，是一条完整的
  生产链路，测的是端到端服务延迟。

### vLLM 版：进程内直驱 EngineCore（测试与引擎同进程）

vLLM 版**没有 GridRunner 那种外层客户端概念**。编排逻辑（`_run_bench_grid`）和引擎
驱动（`BenchHarness`）揉在同一层里，直接调用引擎内部对象。默认单进程；
`dp_size>1` 时 `main` 为每个 DP rank spawn 一进程，各建一份 `BenchHarness`/`LLM`
（见 §4）。

```
┌───────────────────────────────────────────────────────────────┐
│ 单网格进程 (perf_test_runner.py；DP 时每 rank 各一)              │
│                                                                 │
│  main() / _dp_worker()                                          │
│   └─ _run_bench_grid()      ← 网格编排                           │
│       └─ BenchHarness(...)  ← LLM(...) → Inproc EngineCore      │
│           ├─ scheduler = engine_core.scheduler                  │
│           └─ executor  = engine_core.model_executor             │
│       └─ 按 --partial 分流（0=PD / 1=decode fake-KV / 2=prefill）│
│           ├─ run_prefill_bench → submit → run_step()×1          │
│           └─ run_decode_bench  → submit → [prefill setup]       │
│                                 → run_step()×N (decode)         │
│                                                                 │
│  批级墙钟（headline，mark_batch_start → mark_batch_end）：       │
│     含每步 schedule + execute + update（+ 逐步 cuda.sync 副作用）│
│  run_step() 内部 forward_ms：                                   │
│     schedule()                                                  │
│     sync; t0 → execute_model + update_from_output → t1          │
│     （forward_ms 不含 schedule，且不上报，见 §3）                 │
└───────────────────────────────────────────────────────────────┘
```

- **模式由 `--partial` 决定**（RTP 对齐，`perf_test_runner.py:270`，默认 `1`）：`0`=PD
  （真 prefill + decode，双指标）、`1`=decode only（fake-KV 跳 prefill，默认）、
  `2`=prefill only。内部映射 `mode = "prefill" if partial==2 else "decode"`、
  `skip_prefill = (partial==1)`（`:388-389`），再分流到 `run_prefill_bench` /
  `run_decode_bench`。prefill 模式只跑 1 步（那步就是 prefill）；decode 模式先跑 1 步
  做 setup（单独计 `prefill_ms`）——`partial=0` 是真 prefill、`partial=1` 用假 KV
  setup（见 §2 / §11）——再跑 N 步 decode。
- `run_step()` 是**两模式共用的驱动原语**（schedule→execute→update 一次），本身不区分
  阶段——到底 prefill 还是 decode，取决于这步被调度的 batch：每请求调度 token 数 >1 是
  prefill，=1 是 decode（`_detect_phase()` 判定，`assert_phase()` 断言）。

关键前提：`VLLM_ENABLE_V1_MULTIPROCESSING=0`，走 InprocClient，才能在同进程里
直接拿到 `engine_core.scheduler` / `model_executor`。**没有 HTTP、没有 tokenizer
往返、没有前端/gRPC 开销**。上报的 `cost_ms` 是批级引擎步墙钟（含 `schedule()`），
不是「纯 GPU kernel」；也不含服务层。

> 含义：vLLM 版的数字天然不含服务层开销，比 RTP HTTP e2e 更接近引擎内核；但仍含
> schedule CPU 与逐步 sync 等（见 §3 / §12 P0）。RTP-LLM 用 `aux_info.cost_time` 而非
> HTTP 往返时间，已扣掉大部分网络/排队开销，两者因此仍可比，但不是零差异。

> 一句话对照：RTP-LLM 的 `GridRunner` 是"站在引擎外面打服务"的客户端；vLLM 版的
> `_run_bench_grid` 是"钻进引擎内部直接驱动 scheduler/executor"。前者是跨进程的
> *上层调用方*，后者是同进程的*内部驱动者*——这也是两套工具所有其他差异的根源。

---

## 2. 核心机制：怎么得到"定长批 + 纯 decode"

这是整个对比的核心。普通推理引擎是 continuous batching（动态拼批），批大小随时
变化，无法用来测"固定 BS 的稳定 decode 延迟"。两边解决方式完全不同。

### RTP-LLM：定制 C++ `BatchDecodeScheduler`

文件：`rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h`

1. **攒够 N 个再齐发**（`evaluateWaitingStreams`）：等待队列凑满 `batch_size_` 个
   请求，才一次性统一调度成一个定长批。避免动态拼批噪声。
2. **运行时切模式**（`updateSchedulerInfo`）：HTTP `/update_scheduler_info` 端点
   传 `{batch_size, mode}`，动态改 `batch_size_` 和 decode/prefill 模式。测试脚本
   每换一个网格点就调一次。
3. **跳过 prefill 直接 decode**（`initRunningStreams`）：decode 模式下对每个 stream
   调 `setIsContextStream(false)`，把请求当作 prefill 已完成、直接进 decode（KV 是
   未计算的假数据）；同时 `setPerfTest(true)`、`resetBeginTime()` 重置计时起点。

### vLLM 版：绕过动态调度，手动喂批

vLLM **不改调度器**，而是从外部把引擎"摆"成被测状态：

1. **定长批**：一次 `harness.submit(batch_size, seq_len)` 提交 N 个相同长度的合成
   请求，然后手动 `schedule/execute/update`，天然是定长批。用 `_detect_phase()`
   （每请求 scheduled token 数 =1 → decode，>1 → prefill）确认批的纯净度，
   `assert_phase()` 兜底。另有 `assert_batch()`：`schedule()` 若因 KV-cache OOM 或
   调度预算裁切把批**拆开**（只调度了部分请求），会直接报错——否则 `assert_phase`
   只看被调度子集的 phase，一个残缺批会静默通过并污染计时。
2. **切模式**：没有独立端点，`--partial 0/1/2` 启动时直接决定跑
   `run_prefill_bench` 还是 `run_decode_bench`（`perf_test_runner.py:388-389`）。
3. **跳过 prefill**（`submit_decode_only` + `--partial 1`）：这是最需要 hack 的部分。
   RTP-LLM 一行 `setIsContextStream(false)` 原生搞定；vLLM 要：
   - `_register_without_forward()`：经 `executor.collective_rpc(_register_requests_no_forward, ...)`
     把「注册请求但不跑 forward」fan-out 到**每个 TP worker**（`perf_test_harness.py:692-702`），
     每 rank 本地调 `model_runner._update_states()`（或 V2 的 finish/add/update +
     `apply_staged_writes`）建好各自的 block table，**不跑 prefill forward**；
   - **假 token bookkeeping**（`_FAKE_TOKEN_ID=1`，`:194`/`:197`）：被跳过的 sampler
     路径本该把 prefill 的「首个采样 token」写进 runner 本地状态，不补则首个 decode
     step 读未初始化内存 → embedding 越界 → device-side assert。故每 rank 注册时同步
     写入假 token（V1 `token_ids_cpu/num_tokens_no_spec`；V2 `last_sampled_tokens`）；
   - `_fake_update_from_output()`：伪造一个 `ModelRunnerOutput`（假 token），让
     scheduler 越过 prefill 阶段推进到 decode。
   - **TP 支持**：TP=1（UniProc 直调）与 TP>1（MultiprocExecutor 经 cloudpickle 走
     broadcast MQ）**均可用**——registration 经 `collective_rpc` fan-out，不再依赖
     `driver_worker`（`4932a086f` 起）。

| 能力 | RTP-LLM | vLLM 版 |
|---|---|---|
| 定长批 | C++ scheduler 攒批齐发 | 手动 submit N 个 + 手动 step |
| 切 decode/prefill | `/update_scheduler_info` 运行时切 | `--partial` 启动时定 |
| 跳过 prefill | 原生 `setIsContextStream(false)` | 伪造 KV + 假 output（经 collective_rpc，TP=1/TP>1 均可） |

### 深入：跳过 prefill 时，KV cache 是什么时候分配的？decode 读哪些 block？

核心结论：**"跳过 prefill" 跳的是 prefill 的 forward 计算，不是 KV 的分配。**
KV block 照常按完整 prompt 长度真实分配，decode 正常读这些 block，只是里面装的是
没算过的垃圾数据。**"访存/计算量真、数值假"** 是这套 perf 手段的本质——它保证延迟
可信，而输出正确性无意义（benchmark 不看输出）。

#### RTP-LLM

KV 分配**不在 prefill 里**，而在 scheduler 把 stream 从 `WAITING` 推到 `RUNNING`
的状态机里（`GenerateStateMachine::handleWaiting`）：

1. `BatchDecodeScheduler::evaluateWaitingStreams` 对每个 stream 发 `CanRun` 事件，
   然后忙等 `moveToNext()` 直到 `RUNNING`。
2. `handleWaiting()` → `initKVBlock(reserve_step_)` → `cache_manager->malloc()`
   （`StreamCacheResource.cc:319`）：**真实分配**物理 block，数量按 `seqLength()`
   （整个 prompt 长度）算 = `ceil(seqLen / seq_size_per_block) + reserve_step`。
   - 分配 ≠ 计算：malloc 只是领到物理 block 所有权、填进 stream 的 block table；
     block 里的内容是未初始化/残留的垃圾，因为没有 prefill forward 往里写。
   - decode role 首次 malloc 强制关掉 reuse cache（`StreamCacheResource.cc:337-339`），
     拿到全新空 block，不复用 prefix。
3. `initRunningStreams()` 里 `setIsContextStream(false)` 只是标记"这是 decode 步、
   别当 prefill 跑"，KV 分配此时早已完成。
4. **decode 读哪些 block**：就是上面 `initKVBlock` 分配、记录在 `kvCache()`
   （`BatchKVCacheResource`）block table 里的那 `ceil(seqLen/block_size)` 个 block。
   attention kernel 按 block table 地址读它们当"历史 KV"，每步再 `incrKVBlock` 追加
   新 token 的 block。block 数量与寻址都真实正确，所以访存/计算量与真实 decode 一致。

> 别混淆另一条"假 KV"路径：引擎**启动 warmup**（`decodeWarmUp` →
> `fakeInitKVBlock`，`NormalEngine.cc:169`）用 `resizeBlocks(n, 0)` 把 block table
> 全指向 dummy block 0，**连 malloc 都不做**。它是一次性启动预热（触发 CUDA graph
> capture / kernel JIT / 探显存上限），**不进被测循环**，不污染数字。注意它不是
> 生产专用——perf 测试的 `server.py` 恰恰用 `USE_BATCH_DECODE_SCHEDULER=1` +
> `BATCH_DECODE_SCHEDULER_WARMUP_TYPE=0` 主动配了这次 decode warmup。被测循环用的是
> 上面第 2 步的真 `initKVBlock` malloc，两者是同一 perf 流程里的不同阶段。

#### vLLM 版（同构做法）

`submit_decode_only()`（`perf_test_harness.py`）先走正常 `scheduler.schedule()`，让
vLLM 的 block manager **真实分配 KV 块**（和平时 decode 一样，按 prompt 长度）；再用：

- `_register_without_forward()`：手动调 `model_runner._update_states()` 把请求注册
  进 model_runner，但**不跑 prefill forward**；
- `_fake_update_from_output()`：伪造 `ModelRunnerOutput`（假 token）让 scheduler
  越过 prefill 推进到 decode。

结果同样是"块真实分配、内容不计算"：decode 读的就是 block manager 分配、记录在
block table / slot_mapping 里的那些块，访存量真、数值假。区别只是实现层级——RTP-LLM
在 C++ 状态机里原生完成，vLLM 靠进程内 hack 拼出来，且只在 TP=1 下可用。

| | RTP-LLM | vLLM 版 |
|---|---|---|
| KV 分配时机 | 状态机 WAITING→RUNNING（`initKVBlock` malloc） | `submit_decode_only` 内的 `scheduler.schedule()` |
| 分配量 | `ceil(seqLen/block)+reserve`，按完整 prompt | 同（vLLM block manager 按 prompt 长度） |
| 跳过的是 | prefill forward（`setIsContextStream(false)`） | prefill forward（`_register_without_forward`） |
| block 内容 | 未计算的垃圾 | 未计算的垃圾 |
| decode 读的块 | `kvCache()` block table 里那批真实块 | block manager 分配的真实块 |
| 层级 | C++ 原生 | 进程内 hack，经 collective_rpc 支持 TP=1/TP>1 |

---

## 3. 计时与指标定义

### RTP-LLM（`dataclass.py::ResponseInfo`）

从服务返回的 `aux_info` 里算（单位随字段，最终换算成 ms）：

```
total_time            = cost_time            - wait_time
prefill_time          = first_token_cost_time - wait_time
decode_time           = total_time - prefill_time
decode_time_per_token = decode_time / (output_len - 1)     # output_len>1 时
```

- `cost_time`：整轮 begin→last token，`resetBeginTime()` 保证起点干净。
- `wait_time`：排队时间，被扣掉。
- 采样单位：**每个请求**一条 `ResponseInfo`，跨请求求平均/方差。

### vLLM 版（`perf_test_runner.py`）

在同进程用 wall-clock；`mark_batch_*` / `mark_lap` / `run_step` 取时间前都会
`torch.cuda.synchronize()`：

```
mark_batch_start()   # prefill 前，cuda sync 后记起点
  run_step()         # prefill（decode 模式下的第一步；内部含 schedule）
prefill_ms = mark_lap()          # 到首 token，和 batch_start 同一时钟
  run_step() × num_decode_steps  # decode（每步内部都有 schedule）
cost_ms = mark_batch_end()       # 整轮结束

per_token_ms = (cost_ms - prefill_ms) / num_decode_steps
```

| 指标 | 是否含 `schedule()` | 是否上报 |
|---|---|---|
| **`cost_ms` / `prefill_ms` / `per_token_ms`**（headline） | **是**（夹在 mark 窗口内） | **是** |
| `run_step` 返回的 `forward_ms` | **否**（t0 在 `schedule()` 之后） | **否**（算完丢弃） |

- 公式**刻意对齐** RTP-LLM 的 `decode_time_per_token = (cost - prefill)/steps`
  （commit `030c7476a` 的目的）。当 `num_decode_steps == output_len - 1` 时两边一致。
- 采样单位：**每个 round**（一次 `num_iters` 迭代 = 一轮完整 prefill+decode），
  不是每个 step，也不是每个请求。
- **逐步 sync 副作用**：headline 虽用批级 `cost_ms`，但测量循环每步仍调
  `run_step()` → 每步 `cuda.synchronize()`，会排干 GPU、破坏步间 overlap，系统性抬高
  数字——见 §12 P0。

#### 异步调度：配置可能开着，测量路径没用上

vLLM 默认常把 `async_scheduling` 自动打开（`AsyncScheduler` +
`EngineCore.step_with_batch_queue`，让下一拍 `schedule` 与上一拍 GPU 重叠）。但
harness **不调用** `engine_core.step_fn()`，而是手搓 `schedule → sync → execute →
update`，等于拆掉重叠；逐步 sync 更把步间 overlap 掐死。

对本工具目标（定长批 prefill/decode 内核延迟、对标 RTP grid）——**不必优先接
异步调度**；先做 §12 P0（去逐步 sync）。真要测 async 收益，应改走
`engine_core.step_fn()` / `LLMEngine.step()`，且测量循环禁止逐步 sync。

### 聚合：两边都是 trimmed mean

| | RTP-LLM | vLLM 版 |
|---|---|---|
| warmup | `BatchPerfImpl.run` 先跑一次丢弃 | `--num-warmup-iters` |
| 采样次数 | `num_measures`（默认 3） | `--num-iters`（默认 5） |
| 聚合 | 排序去掉 min/max，剩下平均（`measurements[1:-1]`） | 同（`_trimmed_mean`）；<3 个样本退化为普通均值 |
| 采样单位 | 每请求 | 每 round |

> 结论：**指标公式和聚合逻辑一致，可直接对比**；差异在采样单位（请求 vs round）
> 和计时位置（服务端 aux_info vs 进程内 wall-clock）。

---

## 4. DP 与专家并行（EP）

### RTP-LLM

引擎本身有统一的分布式 world，天然支持跨 DP rank 的 EP all-to-all。测试侧
`--dp_size` + `update_scheduler_info` 广播到各 DP rank 即可，无需额外同步代码。

### vLLM 版

vLLM 的每个 DP rank 是**独立进程、独立 `torch.distributed` world**，这带来两种模式：

1. **独立 DP（不开 EP）**：各 rank 用 `CUDA_VISIBLE_DEVICES` 隔离 GPU，完全独立跑，
   互不通信。Dense 模型、或 MoE 不开 EP 时用这个。
2. **跨 DP EP**（`--enable-expert-parallel` + MoE 模型）：需要跨 rank 的 NCCL EP
   group，但独立进程到达 init 各阶段的时机不同，会 **NCCL mismatch 死锁**。
   vLLM 版用 monkey-patch 解决（commit `539b5a466` + `480e866eb`）：
   - patch `Executor.determine_available_memory` 和 `initialize_from_config`，在
     **显存 profiling** 和 **CUDA graph capture** 两个阶段前插
     `multiprocessing.Barrier.wait()`，保证所有 DP rank 一起进入；
   - 每步 `run_step` 里加 `_sync_dp()` barrier，让 EP all-to-all / CUDA graph replay
     在各 rank 间对齐；
   - 用 `VLLM_DP_*` 环境变量让 vLLM 自己分配 GPU（不能再手动设
     `CUDA_VISIBLE_DEVICES`，会和 `init_device` 的 dp_local_rank 逻辑冲突）。

> 这是 vLLM 版进程内驱动方式下的**最大额外工程**。RTP-LLM 的原生分布式 world 不涉及
> 这些同步 hack（可验证属性,非对其动机的推断）。`RUN_GUIDE.md`「多 DP Bench」已与本节对齐。

---

## 5. MoE 稳定性

- **RTP-LLM**：`FAKE_BALANCE_EXPERT` 环境变量强制专家均匀路由，使 MoE decode 延迟
  稳定可复现。
- **vLLM 版**：harness 同样无条件设置 `FAKE_BALANCE_EXPERT=1`。真实 top-k 仍会执行，
  随后按 RTP 的 EP/local-expert 两级 round-robin 覆写 expert IDs，并把权重置为 1；每个
  worker 在测量前通过 `collective_rpc` 审计所有 MoE 层，禁止静默 no-op。
- **剩余差异**：vLLM 用缓存模板的 `copy_ + fill_` 两个原地 CUDA kernel，RTP 用单个
  `fakeBalanceExpertKernel`。二者负载分布和权重一致，但路由 kernel 开销不完全相同。

---

## 6. VL（多模态）模型

- **RTP-LLM**：把这类模型当作纯文本模型跑（如按 `qwen_3_moe` 类型），无需特殊处理。
- **vLLM 版**：`--disable-mm` 把 `limit_mm_per_prompt` 的 image/video 清零，让引擎
  显存 profiling 跳过（很大的）vision dummy batch，只压语言模型 decode。否则会在
  vision dummy batch 处崩。目的就是和 RTP-LLM 的 text 基准对齐。

---

## 7. Profiling / 逐组件耗时分析

### RTP-LLM

请求里带 `gen_timeline=True` + `profile_step`，服务端内建 Kineto profiler 落盘
timeline（`TEST_UNDECLARED_OUTPUTS_DIR/timelines/`）。配套 `analyze_timeline.py`
按 kernel 名归类（Attention / MoE GEMM / Norm / RoPE / ...）。

### vLLM 版：两种采集后端

> 历史注记：早期的 nsys（`cudaProfilerStart/Stop` + `--capture-range=cudaProfilerApi`）
> 后端**已删除**，因为它和 WorkerProfiler 功能重叠且更笨重。现在只有两种：

1. **`--profile`（torch.profiler，单进程，TP=1）**：走专用 `profile_run()`，把 prefill
   排在 profiler 窗口**外**，只抓 N 个 decode step，产出干净 chrome trace。零引擎配置、
   最稳定。**只能看到 driver 进程**，TP>1 的 worker 子进程抓不到。
2. **`--worker-profile-dir`（WorkerProfiler，多 rank）**：vLLM 自带的 profiler，经
   `collective_rpc` 分发到**每个 TP/DP rank**，各自 dump 一份 trace。**TP>1 / DP / EP
   下唯一可用**。底层同样是 `torch.profiler`，只是多了 per-rank fanout。

#### WorkerProfiler 与 torch.profiler 的关系（别当成两种技术）

`WorkerProfiler`（`vllm/profiler/wrapper.py`）不是另一套采集技术，而是套在
`torch.profiler` 外面的**编排/管理层**——它的默认后端 `TorchProfilerWrapper` 底层就是
`torch.profiler.profile`。差异不在"用什么采集"，而在"**在哪个进程采、谁管生命周期**"：

- **运行位置（决定 TP>1 能否用）**：裸 `torch.profiler`（`--profile`）只在 **driver
  进程**里 new，TP=1 时模型就在本进程跑、抓得全；TP>1 时 forward 跑在 **worker 子进程**，
  driver profiler 看不到。`WorkerProfiler`（`--worker-profile-dir`）经 `collective_rpc`
  进**每个 worker 进程**，所以是 TP>1/DP/EP 下唯一能抓到 forward 的方式。fanout 链路：
  `LLM.start_profile()` → `llm_engine.start_profile()` → `collective_rpc("profile")` →
  每个 `gpu_worker.profile()` 起一个 WorkerProfiler。
- **生命周期管理**：WorkerProfiler 多了延迟启动（`delay_iterations`）、录满自动停
  （`max_iterations`）、warmup/active schedule、以及 try/except 容错（profiler 挂了只
  warning，不拖垮引擎）；裸 torch.profiler 这些要自己写。
- **输出**：WorkerProfiler 每 rank 一份 chrome trace（文件名带 `dp/pp/tp` rank 后缀）+
  一份 `profiler_out_<rank>.txt` kernel 汇总表。
- **多后端**：WorkerProfiler 是抽象基类，`TorchProfilerWrapper`（Kineto trace）和
  `CudaProfilerWrapper`（NVTX + `cudaProfilerStart/Stop`，给 nsys 用）都是它的子类——
  早期那条独立 nsys 路径就是被 cuda 后端**收编**后删掉的。
- **前置要求**：WorkerProfiler 需在**构造引擎时**就带 `profiler_config`（harness 靠
  `--worker-profile-dir` 自动配），否则 `worker.profile()` 抛 `RuntimeError`；裸
  torch.profiler 零引擎配置。

> 简记：**WorkerProfiler ≈ torch.profiler + 每-worker fanout + 生命周期管理 + 多后端**。
> TP=1 两者本质一样（都是 torch.profiler），裸的更省事；TP>1 只有 WorkerProfiler 能抓到
> 子进程里的 forward。两者产出同格式 chrome trace，分析层（`perf_test_timeline`）通用。

两种后端产出的 chrome trace 都用 `perf_test_timeline.py` 分析——**其 kernel 分类法
刻意复用 RTP-LLM `analyze_timeline.py` 的 taxonomy**，所以能逐 category 对比：

```
python -m ...perf_test_timeline vllm_decode_bs4_seq128_steps10.json  # 单 trace 分解
python -m ...perf_test_timeline --compare vllm.json rtp.json         # 逐 category diff
...perf_test_runner ... --analyze --rtp-trace rtp.json               # 跑时直接对比
```

### 三层 scope 观测能力（与 RTP 对齐的关键）

性能观测从粗到细、从 CPU 到 GPU 分三层。理解这三层，才知道**哪一层的数字能跨引擎对比**：

| 层 | 是什么 | 谁产生 | CUDA graph 下 | TP>1 | 跨引擎可比性 |
|---|---|---|---|---|---|
| **层一 引擎阶段 scope** | forward/preprocess/sample/… 的 CPU 墙钟 | 引擎手埋 `record_function` | forward 降级为"启动 graph"的 host 时间 | 需在 worker 内注入 | 阶段级可比（见下表） |
| **层二 算子框架 cpu_op** | 每次算子 dispatch 的 CPU 时间 | PyTorch/Kineto 自动 | **消失**（图内不再 dispatch） | worker 抓不到 | 裸 aten 可比；组件对不齐 |
| **层三 kernel（CUPTI）** | kernel 在 GPU 上的真实执行时间 | CUPTI 自动 | **穿透**（仍在） | 各引擎 per-rank 可拿 | **唯一可靠的跨引擎组件归因** |

- **层一是这批新代码的重点**。vLLM 的引擎阶段 scope（`gpu_model_runner: forward/
  preprocess/sample/...`）原生**只存在于 legacy V1 runner**；默认的 V2 runner
  （Qwen3/Llama/Mistral/DeepseekV2/Qwen2Moe 等）**原生没有任何 scope**——这就是为什么
  scope 早期"时有时无"。
  - `--vllm-scopes`：设 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`，并经 `collective_rpc` 把
    注入代码发到**每个 worker 进程**，monkey-patch V2 runner 的
    prepare_inputs/sample/postprocess/forward，emit 与 V1 同名的 scope。**保持真实 V2
    部署路径**，且 TP>1 每个 rank 都能记录（配 `--worker-profile-dir`）。天然走 V1 的
    模型自动检测、只破除 profiler-func 冻结、不重复套。
  - `--vllm-scopes-v1`：逃生开关，强制 `VLLM_USE_V2_MODEL_RUNNER=0` 回退 legacy V1 原生
    scope。
  - `--scope-forward`：额外套 `forward` scope（默认关；CUDA graph 下只量到 host 启动时间）。
- **层一与 RTP 的引擎阶段对齐**（这是 scope 工作的直接目的——让 vLLM 的阶段分解能对上
  RTP 的 `executor.*`）：

  | 阶段 | vLLM（V1 / 注入后的 V2） | RTP-LLM |
  |---|---|---|
  | 前向 | `gpu_model_runner: forward` | `executor.model_forward`（=py_model.forward） |
  | 采样 | `gpu_model_runner: sample` | `executor.sampler_forward` |
  | 输入准备 | `gpu_model_runner: preprocess` | `executor.gather_model_input` |
  | 输出 | `postprocess` / `ModelRunnerOutput` | `executor.dispatch_output` |
  | 调度 | `schedule: allocate_slots` | 包在 `engine.normal.step` 外层 |

  注意：`forward`/`schedule` 与 V1 逐值对齐；`preprocess/sample/postprocess` **名对齐、
  值因 V1/V2 阶段边界不同而有差异**（引擎结构差异，非 bug）。

- **层二**：vLLM 的组件（attention/moe/norm）注册成了 custom op，cpu_op 里有功能名可归因；
  **RTP 的 attn/norm/激活是手写 C++ 融合，不走 dispatcher，在 cpu_op 里隐身** → 组件归因
  对不齐。所以跨引擎的组件对比必须退回层三。
- **层三**：谁发的 kernel 都抓（含 RTP 手写 kernel），是**跨引擎组件归因唯一可靠的层**，
  靠 kernel 名分类（`perf_test_timeline` 就工作在这一层）。

> 结论：部署路径（TP>1 + CUDA graph）下，层一降级、层二消失，**只有层三 kernel 名分类
> 可靠**。要看清被 `triton_*_fused` 融掉的 RoPE/Norm/Activation，用 `--enforce-eager`
> 禁融合恢复 kernel 语义名。

### 一个对齐层面的差异（RTP 天然，vLLM 靠注入）

RTP-LLM 的引擎阶段 scope 是 C++ 里原生手埋的，任何模型、任何部署都在。vLLM 的层一
scope 只在 legacy V1 原生存在，V2（真实部署路径）要靠 harness 经 `collective_rpc` 注入
才有——又是一处「RTP 原生、vLLM 靠 hack 追平」的模式，和第 2 节跳 prefill、第 4 节跨 DP
EP 一脉相承。

---

## 8. 功能覆盖差异

RTP-LLM 有、vLLM 版**目前没有**的功能：

| 功能 | RTP-LLM | vLLM 版 |
|---|---|---|
| Grid 模式（BS × seq_len） | ✅ `GridRunner` | ✅ |
| Prefill / Decode | ✅ `--partial 2/1` | ✅ `--partial 0/1/2` |
| **TPS 二分搜索**（给定 target TPOT 找最大 BS） | ✅ `TpsBinarySearchRunner` | ❌ |
| **Distribution 模式**（真实数据集 seq_len 分布） | ✅ `DistributionRunner` | ❌ |
| **结果可视化出图** | ✅ `plot_decode_results` | ❌ |
| DP | ✅ | ✅ |
| 跨 DP EP | ✅ 原生 | ✅（monkey-patch） |
| 逐组件 timeline | ✅ | ✅（对齐 taxonomy） |

其他约定差异：
- **Prefill 批大小**：RTP-LLM prefill 模式强制 BS=1（`GridRunner` 里
  `batch_size_list = [1] if not is_decode`）；vLLM prefill 支持任意 BS。
- `--partial` 语义：RTP-LLM 只有 `1=decode`、`2=prefill` 两个合法值，**没有
  "both"**（`choices=[1, 2]`）。

---

## 9. 汇总对照表

| 维度 | RTP-LLM `grid_perf_test` | vLLM 版 |
|---|---|---|
| 测试架构 | HTTP client-server 黑盒 | 进程内直驱 EngineCore 白盒 |
| 是否起服务 | 是（`EngineServer`） | 否（InprocClient） |
| 定长批 | 定制 C++ `BatchDecodeScheduler` | 手动 submit + 手动 step |
| 切 decode/prefill | 运行时 `/update_scheduler_info` | 启动时 `--partial 0/1/2` |
| 跳过 prefill | 原生 `setIsContextStream(false)` | 伪造 KV + 假 output（collective_rpc，TP=1/TP>1） |
| 计时位置 | 服务端 `aux_info.cost_time` | 批级 wall-clock（含 schedule；逐步 cuda.sync，见 §3） |
| 指标公式 | `(cost-prefill)/steps` 定义方 | 严格对齐 RTP |
| 聚合 | trimmed mean，采样单位=请求 | trimmed mean，采样单位=round |
| DP+EP | 原生分布式 world | monkey-patch `Barrier` |
| MoE 稳定 | `FAKE_BALANCE_EXPERT` 单 kernel | `FAKE_BALANCE_EXPERT`，`copy_ + fill_` |
| VL 模型 | 原生按 text 跑 | `--disable-mm` 清零 mm 槽位 |
| Profiling 采集 | 服务端 `gen_timeline` | `--profile`(torch, TP=1) / `--worker-profile-dir`(WorkerProfiler, TP>1) |
| kernel 分类器 | `analyze_timeline.py` | `perf_test_timeline.py`（对齐 taxonomy） |
| 引擎阶段 scope | C++ 原生手埋，任何模型/部署都有 | 仅 V1 原生；V2 靠 `--vllm-scopes` 经 `collective_rpc` 注入 |
| TPS 搜索 / 分布模式 / 画图 | 有 | 无 |

**核心取舍**：vLLM 版用一系列进程内 hack（手动喂批、伪造 KV、monkey-patch barrier）
去"逼近"RTP-LLM 那个为压测量身定制的 C++ scheduler 的行为。好处是零服务层开销、
直接可控、更贴近引擎步时间（仍含 schedule，见 §3）；代价是跳 prefill、跨 DP EP
比 RTP 原生更脆弱（依赖 runner 内部私有 API，skip-prefill 已经 `collective_rpc` 化
支持 TP=1/TP>1，见 §11.3）。指标层面两边刻意对齐，数字可比。演进方向见
§12（P0 去逐步 sync 为当前唯一硬缺口；不做 L3-b）。

---

## 10. 白盒 vs 黑盒：客观评价与改进空间

> 本节脱离"为现有方案辩护"的视角,客观评价两种测试形态的本质优劣,并指出
> 两边现有实现里哪些"代价"其实是**实现选择**而非**形态固有**。所有事实性结论带
> `文件:行号`;标注「评价」的是分析性推断。

### 10.1 两种形态的本质优劣

| 维度 | vLLM 白盒(in-process) | RTP-LLM 黑盒(HTTP→gRPC→C++) | 本质差异 or 实现偶然 |
|---|---|---|---|
| decode step 隔离度 | 高:可剥 prefill forward、伪造 KV / output(`perf_test_harness.py:665-720`) | 低:必须走完整 tokenizer→前端→gRPC→状态机→真实 `initKVBlock` | **本质**:白盒能剥外层 |
| profiling 颗粒度 | 高:卡到 `execute_model`、注入三层 scope、跨 rank fan-out WorkerProfiler | 中:靠引擎 `gen_timeline` dump,受引擎暴露的 timeline 限制 | **本质** |
| 生产栈代表性 | **低**:强制 `VLLM_ENABLE_V1_MULTIPROCESSING=0`(`perf_test_runner.py:564`),绕过 MPClient / `run_busy_loop` / 前端 / API server | **高**:走完整生产链路,测用户真实感知延迟 | **本质**:黑盒完胜 |
| 并行度支持 | skip-prefill 经 collective_rpc 支持 TP=1/TP>1(`perf_test_harness.py:692-702`);DP 靠手搓多进程 | TP/DP/PP/EP 由引擎服务自起,客户端只管除尽 | **多为偶然**(见 10.2) |
| 可复现性 | 单进程确定性高 | 依赖网络并发时序凑批,靠调度器忙等硬凑 | **本质**:黑盒天然更差 |
| 维护成本 | 高:耦合大量 vLLM 内部私有 API,内部一改就碎 | 中:只依赖稳定 HTTP 契约 + 一个后端接口 | **本质**:白盒耦合内部 |
| CI 友好度 | 好:一个 `python -m` 进程 | 差:需拉真实服务、等端口、收 coredump | **本质** |

一句话:**白盒在「隔离 + profiling + CI」本质占优,黑盒在「生产代表性 + 透明支持并行度」本质占优。**

### 10.2 关键澄清:vLLM 那几个"代价"多是实现选择,不是白盒固有

**(a) 手写 `multiprocessing.Barrier` —— 执行步那道基本冗余。**
`dp_size>1` 时 `execute_model` 内部必然先跑阻塞式 `dist.all_reduce`
(`vllm/v1/worker/gpu/dp_utils.py:33-36`),这本身就是全 rank 硬 rendezvous——
任一 rank 不进 `execute_model`,其他 rank 的 all_reduce 就返回不了。所以执行步前的
`_sync_dp` Barrier(`perf_test_harness.py:443`)**正确性价值是重复的**,其真实价值
只是 timing hygiene(把慢 `schedule()` 挡在计时窗外)。唯一有实义的是 **init 阶段**那道
(`_patch_executor_for_dp_sync` 包住 profiling/图捕获,`perf_test_harness.py:400-433`),
而它之所以需要,恰恰是因为 harness 主动 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 绕开了生产的
`DPEngineCoreProc.run_busy_loop` 编排——**是"绕开生产编排"的连带代价,不是白盒的代价**。〔评价〕

**补:harness 的 Barrier 只在 EP 路径传入**(`perf_test_runner.py:619` `use_dp_env = enable_expert_parallel and _detect_moe(...)`,`:587` `dp_barrier=barrier if use_dp_env else None`)。
**非 EP 的所谓"DP"根本不是 DP**:走 `CUDA_VISIBLE_DEVICES` 隔离
(`perf_test_runner.py:580`),就是 N 个互不相干的单卡引擎各测各的。`--batch-sizes` 现在
是 **per-DP-rank** 语义(`4a80e2db4` 起,`local_batch_sizes = list(batch_sizes)` `:582`,
不再 `//dp_size` 拆分、也不再 `*=dp_size` 回乘),每 rank 各跑 `bs` 个请求、total=bs×dp_size,
对齐 RTP GridRunner 的 batch_size 列。表格默认只报 rank0。

**(b) skip-prefill 的 TP=1 限制是实现选择,非白盒固有——已消除。** 经 `collective_rpc`
fan-out 到每个 worker 补齐 TP>1(`4932a086f`,机制见 §2 / §11.3)。〔评价:已兑现〕

**(c) 更贴生产的白盒形态,以及一个必须避开的陷阱。** 直觉方案是"开真实多进程引擎 +
`collective_rpc` 往 worker 注入探针"。但这里要分清两个变体,可行性与代价差别很大:

- **L3-b(探针直驱 runner,绕过 scheduler)**:collective_rpc 到 worker,探针本地造假 prefill +
  直接调 `model_runner` execute + 计时,**不经 scheduler/busy_loop**。技术上能做,但**代价被低估**:
  一旦 scheduler 不在环,worker 侧就得**自己模拟 scheduler 的产出**——因为 `KVCacheManager` 在
  Core 进程(`vllm/v1/core/sched/scheduler.py:254`)、worker 只是消费 `block_ids` 建 block table
  (`vllm/v1/worker/gpu/model_runner.py:796`),探针必须自行指派 block_ids、拼
  `num_computed_tokens`/`num_scheduled_tokens`、逐步推进。好在这是"静态输出契约"而非"分配智能"
  (定长 decode 不需要 prefix cache/驱逐/抢占),但**紧耦合 `NewRequestData`/`_update_states`/
  `BlockTables` 等内部结构**,且计时步**不走真实调度循环**——"编排是生产的"只对进程模型/NCCL 组成立,
  对调度循环不成立。〔评价〕

- **已落地的做法(`4932a086f`,机制见 §2 / §11.3):调度权不交出。** setup 仍调真
  `scheduler.schedule()`(Core 内)做**真 KV 分配**,`collective_rpc` 只把这份**已算好的合同**
  登记到每个 TP worker(登记+不 forward),`update_from_output` 仍在 Core 推进状态机;计时步照旧
  三拍。这样**从根上避开了"worker 手搓 block_ids/KV 布局"的陷阱**——因为 schedule() 始终在环、
  KV 由真 scheduler 分配。它**已补上** TP>1 缺口,又不退化成 L3-b 那种无调度器裸 runner。**这就是
  当前 skip-prefill TP>1 的实际落地形态。**

### 10.3 对称澄清:RTP 的凑批机制也不是黑盒固有

RTP 为"经 API 凑定长批"付的代价——`request_id % dp_size` 兜底路由
(`rtp_llm/cpp/model_rpc/model_rpc_client.py:454`)、`/update_scheduler_info` 广播、
除尽检查(`batch_perf_impl.py:167-170`)、调度器 30s 忙等硬闸门
(`BatchDecodeScheduler.h` `schedule()`)——同样不是黑盒必需。**引擎本就有 fake-stream 能力**:
`makeFakeStream` / `fakeInitKVBlock`(`cpp/normal_engine/speculative/MtpExecutor.cc:53-57`)、
`need_fill_fake_stream_`(`FIFOScheduler.cc`)、`is_fake_stream` 模型输入(`ModelTypes.cc`)。
补一个后端接口(每 rank 本地自造 N/dp 个 fake decode stream)即可免路由、免网络往返、
免忙等。当前这套是"复用现有请求路径、不改 pybind"的实现权衡。〔评价〕

### 10.4 最终裁决与各自最该改的一点

- **目标=隔离纯 decode step GPU 成本 → 白盒(vLLM 方向)更优。**
  最该改:**测量循环去掉 per-step `torch.cuda.synchronize()`**。decode 循环每步调
  `run_step`(`perf_test_runner.py:173`),其中每步 `torch.cuda.synchronize()` 后才计时
  (`perf_test_harness.py:476`);虽然 headline 用的是批级 `cost_ms`(`:185`,每步的
  `forward_ms` 实际被丢弃),但**这个同步调用每步照跑,强制排干 GPU、破坏 decode 步间
  overlap,从而抬高批级 `cost_ms`**。应改用无同步的 `run_step_no_timing`
  (`perf_test_harness.py:489`,`profile_run` 已这么做),只在批首尾各同步一次(保留
  `assert_phase/assert_batch` 检查)。〔评价:量级未实测,方向确定〕

- **目标=测真实生产栈 → 黑盒(RTP 方向)更优。**
  最该改:把凑批从"调度器忙等 + 网络时序"换成**引擎级 fake-stream 测试接口**
  (引擎已有能力,差一个入口)。

- **两边共同的现象**〔评价:描述工具当前形态,非推断其设计动机〕:两个工具目前都用**单一工具**
  同时覆盖"隔离"与"生产代表性"这两个互斥目标,于是各自在不擅长的方向堆 hack(vLLM 堆
  Barrier / 伪 DP;RTP 堆忙等闸门 / 取模路由)。**一种更清晰的组织方式
  是分层**:kernel 级用白盒(修掉 per-step sync),全栈级用黑盒(改 fake-stream 接口)。

---

## 11. 长序列 + 大 batch decode 压测:跳 prefill 的必然性、伪造 KV 的边界与现状

> 本节针对**核心目标场景——长 KV 序列 + 大 batch 的 decode 性能**——澄清一个容易被
> 误判的问题:"能不能真跑一次 prefill 再测 decode"。结论:**在该场景下真跑 prefill
> 物理上不可行,伪造 KV 跳 prefill 是唯一正确做法**(与 RTP-LLM 的 `setIsContextStream(false)`
> 殊途同归)。〔事实〕带行号,〔评价〕为分析性推断。

### 11.1 为什么长序列 + 大 batch 必须跳 prefill〔评价,基于算子常识〕

真跑 prefill 要对 `batch × seq_len` 个 token 做一次前向。以 batch=256、seq_len=8192 为例
≈ 210 万 token 的 prefill:

- **激活峰值 OOM**:harness 关闭了 chunked prefill(单步大激活,见文档 §2 "vLLM 版" 说明),
  大 batch 的单次 prefill 前向激活直接爆显存。
- **chunked prefill 也救不了迭代速度**:分块只解决 OOM,那 210 万 token 的**总计算量**照旧,
  每轮测量会慢到无法迭代;而且 prefill 的计算成本会淹没我们要测的 decode 信号。

因此"真跑 prefill 再 decode"(文档 §10.4 对**短序列**场景的推荐)在本场景**不适用**。
RTP-LLM 同样用 `setIsContextStream(false)` 完全跳过 prefill;长序列 + 大 batch 下真跑 prefill
物理上不可行,是两种工具都面对的客观约束(此为约束事实,非对 RTP 选型动机的推断)。

### 11.2 伪造 KV(数值假)对 decode 数字的影响边界

现有 `submit_decode_only`(`perf_test_harness.py:665`)伪造的状态:真实分配 KV **块**、
把请求 `num_computed_tokens` 快进过 prefill 并塞进 `scheduler.running`、经 `collective_rpc`
在每 rank 注册 block table(`_register_without_forward` → `_register_requests_no_forward`
`:692-702`/`:197`)、伪造一个 `ModelRunnerOutput` 让调度器推进(`_fake_update_from_output`
`:705`);**唯独不跑 prefill forward → KV 块内容是 zero,从未被计算**
(自述 "matching RTP-LLM's setIsContextStream(false)")。

| 模型类型 | 数值假是否影响 decode 性能数字 | 原因 |
|---|---|---|
| **Dense** | **不影响** | decode 成本 = 读 KV 做 attention(访存带宽瓶颈)+ projection/MLP GEMM + sampler,只取决于 KV 的**形状/布局/dtype/数量**,不取决于**数值**。"访存量真、数值假"——而 decode 压测量的就是访存量 |
| **MoE** | **影响,需额外处理** | 专家路由(gating)按 hidden state **数值**选专家;garbage KV → garbage hidden state → 随机/畸形路由 → 专家负载不均 → **每轮方差大、专家利用率不代表真实分布** |

**MoE 均衡已对齐〔事实〕**:vLLM harness 与 RTP-LLM 一样无条件启用
`FAKE_BALANCE_EXPERT`。vLLM 保留真实 top-k 的开销,随后按 RTP 公式覆写 expert IDs 和
weights；模板在 profile/CUDA Graph capture 前创建并由同配置 MoE 层共享。`EPLBConfig`
(`vllm/config/parallel.py:57`)仍是另一类功能——它会运行时搬迁专家，fake balance 模式明确
拒绝与 EPLB 同时启用。Dense 模型不受该开关影响。

### 11.3 伪造 KV 这条路的现状:TP 无关 + 残留成本

**(a) TP 支持(已实现)。** skip-prefill 经 `collective_rpc` fan-out 到每个 worker,TP=1/TP>1
均可(`4932a086f`);机制细节见 §2「跳过 prefill」。

> **`collective_rpc` 是什么**:`LLM`/`Executor.collective_rpc` 把方法广播到**本
> engine 的各 TP worker**(经 `model_executor`),**打不到** `EngineCore.scheduler`。
> 调度决策仍在 Core;`SchedulerOutput` 在 Core 算好后再下发。因此 skip-prefill 只能是
> 「Core `schedule` 出合同 → rpc 登记到 worker」,不能指望 rpc 在 worker 里「造调度」
> (那是 §10.2c 已否决的 L3-b)。跨 DP 也不在一次 rpc 覆盖范围内(DP 仍靠多进程编排)。

**外加:假 token bookkeeping**(`_FAKE_TOKEN_ID=1`,`:194`)。被跳过的 sampler 路径本该把 prefill
的「首个采样 token」写进 runner 本地状态,不补则首个 decode step 读未初始化内存 → embedding
越界 → device-side assert。修法在每 rank 注册时同步写入假 token(V1 `token_ids_cpu/`
`num_tokens_no_spec`,`:228-231`;V2 `last_sampled_tokens`,`:244`)。Qwen2.5-0.5B TP=1/2 验证
per_token 与真 prefill 路径一致。

**(b) 残留脆弱性**:即便 collective_rpc 化,仍调 `_update_states` 等 runner 私有方法——这是
vLLM 里跳 prefill **不可消除**的成本(无公开的"注入已 prefill 请求"API)。但与 RTP 通过自身
状态机做同样事是同一量级的内部耦合,且比"额外伪造 output 路径"干净:一个 well-scoped 的
collective_rpc 函数,版本升级只需盯这一处。〔评价〕

**(c) 稳定性来源**:来自 §1/§10 的定长批机制——stock `Scheduler` + 一次性提交 N 个 +
`ignore_eos` + KV 留足,准入闸门贪婪(`vllm/v1/core/sched/scheduler.py:629-631`),每步恰好 N
个 decode token,零抖动。**与 prefill 无关**,伪造 KV 不破坏它(dense);MoE 再叠加 (a) 之外的
强制均衡路由。

### 11.4 本场景推荐组合

| 关注点 | 推荐 | 依据 |
|---|---|---|
| 定长批 | stock `Scheduler` + 受控提交 | 贪婪调度器天然产出定长批(`scheduler.py:629-631`) |
| 跳 prefill | **保留伪造 KV**(真 prefill 物理不可行),`_register_without_forward` 已 `collective_rpc` 化、TP=1/TP>1 均支持 | `perf_test_harness.py:197/701` |
| Dense 稳定性 | 无需额外处理 | 数值假不影响访存型 decode 成本 |
| MoE 稳定性 | 使用 harness 内置 fake balance | IDs/weights 与 RTP 对齐；vLLM 为两个原地 kernel |

**一句话**:长序列大 batch 场景下,跳 prefill + 伪造 KV 是唯一可行且正确的选择
(缺官方「一进来就是 decode」API;更简单的「真跑 prefill」仅适短/小)。(1) `collective_rpc`
化已把它从 TP=1 解放(`4932a086f`),(2) MoE fake balance 也已补齐。稳定性靠定长批调度
和确定性专家路由,不靠 prefill。

---

## 12. vLLM 白盒方案的可优化项(按优先级)

> 面向"尽量贴近生产 + 结果稳定"的定长批长序列 decode 微基准目标,汇总当前 vLLM 版
> harness 的可优化项。〔事实〕带行号;〔评价〕为分析性推断,不涉及对任何实现动机的判断。
> 结论:**形态(白盒 in-process)与该目标匹配**;以下为实现层的完成度改进,非路线问题。
> (skip-prefill TP>1 缺口已由 `4932a086f` 补齐,见 §11.3;当前唯一直接影响数字的硬缺口是 P0。)

### P0 —— 直接影响所报数字:测量循环的 per-step `torch.cuda.synchronize()`

decode 测量循环每步调 `run_step`(`perf_test_runner.py:173`),其中每步先
`torch.cuda.synchronize()` 再计时(`perf_test_harness.py:476`)。headline 指标虽取**批级**
`cost_ms`(`perf_test_runner.py:185`,每步 `forward_ms` 实际被丢弃),但该同步调用每步照跑,
**强制排干 GPU、破坏 decode 步间 overlap,系统性抬高 `cost_ms`**。〔评价:量级未实测,方向确定〕

- **修法**:测量循环改用 `run_step_no_timing`(`perf_test_harness.py:489`),只在批首尾各同步
  一次(`mark_batch_start/end` 已在做)。采集路径 `profile_run`(`:418`)已全程用
  `run_step_no_timing`——本改动即让墙钟测量路径对齐采集路径。
- **优先级理由**:最便宜,且直接决定 decode 数字是否偏高。

### P1（已完成）—— MoE 稳定性(仅 MoE 模型)

实现已在真实 `select_experts` 后安装确定性 post-select wrapper，并在权重和最终 modular
kernel 准备完成后预生成共享模板。模板复刻 RTP 的 DP/EP offset，weights 恒为 1；worker
审计要求所有 MoE 层均完成 finalization。monolithic、EPLB 和非 CUDA 配置显式失败。

### P2 —— 清晰度 / 防误读:非 EP 的"DP"不是 DP;执行步 barrier 冗余

- Barrier 只在 EP 路径传入(`perf_test_runner.py:619` `use_dp_env = enable_expert_parallel and
  _detect_moe(...)`,`:587`);非 EP 走 `CUDA_VISIBLE_DEVICES` 隔离(`:580`),是 N 个独立
  单卡引擎各测各的。`--batch-sizes` 现为 **per-rank**(`4a80e2db4`,`local_batch_sizes =
  list(batch_sizes)` `:582`,total = bs×dp_size)——**语义不是 DP,易被误读为"测了 DP"**。
  - **非 EP 路径的实际用途**:各副本互不通信,唯一的增量信息是"各 GPU 数字一不一致"。
    代码印证:汇总**只报 rank 0**(`:686` `results = rank_results[0]`),其余 rank 仅用来和
    rank 0 比对、差异 **>10% 打 `WARNING`**(`:688-698`)。→ 用途是**跨 GPU 一致性抽查**
    (抓慢卡 / 热降频 / binning 差异),而非测 DP 协调。附带效果:rank 0 的数字是在**整机
    N 卡同时满载**下测的,比空闲整机跑单卡更接近生产热态——这也是该路径值得保留(而非删除)
    的理由。
  - → 因此这是**命名/文档问题,不是删能力**:把它与"真 EP-DP"显式分开(如另起
    `--replicas` 或标注"independent 单卡放大 / 满载一致性抽查"),消歧义即可,不必移除。
- 执行步前的 `_sync_dp` barrier(`perf_test_harness.py:443`)与 `execute_model` 内的阻塞
  `dist.all_reduce`(`vllm/v1/worker/gpu/dp_utils.py:33-36`,本身即全 rank rendezvous)功能重叠,
  **正确性价值冗余**,真实价值仅 timing hygiene。→ 可只保留 init 阶段 barrier
  (`_patch_executor_for_dp_sync` `:400-433`)。〔评价〕

### P3 —— 可选:开 MP Core(独立进程),仍走引擎三拍;明确不做 L3-b

当前为拿进程内句柄而 `VLLM_ENABLE_V1_MULTIPROCESSING=0`(`perf_test_runner.py:564`/`:602`),绕过
`run_busy_loop` / MPClient。更贴生产进程模型的演进:
`MULTIPROCESSING=1`,Core 独立进程跑 busy loop;driver 经 **Client→Core 控制面**驱动
(不能再 `self.scheduler.xxx`);`collective_rpc` **只**用于「Core 已 `schedule` 之后把
`SchedulerOutput` 登记进各 worker」(skip-prefill setup)。**计时步仍是 Core 里的
`schedule → execute → update`。**

**明确不做 L3-b**:用 `collective_rpc` 在 worker 里直驱 `model_runner`、绕过 scheduler
做假 prefill + 计时——进程/NCCL 更真,但调度循环不在被测栈,口径变成 runner 微基准
(见 §10.2c)。也不要把「假 prefill + 计时」都塞进同一个 rpc 探针里表述。

异步调度接 `step_fn` 同属可选增强,对本场景(定长批内核延迟)ROI 低于 P0,见 §3。

### 小结

| 项 | 影响面 | 成本 | 是否本场景必修 |
|---|---|---|---|
| P0 去 per-step sync | decode 数字准确性 | 低 | 是（当前唯一硬缺口） |
| P1 MoE 强制均衡路由（已完成） | MoE 稳定性 | 中 | harness 已内置 |
| P2 澄清伪 DP / 去冗余 barrier | 可读性 / 结果可信度 | 低 | 建议 |
| P3 MP Core(独立进程);不做 L3-b | 生产进程模型保真度 | 高 | 可选演进 |

---

## 附：文件对应关系

| 角色 | RTP-LLM | vLLM 版 |
|---|---|---|
| 入口 / 编排 | `batch_decode_test.py` | `perf_test_runner.py` |
| 单网格点执行 | `batch_perf_impl.py::BatchPerfImpl` | `perf_test_runner.py::run_*_bench` |
| 网格编排 | `grid_runner.py::GridRunner` | `perf_test_runner.py::_run_bench_grid` |
| 引擎驱动 | `EngineServer` + C++ `BatchDecodeScheduler` | `perf_test_harness.py::BenchHarness` |
| 指标定义 | `dataclass.py::ResponseInfo` | `perf_test_runner.py::BenchResult` |
| timeline 分析 | `analyze_timeline.py` | `perf_test_timeline.py` |
| 运行文档 | `BUILD`（args）+ 内部 wiki | `RUN_GUIDE.md` |
