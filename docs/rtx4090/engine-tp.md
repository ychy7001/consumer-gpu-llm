# FP8 kernel 调优、SGLang 对比与单卡副本

2026-10-31 凌晨实验，回答三个问题：

1. vLLM 日志里 FP8 block kernel 用的是默认配置（`Using default W8A8 Block FP8 kernel config`），调好以后能快多少？
2. 同样是 FP8，SGLang 0.5.22（MTP 或 DFlash2 投机解码）能否胜过 vLLM？
3. 两张卡跑 1 个 TP2 副本，还是 2 个单卡（TP1）副本？

结论：kernel 调优只多出约 1.5%；在批处理负载下 SGLang 两种投机解码都比 vLLM 慢；**2 个 TP1 副本的总吞吐比 1 个 TP2 高 63%**。瓶颈在 TP2 走 PCIe 的 all-reduce（无 P2P），不在 GEMM kernel。

## 方法

- 实验在 GPU 2,3 上做，直连 `:8005`；GPU 0,1 上的另一个 TP2 副本全程经 Router 服务线上流量，由 [guard_single_backend.sh](../../scripts/guard_single_backend.sh) 兜底。CPU/PCIe 与线上共享，数据只用于横向比较。
- 每组同一套负载，脚本在 [experiments/](../../experiments/)：
  - 纯 prefill：[prefill_probe.py](../../scripts/prefill_probe.py)，随机 token id，单请求 2k/16k/64k 各 2 轮，另测一次 124k。
  - decode：[decode_probe.py](../../scripts/decode_probe.py)，真实提示词，单路和 16 路并发，`max_tokens 512`。
  - 批处理形态：`vllm bench serve` 随机数据，输入 2048、输出 160、并发 24、144 个请求，模拟线上知识库导入（实测线上均值 2066/165）。
  - 质量：[gsm_probe.py](../../scripts/gsm_probe.py)，GSM8K 前 200 题，提示词与判分沿用 HyperQwen 的 `quality_battery.py`，关思考、贪心解码。
- vLLM 基线即生产配置：`--max-num-seqs 64`、prefix caching、MTP2、gmu 0.78、kv fp8、262k 上下文、mbt 默认 2048。
- 每组只跑一次。单路 decode 只有 3 个样本，同配置不同组之间相差可达 ±15%，只看量级。

## FP8 block kernel 调优

用 vLLM 的 `benchmarks/kernels/benchmark_w8a8_block_fp8.py` 调 TP2 下 5 个形状（以及猜测的 TP1 形状），patch 后的脚本在 [scripts/benchmark_w8a8_block_fp8.py](../../scripts/benchmark_w8a8_block_fp8.py)：

- `TUNE_SHAPES`、`TUNE_BATCHES` 指定形状和 batch；`TUNE_FAST=1` 把搜索空间从 1280 个配置裁到 96 个，一个形状约 1 分钟。
- 上游多卡模式按 batch 分卡，但每张卡写同一个文件、互相覆盖；改成按形状分给两个单卡容器。
- vLLM 0.25.1 的 block FP8 只在包内 `quantization/utils/configs/` 查找配置，不认 `VLLM_TUNED_CONFIG_FOLDER`，[deploy_exp.sh](../../scripts/deploy_exp.sh) 用 `KERNEL_CFG` 逐个文件挂载进去。结果在 [configs/rtx4090-fp8-block/](../../configs/rtx4090-fp8-block/)。

| 配置 | 16k prefill tok/s | 批处理总 tok/s | TTFT P50 / P99 ms | TPOT P50 / P99 ms |
| --- | --- | --- | --- | --- |
| 默认 kernel，mbt 4096 | 2800 | 2350 | 1556 / 16543 | 116 / 246 |
| 调优 kernel，mbt 4096 | 2875 | 2389 | 1566 / 16622 | 112 / 244 |
| 调优 kernel，MTP3 | 2800 | 2339 | 1652 / 17264 | 110 / 274 |
| 调优 kernel，mbt 8192 | 2864 | 2317 | 1631 / 17090 | 117 / 236 |

四组都在噪声范围内。MTP3 每步多接受 0.3 个 token（2.14 → 2.46），但并发 24 下验证开销抵掉了收益。

## 引擎与并行方式

| 配置 | 16k prefill | 124k TTFT | 单路 / 16 路 decode | 批处理总 tok/s | TTFT P50 / P99 ms | GSM8K |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM TP2（生产） | 2754 | 62.2 s | 58 / 533 | 2353 | 1791 / 17196 | 96.5% |
| SGLang TP2 + MTP | 2537 | 62.6 s | 72 / 488 | 1632 | 1615 / 19083 | 96.5% |
| SGLang TP2 + DFlash2 | 2567 | 61.2 s | 94 / 437 | 1408 | 6299 / 23257 | 97.0% |
| SGLang TP1 + MTP（单卡） | 3966 | 未完成 | 50 / 251 | 1197 | 30286 / 36598 | 95.5% |
| vLLM TP1（单卡） | 3861 | 58.7 s | 38 / 317 | 2598 | 3371 / 16094 | 97.0% |
| **vLLM 2×TP1，各并发 12** | — | — | — | **3836** | 1030 / 7749 | — |
| vLLM 2×TP1，各并发 24 | — | — | — | 4774 | 6178 / 18390 | — |

prefill 单位 tok/s，decode 单位 tok/s（16 路为总输出）。2×TP1 两行是两个副本同时压测的合计，与上面 1 个 TP2 在并发 24 下对比。

- **单卡 prefill 比 TP2 快约 40%**：一张卡算完整个模型，比两张卡各算一半再经 PCIe 同步更快。TP2 只在单路 decode 上占优（显存带宽翻倍），58 对 38 tok/s。
- **2×TP1 同并发下总吞吐 +63%，TTFT P99 从 17 s 降到 7.7 s。** 各加到并发 24 还能涨到 4774 tok/s，但排队让 TTFT 回到与 TP2 相近。
- **SGLang 在这个负载下不占优**：低并发 decode 更快（DFlash2 单路 94 tok/s），但并发 24 时 TPOT 约 195 ms，比 vLLM 慢 66%。Qwen3.8 的 GDN 状态槽按请求预留，TP1 下 `--mamba-full-memory-ratio 2` 只够 7 路并发；TP2 是 33 路（MTP）和 23 路（DFlash2）。调大 ratio 会挤掉 KV，在单卡上撑不起 262k 上下文。
- 质量：五组 GSM8K 在 95.5%–97.0%，200 题的 1 个百分点相当于 2 题，视为持平。

## vLLM TP1 起不来的原因

单卡第一次按生产参数（`--max-num-seqs 64`）启动，gmu 0.92 和 0.86 都在 CUDA graph 显存预估时 OOM：权重 29 GB，再按 64 路捕获 graph，超出 48 GB。与上下文长度和 gmu 无关。

改成 `--max-num-seqs 32 --compilation-config '{"max_cudagraph_capture_size":32}'` 后，262k 上下文正常启动：gmu 0.92 时 KV 池 357k token，0.88 时 302k token（单个 262k 请求的 1.15 倍）。

## 推荐与代价

候选部署：4 个 TP1 副本（DP4），每个 `--max-num-seqs 32`、graph 捕获上限 32、262k 上下文，Router 挂 4 个 worker。按两卡结果外推，批处理吞吐约为现在 2×TP2 的 1.6 倍。

代价：

- 单路 decode 从约 58 降到约 38 tok/s，交互式对话体感变慢。
- 单卡 KV 池只有约 1.15 个 262k 请求：一个超长请求会占满所在副本，其他请求只能排到别的副本。
- GPU 0、2 上还有 embedding 和 reranker（各约 4.4 GB），这两张卡的 gmu 要取 0.88。

四卡验证还没做：需要停掉全部副本，另约时间。
