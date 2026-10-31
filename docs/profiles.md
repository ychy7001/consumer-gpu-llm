# 单卡 profile：按场景选 single / batch（RTX 4090 与 RTX 3090，2026-10-31）

两张卡都是单卡 PCIe 部署 HyperQwen（Qwen3.8-27B W4A16 + INT8 激活）。这一轮用同一镜像、同一套测试在两张卡上各跑一遍 single 模式（低并发、带投机解码）和 batch 模式（高并发），用来回答：要不要按场景、按卡分 profile。

结论：**profile 按负载分，不按卡分；每张卡最多两套。** 卡之间只差 `MAX_LEN`、`GPU_UTIL` 等参数。

## 测试方法

镜像 `ghcr.io/syv-ai/hyperqwen`（10-31 的 latest），视觉统一开启（每请求 5 图、单图 8192 token）。每组测：

- prefill：输入 2k / 16k / 64k / 117k，单请求；
- decode：短上下文 1 / 4 / 16 并发，32k 上下文 1 / 4 并发，输出 512 token；
- 批处理：输入 2048 / 输出 160，并发 8 与 24；
- 质量：GSM8K-200；OCR 12 张合成中文图（扫描、截图、拍照）的字符错误率，外加一个 5 图请求。

脚本在 [experiments/cards-20261031/](../experiments/cards-20261031/)，原始数据在 `results/rtx4090/20261031-cards/` 与 `results/rtx3090/20261031-cards/`，每组一行汇总可用 [scripts/card_summary.py](../scripts/card_summary.py) 生成。

## RTX 4090 48G（262k 上下文，GPU_UTIL 0.92）

| 配置 | prefill 16k / 117k | decode 单路 | 4 并发每路 | 16 并发合计 | 32k×4 合计 | 批处理 c24 / TTFT P99 | GSM8K | KV 池 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch | 3714 / 2216 | 55 | 53 | **712** | 46 | 3259 / **11.8 s** | 95.0% | 884k |
| single，不开投机 | 4663 / 2516 | 56 | 54 | 401 | 54 | 2650 / 15.5 s | 95.5% | 892k |
| single + MTP，8 槽 | 4464 / 2438 | 65 | 58 | 413 | 56 | 3110 / 13.9 s | 95.0% | 804k |
| **single + MTP，16 槽** | **4431 / 2430** | 64 | 60 | 440 | 56 | **3306** / 13.3 s | 96.0% | 808k |
| single + DFlash2 | 3376 / 987 | **115** | **99** | 383 | 41 | 3030 / 15.6 s | 95.0% | 670k |

## RTX 3090 24G（150k 上下文）

| 配置 | prefill 16k / 117k | decode 单路 | 4 并发每路 | 16 并发合计 | 32k×4 合计 | 批处理 c24 / TTFT P99 | GSM8K | KV 池 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch | 1775 / 985 | 47 | 45 | **583** | 24 | 1666 / 26.4 s | 95.0% | 200k |
| single，不开投机 | 2289 / 1123 | 47 | 45 | 340 | 29 | 1725 / 26.0 s | 95.0% | 209k |
| **single + MTP** | 2176 / 1091 | 86 | 77 | 516 | **30** | **1967** / 23.8 s | 95.0% | 181k |
| single + DFlash2（上限 131k） | 2159 / 1057 | **118** | 80 | 271 | 25 | 1622 / 31.3 s | 95.0% | 138k |

两张卡所有组的 OCR 字符错误率都在 0.04%–0.09%，5 图请求全部成功；GSM8K 在 95.0%–96.0%（200 题，1 个点是 2 题），视为持平。

## 选哪套

| 场景 | 负载特点 | 4090 | 3090 |
| --- | --- | --- | --- |
| 会议纪要、agent 交互 | 一两个用户，长输入 + 长输出 | single + MTP | **single + MTP（唯一推荐）** |
| 资料库批处理 | 高并发，长输入 + 短输出 | single + MTP 16 槽，或 batch | 不适合（KV 池太小） |
| 大量并发长输出 | 十几路同时生成长文 | batch | batch |
| 短上下文快问快答 | 低并发，短输入 | single + DFlash2 | single + DFlash2 |

- **3090 只要一套。** single + MTP 除 16 并发 decode（516 对 583）和 KV 池（181k 对 200k）外都不输 batch，单路 decode 快 1.8 倍（86 对 47）。KV 池 181k，一次只容得下一个长请求，高并发本来就不是它的场景。DFlash2 单路更快，但上下文上限 131072，不满足 150k。
- **4090 资料库：single + MTP 16 槽与 batch 基本打平。** 前者 prefill 快 19%、c24 吞吐持平，代价是 TTFT P99 多 1.5 s、16 并发 decode 只有 batch 的 62%。客户端输出短时选前者，并发生成长文多时选后者。线上目前是 batch，等下面 4090 的 MTP 问题查清再决定是否切。
- 不开投机解码在两张卡上都没有优势，不作为 profile。

## 尚未解决

- **4090 上 MTP 每步太慢。** 不开投机时每步约 18 ms（3090 约 21 ms，与带宽比一致）。开 MTP 后 4090 平均每步接受 3.29 个 token，比 3090 的 2.72 还高，但每步耗时约 50 ms，是普通步的 2.8 倍；3090 只有 1.5 倍。所以 4090 单路只有 65 tok/s，低于 3090 的 86。DFlash2 在 4090 上验证开销正常（单路 115），问题只在 MTP 路径。
- **4090 上 DFlash2 长上下文 prefill 掉到 MTP 的 40%**（117k：987 对 2438），3090 上没有这个问题（1057 对 1091）。HyperQwen 在算力 8.6 的卡上默认走 FA2 fp8 KV 插件，4090 是 8.9，退回 int8 + Triton。
- 两条都指向同一类原因：为 3090（sm_86）调的 kernel 在 4090（sm_89）上没生效。待查方向：4090 用 150k 上下文复测、草稿数降到 2、开关 `SPEC_ATTN`。

## 启动参数

3090，single + MTP（150k）：

```bash
-e SPEC=mtp -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8 -e MTP_DRAFT_VOCAB=0 \
-e VISION=1 -e PREFIX_CACHE=1 -e EXTRA_ARGS="$VIS_ARGS" ... single
```

4090，single + MTP 16 槽（262k）。single 模式默认按 24G 卡固定 KV 池字节数，48G 卡要置空 `KV_MEM` 改按 `GPU_UTIL` 分配：

```bash
-e SPEC=mtp -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8 -e MTP_DRAFT_VOCAB=0 \
-e MAX_LEN=262144 -e GPU_UTIL=0.92 -e KV_MEM= -e MAX_SEQS=16 \
-e VISION=1 -e PREFIX_CACHE=1 -e EXTRA_ARGS="$VIS_ARGS" ... single
```

4090，batch（线上配置）：

```bash
-e KV=fp8 -e INT8_ACT=int8 -e MAX_LEN=262144 -e GPU_UTIL=0.92 \
-e VISION=1 -e PREFIX_CACHE=1 -e EXTRA_ARGS="$VIS_ARGS" ... batch
```

其中 `VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'`。batch 模式不开 `VISION=1` 时默认 `--language-model-only`，图片请求会返回 400。
