# 单卡 TP1 调参与 HyperQwen（2026-10-31）

接 [engine-tp.md](engine-tp.md)：两卡实验显示 2×TP1 比 1×TP2 吞吐高约 60%。这一轮在单张 4090（GPU3）上把 TP1 调到最好，再与 [HyperQwen](https://github.com/syv-ai/hyperqwen) 的 W4A16 量化版对比，最后把视觉和 MTP 纳入考虑，并替换了生产上的一个 TP2 副本。

负载与前文一致：批处理形态（输入 2048 / 输出 160，144 条，并发 24），单位为总 tok/s。原始数据在 [results/rtx4090/20261031-tp1-hyperqwen/](../../results/rtx4090/20261031-tp1-hyperqwen/)，脚本为 [experiments/tp1-tune-20261031.sh](../../experiments/tp1-tune-20261031.sh)、[hq-ab-20261031.sh](../../experiments/hq-ab-20261031.sh)、[hq-vision-ab-20261031.sh](../../experiments/hq-vision-ab-20261031.sh)。

## vLLM FP8 单卡调参

基线：vLLM 0.25.1，FP8，`--max-num-seqs 32`，MTP2，262k 上下文。

| 配置 | 批处理 tok/s | TTFT P50 / P99 ms | 单路 decode |
| --- | --- | --- | --- |
| 基线 MTP2 | 2547 | 3530 / 16824 | 38 |
| + 调优 FP8 kernel | 2464 | 3594 / 17472 | — |
| mbt 4096 | 2598 | 3451 / 17973 | — |
| max-num-seqs 48 | 2664 | 3341 / 16095 | — |
| MTP1 | 2621 | 1193 / 11951 | — |
| **不开投机解码** | **2824** | 1012 / 11097 | 31 |
| vLLM 0.31 | 1839 | 6030 / 37360 | — |
| vLLM 0.31 + 调优 kernel | 2002 | 5941 / 23704 | — |

- 单卡并发 24 时 MTP 是负收益：关掉后吞吐 +11%，TTFT P99 从 16.8 s 降到 11.1 s，代价是单路 decode 38 → 31 tok/s。
- kernel 调优、mbt、seqs 都在噪声范围内（±5%）。
- vLLM 0.31 在这张卡上慢约 30%，16k prefill 从 3857 降到 2296 tok/s。暂不升级。

## HyperQwen 量化对比

镜像 `ghcr.io/syv-ai/hyperqwen`（batch 模式），同为单卡 262k 上下文。

| 配置 | 批处理 c24 / c48 | 16k prefill | 单路 decode | 124k TTFT | GSM8K | KV 池 token |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM FP8，不开投机 | 2824 / — | 4008 | 31 | — | — | 357k |
| **W4A16 + INT8 激活 + fp8 KV** | **3268 / 3621** | 3729 | 55 | 58.7 s | 94.5% | 902k |
| W4A16 + fp8 KV | 2553 / 2730 | 2784 | 57 | 70.3 s | 95.0% | 911k |
| W4A16 + INT8 激活 + kvarn | 1181 / 1269 | — | — | 53.4 s | 97.0% | 1.88M |

- INT8 激活这一档比 vLLM FP8 单卡吞吐 +16%，单路 decode 快 77%，KV 池是单个 262k 请求的 3.4 倍（FP8 同 gmu 0.92 下是 1.36 倍）。16k prefill 低 7%。
- GSM8K 94.5%，比 vLLM FP8 单卡（[engine-tp.md](engine-tp.md) 中 97.0%）低 2.5 个点，200 题里相当于 5 题。后面视觉那组复测是 96.0%，视为轻微下降。
- kvarn 把 KV 池翻倍，但吞吐只剩三分之一，只适合超长上下文、低并发的场景。

## 视觉与 MTP

HyperQwen 的 batch 模式默认带 `--language-model-only`，**图片请求直接返回 400**（`At most 0 image(s)`）。`VISION=1` 打开后默认每请求 1 张图、单图 2048 token。我们的客户端单次最多发 5 张图、不做缩放，所以需要通过 `EXTRA_ARGS` 覆盖：

```bash
-e VISION=1 -e EXTRA_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
```

`longest_edge` 是像素上限，约等于 token 上限 × 1024。

OCR 质量用合成中文图测（[scripts/gen_ocr_set.py](../../scripts/gen_ocr_set.py) 生成 A4 扫描、带表格截图、手机拍照三类，[scripts/ocr_probe.py](../../scripts/ocr_probe.py) 算字符错误率 CER），12 张，并发 4：

| 配置 | CER | 单图 token | 12 张耗时 | KV 池 | 批处理 c24 | 5 图请求 |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM FP8，不限分辨率 | 0.05% | 3000–12600 | 189 s | — | — | — |
| HyperQwen 2k 上限 | 0.17% | ~2200 | 67 s | — | 3267 | 正常 |
| HyperQwen 4k 上限 | 0.08% | — | — | 898k | 3262 | — |
| **HyperQwen 8k 上限** | **0.04%** | ~8300 | 92 s | 884k | 3270 | 正常 |
| 8k 上限 + MTP1 | 0.07% | — | — | 814k | 3240 | 断连 |

- 打开视觉对文本吞吐没有影响，KV 池只少 2%。
- 8k 上限的 CER 与 vLLM 不限分辨率持平，耗时只有一半。选 8k。
- MTP1 在 c24 持平、c48 降 8%（3621 → 3315），KV 池少 8%，5 图请求断连，单路 decode 反而从 55 降到 51 tok/s。batch 模式下不采用。
- 这组 MTP 只是 batch 模式里用 `--speculative-config` 硬加的 k=1，模型是基础版。RTX 3090 上单路约 85 tok/s 的是 single 模式：fast 变体（int4 lm_head 与 MTP 头、自带 draft 词表）、k=3、split-KV verify、max-num-seqs 8。10-31 晚已在 4090 上补测，见 [../profiles.md](../profiles.md)。

## 生产替换

把一个 TP2 副本（GPU2,3）换成两个 HyperQwen 单卡实例，Router 挂三个 worker。GPU2 上有 reranker，`GPU_UTIL` 取 0.82（KV 727k），GPU3 取 0.92（KV 884k）。

第一次替换漏了视觉：约 2 分钟内 43 个请求有 20 个带图返回 400，摘除回滚。修正后的上线顺序：

1. 先 drain 旧副本（`/remove_worker`，等 running 归零）再停。
2. 新实例启动后检查模型名，跑两条文本预热。
3. **跑视觉冒烟（扫描 + 拍照各 1 张，再加 1 个 5 图请求），全部通过才 `/add_worker`。**
4. 回滚脚本提前写好：摘除新实例、停掉、恢复旧容器的 `restart=always` 并启动。

上线后第一个小时，两个实例各约 1600 个请求，0 个 4xx/5xx。另一个 TP2 副本按同样方式替换（GPU0 上有 embedding，`GPU_UTIL` 约 0.80）。
