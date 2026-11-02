# 决策手册：4× RTX 4090 48G 部署 Qwen3.8-27B

把 9 月到 11 月的实验结论压成选型规则。每条结论后附出处，数字以出处为准。

## 默认部署

四卡各一个 HyperQwen 实例（DP4），single + MTP + PTH，32 槽，CG128，Router `power_of_two`。

| 指标（整机，经 Router） | 数值 | 出处 |
| --- | --- | --- |
| 对话 1k 入 / 256 出，合计 tok/s | c64 约 11,000，c128 约 13,200（11-02 三轮均值，单轮最高 11,800 / 14,100）；TTFT P99 ≤ 5 s 时约 85 路并发 | [final.md#11-02-三轮复测](final.md#11-02-三轮复测) |
| 单路 decode | 约 120 tok/s | [final.md#槽位数与-cuda-graph](final.md#槽位数与-cuda-graph) |
| 单卡纯 decode 峰值（c32） | 1,338 tok/s | 同上 |
| 16k / 64k prefill，单请求 | 约 4,500 / 3,300 tok/s | [final.md#长-prefill-切块](final.md#长-prefill-切块) |
| 128k / 200k 单请求 TTFT | 约 58–60 s / 116–119 s | [final.md#长文档](final.md#长文档)、[final.md#11-02-三轮复测](final.md#11-02-三轮复测) |
| 公开测试集与原 FP8 的差距 | 配对检验不显著；DocVQA 约 -0.7 ANLS | [final.md#公开测试集](final.md#公开测试集) |
| 对比原部署（2×TP2 FP8） | DP4 batch 已是 2.2–2.8 倍（批处理负载），全 single 再 +18–52%；单路 decode 58 → 120 | [dp4.md](dp4.md)、[final.md](final.md) |

## 选型规则

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| TP2 还是单卡 DP | 单卡 DP。PCIe 卡上 TP2 只在单路 decode 占优，prefill 慢约 40%，总吞吐低 | [engine-tp.md](engine-tp.md) |
| vLLM FP8 还是 HyperQwen W4A16 | HyperQwen。单卡批处理快约 25%，KV 池更大，质量差距在噪声内 | [tp1-hyperqwen.md](tp1-hyperqwen.md)、[final.md](final.md) |
| single 还是 batch 模式 | single + MTP + PTH。高并发吞吐 +18–52%，单路 decode 2 倍 | [final.md#single-与-batch四卡同模式对比](final.md#single-与-batch四卡同模式对比) |
| 槽位与 CUDA graph | `MAX_SEQS=32 CG=128`。CG 要覆盖 槽位 × (草稿数 + 1)；16 槽并发过 16 就排队 | [final.md#槽位数与-cuda-graph](final.md#槽位数与-cuda-graph) |
| 投机解码方法 | MTP（3 草稿）。DFlash2 单路更快，但长 prefill 和高并发不如 MTP | [profiles.md](../profiles.md) |
| 4090 开 MTP 必须加什么 | `PTH_FA=1`。否则 FlashInfer 降为 PIECEWISE graph，单路从 121 掉到 66 | [profiles.md#4090-慢的原因与修正10-31-夜补测](../profiles.md#4090-慢的原因与修正10-31-夜补测) |
| Router 策略 | `power_of_two`。cache_aware 在共享系统提示词时把请求全压到一张卡，吞吐只剩 37% | [dp4.md](dp4.md) |
| `max-num-batched-tokens` | 默认 2048。调大对 prefill 影响不到 3% | [prefill.md](prefill.md)、[final.md#长-prefill-切块](final.md#长-prefill-切块) |
| 长文档与对话混跑 | 对话 P99 会放大 3–5 倍。若必须混跑，试 `--max-num-batched-tokens 8192 --long-prefill-token-threshold 2048` | [final.md#长-prefill-切块](final.md#长-prefill-切块) |
| 功耗受限 | 300 W：功耗 -23%，decode -13%，prefill -9%。不要低于 300 W | [final.md#功耗](final.md#功耗) |
| 同卡放 embedding / reranker | 可以。满载时延迟 P99 约 100–116 ms；那张卡 `GPU_UTIL` 降到 0.80–0.82 | [final.md#混合负载与同卡混部](final.md#混合负载与同卡混部) |
| Router 健康检查参数 | 默认即可。快速参数只把意外崩溃后的接流从 117 s 缩到 11 s，失败数不变；c32 时约 6 个已开始输出的流式请求会中断，重试接不回 | [final.md#故障切换](final.md#故障切换) |
| SGLang | 这台机器不用。并发 24 时 TPOT 比 vLLM 慢 66% | [engine-tp.md](engine-tp.md) |

## 不要做的事

- **不要用固定分差判质量。** GSM200 单跑有 ±3 题抖动，1400 题的 MMLU-Pro 一个标准误约 1.2 个点。要看多跑均值或逐题配对检验。
- **评测长上下文时不要把输出上限压到 2048。** thinking 关闭时模型也会先写长分析，上限 2048 让 HyperQwen 在 LongBench 上假性落后 3–6 个点（[final.md#公开测试集](final.md#公开测试集)）。
- **不要信 random token 的 MTP 接受率。** 真实文本接受长度更高，投机解码的收益要用真实提示词测（[baseline.md](baseline.md)）。
- **不要指望 DP 缩短单个长文档的 TTFT。** 一个请求只在一张卡上，128k 约 60 s 与 TP2 时代相近；DP 增加的是并发容量。
- **不要用 cache_aware 处理共享前缀的批量请求。**

## 未覆盖

- **TP4、PP2：** 不再做。PCIe 卡上 TP2 已经输给单卡，TP4 的同步开销只会更大；PP 的气泡在单请求场景无收益，长文档 TTFT 的瓶颈在单卡算力，应该靠 context parallel 或更快的 prefill kernel 解决。
- **真实业务流量对比：** 截至 11-02 没有业务流量，切换后的表现等内测反馈。
- **负载感知路由：** 后端 `/get_load` 返回 404，`power_of_two` 实际不看负载。长文档多时会叠到同一张卡，需要 Router 或后端补负载上报。
- **生产 Router 的重试语义：** 故障演练用的是单独起的 Router，生产实例的重试参数未单独验证。

## 复现清单

| 项 | 值 |
| --- | --- |
| 驱动 | 595.84 |
| 推理镜像 | `ghcr.io/syv-ai/hyperqwen`，2026-10-30 构建（镜像 ID `sha256:e69714eab977…`），vLLM 0.32.0 |
| 模型 | `Qwen3.8-27B-W4A16-AutoRound`（HyperQwen 自带下载脚本），MTP 头随模型 |
| 参考 FP8 | `vllm/vllm-openai:v0.25.1` + `Qwen/Qwen3.8-27B-FP8` |
| Router | vllm-router 0.1.15，`--policy power_of_two --max-payload-size 67108864 --request-timeout-secs 1800` |
| 评测 | evalscope 1.14.1（`pip install 'evalscope[ifeval]'`），[experiments/ext-20261101/pubeval.py](../../experiments/ext-20261101/pubeval.py) |

单卡启动（GPU1 为例，`/models` 下有 `hyperqwen/models` 与每卡一个编译缓存目录）：

```bash
docker run -d --name vllm-hq-g1 --restart always --ipc host --gpus '"device=1"' -p 127.0.0.1:8003:8000 \
  -v /models/hyperqwen/models:/app/models -v /models/hyperqwen/cache-gpu1s:/cache \
  -e HOME=/cache -e HOST=0.0.0.0 -e PORT=8000 -e PREPARE=0 -e VERIFY=0 -e PREFIX_CACHE=1 -e VISION=1 \
  -e SPEC=mtp -e CTX=long -e PTH_FA=1 -e DRAFT_TOKENS=3 -e MTP_DRAFT_VOCAB=0 \
  -e MAX_LEN=262144 -e MAX_SEQS=32 -e CG=128 -e INT8_ACT=int8 -e PREFILL_ATTN=int8 \
  -e GPU_UTIL=0.92 -e KV_MEM= \
  -e EXTRA_ARGS="--served-model-name Qwen3.8-27B-FP8 $VIS_ARGS" \
  ghcr.io/syv-ai/hyperqwen:latest single
```

`VIS_ARGS` 见 [profiles.md#启动参数](../profiles.md#启动参数)。`--served-model-name` 沿用替换前的名字，客户端不用改。同卡有 embedding / reranker 时 `GPU_UTIL` 取 0.80 / 0.82。上线前过单图 OCR 和 5 图请求再接入 Router，切换脚本 [g-switch.sh](../../experiments/ext-20261101/g-switch.sh) 按这个流程逐卡执行。
