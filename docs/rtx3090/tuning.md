# RTX 3090 + HyperQwen：150k 上下文下的最佳配置

2026-10-30 实验。前提：保留 150k 上下文，对话质量不明显下降；在此之下尽量提高 prefill 和 decode 速度。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | 1× RTX 3090 24G，功耗上限 350W（HyperQwen 官方数据是 250W） |
| 引擎 | HyperQwen 镜像 `ghcr.io/syv-ai/hyperqwen:latest`（vLLM 0.31.0 补丁版），单用户模式 `single` |
| 模型 | Qwen3.8-27B-W4A16（AutoRound，fast 变体：int4 lm_head 与 MTP） |
| 公共参数 | `MTP_DRAFT_VOCAB=0 VISION=1 PREFIX_CACHE=1`，max-num-seqs 8，mbt 2048 |

启动方式（只换 `-e` 参数）：

```bash
podman run -d --name hq --device nvidia.com/gpu=0 --network host --ipc host \
  -v hq-cache:/cache -v /path/to/hyperqwen-models:/app/models \
  -e MTP_DRAFT_VOCAB=0 -e VISION=1 -e PREFIX_CACHE=1 \
  -e SPEC=mtp -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8 \
  ghcr.io/syv-ai/hyperqwen:latest single
```

## 结论

| 组 | 配置 | prefill 16k / 50k / 128k tok/s | decode 短 / 32k 上下文 tok/s | PPL（相对 a） | GSM8K-200 |
| --- | --- | --- | --- | --- | --- |
| a | `SPEC=mtp CTX=long`（原配置） | 1121 / 968 / 745 | 82.0 / 80.4 | 8.405 | 96.0% |
| b | a + `INT8_ACT=int8 PREFILL_ATTN=int8` | **2208 / 1638 / 1032** | 87.5 / 70.8 | 8.718（+3.7%） | 96.0% |
| c | b，但 INT8 只用于 MLP（`INT8_LAYERS=mlp`） | 1764 / 1367 / 928 | 83.4 / 68.9 | 8.586（+2.2%） | 95.5% |
| d | b，投机改 `SPEC=dflash2` | 2150 / 1594 / — | **96.8** / 69.1 | 同 b | 同 b |

1. **推荐 b。** prefill 在 16k 翻倍，128k 快 39%；50k 输入的首 token 时间从 53s 降到 31s。代价是 PPL +3.7%、长上下文 decode 慢约 12%；GSM8K 不变。一次 30k 上下文、输出 500 token 的 agent 轮次，按上表算约 21s，a 约 33s。
2. **c 是保守选项。** PPL 代价少一半（+2.2%），prefill 提升也少一半左右。
3. **d 不满足 150k。** dflash2 下 HyperQwen 把上下文上限设为 131072，128k 请求返回 400。它的短对话 decode 最快（97 tok/s），但 4 路并发时最慢一路只有 57 tok/s。
4. `CTX=long`（150k）和默认 `CTX=fast` 的 prefill 差距在 5% 以内（见下面的矩阵），保留长上下文几乎不花 prefill 的代价。

INT8 组长上下文 decode 变慢的原因没有查。

## 和官方数据对照

官方 README 的 1800 tok/s 来自 `INT8_ACT=int8` 的单用户模式（1k 输入 1845，4k 1937），我们原配置没开这个选项。开了以后，同样单请求测法：

| 输入 | 1 long | 2 fast | 3 fast+INT8_ACT | 4 fast+INT8_ACT+PREFILL_ATTN | 5 long+INT8_ACT+PREFILL_ATTN | 官方 W4A16 / INT8_ACT |
| --- | --- | --- | --- | --- | --- | --- |
| 1k | 1258 | 1216 | 2448 | 2433 | 2415 | 1437 / 1845 |
| 4k | 1190 | 1226 | 2561 | 2553 | 2430 | 1494 / 1937 |
| 16k | 1161 | 1204 | 2278 | 2291 | 2235 | 1410 / 1791 |
| 50k | 985 | 1036 | 1748 | 1779 | 1640 | 1200 / 1423（51k） |

- INT8_ACT 下我们比官方快约 30%，可能是功耗（350W 对 250W）。
- 不开 INT8 时我们比官方慢 15–20%。官方这组用的是 dflash2，我们是 MTP，没有单独验证。
- `PREFILL_ATTN=int8` 单独加上去在 ±2% 以内，和官方说法（「上下几个百分点」）一致。

## 方法

- prefill：[prefill_probe.py](../../scripts/prefill_probe.py)，随机 token id、`max_tokens 4`，单请求，每档 2–3 轮取中位数。
- decode：[decode_probe.py](../../scripts/decode_probe.py)，[data/real_prompts.jsonl](../../data/real_prompts.jsonl) 的真实提示词，thinking 开，temperature 0.6 / top_p 0.95，`max_tokens 512`。32k 上下文时在提示词前拼接本仓库文档（实际约 28k token）。单路 decode = (输出 token − 1) / (E2E − TTFT)。
- 质量：HyperQwen 自带的 `bench/quality_battery.py`，PPL（wikitext-2、fineweb-2 丹麦语、vLLM 源码，共 32,645 token）与 GSM8K 200 题（greedy，thinking 关）。数据从 Hugging Face 官方下载；GSM8K 并发从 32 改为环境变量 `GSM_CONC`。
- 代码 PPL 的语料是镜像里 vLLM 0.31.0 的 `v1/core`（8 个文件），和官方表不同，**PPL 只在本页各组之间比较**。GSM8K 200 题的标准误约 1.4 个百分点，各组的差异在噪声内。

原始数据：[results/rtx3090/](../../results/rtx3090/)。
