# DP4：四个单卡副本 + Router（2026-11-01）

> 本文的部署是 11-01 白天的中间状态（3 batch + 1 single）。最终部署见 [final.md](final.md)。

10-31 先把 GPU2、3 换成 HyperQwen 单卡实例（[tp1-hyperqwen.md](tp1-hyperqwen.md)），11-01 凌晨把剩下的 vLLM FP8 TP2（GPU0,1）也拆成两个单卡实例，四张卡全部单卡 DP，Router 挂 4 个 worker。本文是整机实测：Router 级吞吐、路由策略、长上下文质量。

## 部署

| 卡 | 模式 | GPU_UTIL | KV 池（token） | 备注 |
| --- | --- | --- | --- | --- |
| GPU0 | batch | 0.80 | 696,198 | 同卡有 Qwen3-Embedding（约 5.3G） |
| GPU1 | batch | 0.92 | 883,666 | |
| GPU2 | batch | 0.82 | 约 73 万 | 同卡有 Qwen3-Reranker |
| GPU3 | single + MTP + PTH，CG32，16 槽 | 0.92 | 798,768 | 灰度，配置见 [profiles.md](../profiles.md) |

batch 实例参数：`KV=fp8 INT8_ACT=int8 MAX_LEN=262144 PREFIX_CACHE=1 VISION=1`，EXTRA_ARGS 带 5 图 / 8192 图像 token 的视觉参数。每个实例上线前必须过单图 OCR + 5 图请求，再接入 Router。替换与回滚脚本见 [experiments/dp4-20261101/](../../experiments/dp4-20261101/)。

## Router 级吞吐

负载同 10-31：random 2048 入 / 160 出（批处理），另加 16k 入 / 256 出（长 prefill）和 8192 共享前缀 + 1024 入 / 160 出（同一系统提示词的批量请求）。合计 = 输入 + 输出 token / 时长。全部 0 失败。

| 负载 | 并发 | power_of_two 合计 tok/s | cache_aware 合计 tok/s | TTFT P99 ms（p2c / ca） | TPOT P99 ms（p2c / ca） |
| --- | --- | --- | --- | --- | --- |
| 批处理 | 48 | 10,492 | 11,089 | 6,805 / 5,794 | 118 / 67 |
| 批处理 | 96 | 11,834 / 12,935 | 13,017 / 13,128 | 11,918–13,119 / 11,559–11,693 | 161–221 / 103–104 |
| 16k 长 prefill | 16 | 9,497 / 10,057 | 11,699 / 12,307 | 22,256–22,304 / 16,138–17,017 | 185–191 / 78–83 |
| 共享 8k 前缀 | 48 | **41,376** | 15,296 | 9,739 / 25,212 | 86 / 182 |

两格的数字是两轮（第一轮先 p2c，第二轮先 ca，排除顺序影响）。原始数据 [results/rtx4090/20261101-dp4/](../../results/rtx4090/20261101-dp4/)。

- **对比旧部署：** 2×TP2 FP8 合计约 4,700 tok/s（[engine-tp.md](engine-tp.md)），DP4 HyperQwen 在 c48 已到 10,492，c96 约 12,000–13,000，约 **2.2–2.8 倍**。来源是两层叠加：TP1 比 TP2 每卡快（省掉 PCIe 同步），HyperQwen W4A16 + INT8 激活比 FP8 单卡批处理再快约 25%。
- **路由策略：** 无共享前缀时 cache_aware 略好，长 prefill 高 20%，TPOT 长尾明显更低。原因是这四个后端的 `/get_load` 都返回 404，power_of_two 拿不到真实负载，相当于随机挑两个比较；cache_aware 在前缀不匹配时按 Router 自己记的在途数做均衡。
- **但共享前缀时 cache_aware 只有 power_of_two 的 37%：** 288 个请求全部路由到同一张卡（且正好是 KV 最小的 GPU0），其余三张卡空闲。09-30 两后端时已见过同样现象（[baseline.md#router](baseline.md#router)），四后端更严重。资料库批处理请求大多带同一段系统提示词，**生产保持 power_of_two。**

## 长上下文质量

[scripts/longctx_probe.py](../../scripts/longctx_probe.py)：wikitext-2 真实文本做干草堆，32k / 64k / 128k / 200k 每档 5 题，thinking 关、temperature 0，同一 seed 三种部署同题对比。

- 多键检索（随机深度藏 8 条 7 位数访问码，问其中 4 条）：

| 部署 | 32k | 64k | 128k | 200k（实际 202k token） |
| --- | --- | --- | --- | --- |
| vLLM FP8 TP2（原生产） | 100% | 100% | 100% | 100% |
| HyperQwen batch（fp8 KV） | 100% | 100% | 100% | 100% |
| HyperQwen single + MTP + PTH（int8 KV） | 100% | 100% | 100% | 100% |

- 变量追踪（4 跳赋值链 + 2 条交错干扰链，问哪些变量等于目标值）：三种部署平均 Jaccard 0.40–0.50，完全正确 1–4 / 20，且 32k 并不比 128k 好，同一题在三种部署间 0.2 到 1.0 乱跳。这项在 thinking 关闭时测到的是模型多跳推理的上限，不区分部署，不作结论。
- 200k 单请求端到端约 210–226 s（并发 2），三种部署差距在 10% 内。

结论：W4A16 权重、INT8 激活、int8 / fp8 KV 和 MTP 在 200k 以内没有可测到的检索损失。

## 后续

- 扩大 CUDA graph 捕获尺寸后 GSM8K 低于 95% 的现象，复跑 3 次后判为噪声（[profiles.md](../profiles.md)）。
- 同日晚上四卡全部切到 single + MTP + PTH（32 槽，CG128），公开测试集、容量与切换记录见 [final.md](final.md)。
