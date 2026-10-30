# 基线调优（2026-09-29～30）

把一套「能跑」的部署调到「合理可用」。环境见 [README](../README.md#环境)。原部署的额外参数只有 `--max-num-seqs 6`，其余见 [deploy_llm.sh](../scripts/deploy_llm.sh)。

## 定稿

```bash
--max-num-seqs 64 --enable-prefix-caching \
--speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
--gpu-memory-utilization 0.78
```

Router 升级到 vllm-router 0.1.15，加 `--max-payload-size 67108864`，策略保持 `power_of_two`。

| 指标 | 原部署 | 定稿 |
| --- | --- | --- |
| 峰值输出吞吐（并发 64，in1k/out256） | 284 tok/s | 681 tok/s（2.4×） |
| 并发 32 的 TTFT P50 | 18.9 s | 0.8 s |
| 真实文本单流 TPOT P50 | 30.6 ms | 17.4 ms |
| 200k 上下文经 Router | 全部 413 | 可用 |

## 方法

每组实验重建两个后端，经 Router 跑同一套场景（[suite.sh](../scripts/suite.sh)）：

- `sweep-c{1..64}`：in1k/out256 吞吐-并发扫描。
- `chat` 512/256、`rag` 4k/512、`summ` 16k/1k、`longout` 1k/2k：典型长度。
- `prefix-c16`：共享 4k 前缀，评估 prefix caching。
- `real-c16`/`real-c1`：[合成提示词](../data/real_prompts.jsonl) + chat 模板 + thinking，评估 MTP 这类与内容相关的优化。random token 的 MTP 接受率偏低，不能代表真实文本。
- full 模式另加 64k/128k/200k 长上下文与 embedding/reranker。

压测客户端是 `vllm bench serve`，常驻容器，`--network host`。每轮开跑前检查后端没有外部在途请求。

## 参数实验

结果：[results/20260929-param-sweep/](../results/20260929-param-sweep/)（e1–e4、base、final）与 [results/20260929-30-llmbench/runs/](../results/20260929-30-llmbench/runs/)（MTP1、gmu）。

| 实验 | 参数 | 结论 |
| --- | --- | --- |
| e1-seq32 | seqs 32 | 并发 32 吞吐 255→506 tok/s，TTFT P50 18.9s→0.56s |
| e1-seq64 | seqs 64 | 并发 ≤32 与 seqs32 相同；并发 64 多 7.5% |
| e2 | + prefix caching | 随机 prompt 无变化；共享 4k 前缀 TTFT P50 3.1s→0.65s |
| e3 | seqs64 + MTP2 | 真实文本单流 TPOT 30.6→17.2ms，并发 16 吞吐 +34%；并发 64 持平，TPOT P99 110→176ms |
| e4 | + max-num-batched-tokens 8192 | 全场景无收益或略差，未采用 |
| MTP1 | num_speculative_tokens 1 | c64 TPOT P99 162→130ms，但单流和吞吐全面变差（longout 692→494 tok/s），未采用 |
| gmu 0.78 | 0.82 → 0.78 | 两次复测与 0.82 一致；同卡 reranker 所在 GPU 余量 1.5G→4G（压测峰值） |

要点：

- **max-num-seqs 6 是首要瓶颈**。每后端 KV 池约 124 万 token，却只允许 6 个在途序列，并发 ≥16 时多出来的请求全在排队。
- **Prefix caching 默认关闭**（混合注意力模型）。开启后走 Mamba `align` 模式（上游标注实验性），注意力块对齐到 1568 token，短于 1568 的共享前缀不会命中。
- **MTP**：random token 接受率 58–63%；真实文本 real-c16 接受长度 2.17。收益以真实文本为准。
- **gmu 0.78 第一次测出下降是噪声**：random token 场景的 MTP 接受率波动大，单次结果不足以判断，需要复测。
- **长 prompt 的 TTFT 由 prefill 算力决定**：16k 输入 TTFT P50 约 7s，128k 约 68s，调 batched tokens 无收益。这是 TP2 + PCIe 下的现状，更深入的调度见 [roadmap](roadmap.md)。

## 长稳与混部

50 分钟：LLM 1.0 req/s Poisson（输入 2k±50%，输出 384），同时 embedding 20 req/s、reranker 10 req/s。

| 服务 | 请求数 | 失败 | 延迟 |
| --- | --- | --- | --- |
| LLM（经 Router） | 3000 | 0 | TTFT P50/P99 1.2s / 4.4s；TPOT P50/P99 38.8 / 103ms |
| Embedding | 60000 | 0 | E2E P50/P99 30 / 63ms |
| Reranker | 30000 | 0 | E2E P50/P99 24 / 48ms |

无抢占，容器零重启；两个后端 token 分流约 47:53；KV 使用率平均 10%、峰值 38%；GPU 峰值 62℃，210–260W/卡。TPOT P99 103ms 超过暂定的 80ms，主要来自 MTP 在大 batch 下的长尾。

## Router

结果：[runs/20260930-*](../results/20260929-30-llmbench/runs/)、[failover-20260930-085854/](../results/20260929-30-llmbench/failover-20260930-085854/)。

**请求体上限**：0.1.12 有约 2MB 硬上限，`--max-payload-size` 无效，超过返回 413；直连后端 20MB 正常。200k token 的中文请求经 `\uXXXX` 转义后更容易超限。0.1.15 修复。

**cache_aware vs power_of_two**（quick 套件，两次对比）：

| 场景 | 指标 | power_of_two | cache_aware |
| --- | --- | --- | --- |
| prefix-c16 | out tok/s | 335 | 251（-25%） |
| prefix-c16 | TTFT P99 ms | 6910 | 10426 |
| prefix-c16 | 后端平均 running（a / b） | 5.2 / 5.8 | 11.4 / 0.0 |
| prefix-c16 | prefix 命中率 | 67% | 68% |
| real-c16 / sweep-c64 | out tok/s | 634 / 690 | 632 / 709 |

cache_aware 把共享前缀的请求全部路由到同一个后端，另一个闲置。只有 2 个后端时，两边各自都能缓存住公共前缀，集中路由只换来 1 个百分点命中率。后端很多、前缀种类远超单后端缓存容量时才值得考虑。

**故障摘除**：16 并发、短输出、600s，T+30s kill 一个后端，T+121s 再 start。

| 阶段 | 请求 | 失败 | P50 / P99 |
| --- | --- | --- | --- |
| 稳态 | 464 | 0 | 1.04 / 1.83s |
| kill 后（单后端） | 1177 | 0 | 1.21 / 1.97s |
| start 后到 /health 200（T+344s） | 2850 | 0 | 1.21 / 1.90s |
| 健康后到重新接流（T+447s） | 1350 | 0 | 1.20 / 2.15s |
| 恢复双后端 | 2113 | 0 | 1.03 / 1.73s |

- kill 瞬间失败的 12 个请求全部由 Router 重试到另一个后端，客户端 0 失败。
- 摘除靠熔断器立即 open；健康检查（默认 60s×3）来不及参与，`active_workers` 始终是 2。
- 熔断器每 60s 做一次 half_open 探测，后端就绪后约 104s 才重新接流。可调小 `--cb-timeout-duration-secs`，未实测。
- 这是低负载短输出场景，高负载下单后端会更早饱和。

## 编译缓存

结果：[compile-cache-20260930-091210/](../results/20260929-30-llmbench/compile-cache-20260930-091210/)。只重启一个后端，另一个继续服务。

| 轮次 | docker start → /health 200 | init engine | 其中 compilation | 主模型 warmup |
| --- | --- | --- | --- | --- |
| 热（缓存在） | 138s | 29.6s | 11.2s | 1.7s |
| 冷（移走缓存） | 317s | 213.0s | 106.5s | 72.7s |
| 热（恢复缓存） | 136s | 29.3s | 10.9s | 1.2s |

vLLM 的编译缓存有三层：

| 内容 | 默认位置 | 丢失代价 |
| --- | --- | --- |
| torch.compile 图、AOT | `~/.cache/vllm/torch_compile_cache` | 约 90s 编译 |
| inductor kernel | `/tmp/torchinductor_<user>` | 约 70s warmup |
| triton kernel | `~/.triton/cache` | 计入上一项 |

后两层默认在容器可写层：`docker stop/start` 能保留，重建容器（rm + run）就丢。所以只挂载 `~/.cache/vllm` 时，重建后 warmup 仍要约 70s。[deploy_llm.sh](../scripts/deploy_llm.sh) 设置 `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR` 指到已挂载的缓存卷里：

| 重建方式（rm + run → 就绪） | 耗时 | init engine | warmup |
| --- | --- | --- | --- |
| 只挂图缓存 | 约 234s | 115.5s | 72.3s |
| 三层都挂（两个后端） | 170s / 160s | 42.9s / 29.0s | 4.0s / 1.5s |

缓存键包含模型、参数和 vLLM 版本，参数变了会生成新的 hash 目录。冷启动新生成约 1G；累积到 6.6G 时大部分是历史配置的旧目录。

## 运维教训

- 滚动重建一次只动一个后端（`ONLY=<容器名>`），另一个和 Router 继续服务。
- 后端重启后首批请求 13–16s（预热），未失败。就绪后先预热再接流量，**未实现**。
- 改系统服务参数（Router 的 systemd override）的脚本必须有白名单校验、写入前备份、就绪检查失败自动恢复。曾因 hook 里的 `{policy}` 占位符没渲染，sed 写出 `--policy power_of_two{policy}`，Router 反复重启约 8 小时无人发现。
