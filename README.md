# Consumer GPU LLM Serving

在消费级 NVIDIA 显卡上部署 Qwen3.8-27B 的调优记录：部署脚本、压测套件、原始结果与结论。所有结论附原始数据，并明确区分「已验证」与「未验证」。

| 硬件 | 部署 | 文档 |
| --- | --- | --- |
| 4× RTX 4090 48G（PCIe） | 4 个 HyperQwen 单卡实例（DP4）+ Router | [docs/rtx4090/](docs/rtx4090/) |
| 1× RTX 3090 24G | HyperQwen，W4A16 + INT8 激活，150k 上下文 | [docs/rtx3090/tuning.md](docs/rtx3090/tuning.md) |

下文「环境」到「复现」都是 RTX 4090 这套。

> 本项目不是 NVIDIA、Qwen 或 vLLM 官方发布。数据来自一台生产混部机器（同机还有其他业务），主机名与路径已替换。

叙述版本写在知乎专栏（筹备中），按基线调优、并行方式对比、长上下文等分篇；仓库这边是原始数据和复现入口。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | 4× RTX 4090 48G，PCIe 互联 |
| 模型 | Qwen3.8-27B-FP8，`max-model-len 262144`，`kv-cache-dtype fp8` |
| 推理 | HyperQwen（`ghcr.io/syv-ai/hyperqwen`，基于 vLLM 0.32；3 卡 batch 模式，1 卡 single + MTP 灰度）；此前为 vLLM 0.25.1 FP8 |
| 拓扑 | 每卡一个 HyperQwen 实例（W4A16 + INT8 激活，2026-11-01 起 DP4）+ [vllm-router](https://github.com/vllm-project/router) 0.1.15，`power_of_two` |
| 同卡混部 | Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B |

## 状态

| 主题 | 状态 | 入口 |
| --- | --- | --- |
| 基线调优：max-num-seqs、prefix caching、MTP、gmu | 已验证（2026-09-29～30） | [docs/rtx4090/baseline.md](docs/rtx4090/baseline.md) |
| Router：策略、故障摘除、请求体上限 | 已验证 | [docs/rtx4090/baseline.md#router](docs/rtx4090/baseline.md#router) |
| 编译缓存三层持久化 | 已验证 | [docs/rtx4090/baseline.md#编译缓存](docs/rtx4090/baseline.md#编译缓存) |
| FP8 block kernel 调优；SGLang 对比；TP1 vs TP2（两卡） | 已验证（2026-10-31） | [docs/rtx4090/engine-tp.md](docs/rtx4090/engine-tp.md) |
| 单卡 TP1 调参、vLLM 0.31、HyperQwen 量化、视觉与 MTP、生产替换 | 已验证（2026-10-31） | [docs/rtx4090/tp1-hyperqwen.md](docs/rtx4090/tp1-hyperqwen.md) |
| 单卡 profile：single / batch 模式按场景选择（4090 与 3090 同一套测试） | 已验证（2026-10-31） | [docs/profiles.md](docs/profiles.md) |
| DP4（4 个单卡副本）四卡实测、路由策略、长上下文质量 | 已验证（2026-11-01） | [docs/rtx4090/dp4.md](docs/rtx4090/dp4.md) |
| prefill 调度：max-num-batched-tokens 四档；与 RTX 3090 对比 | 已验证（2026-10-30） | [docs/rtx4090/prefill.md](docs/rtx4090/prefill.md) |
| PP 对照、HyperQwen 等其他引擎 | **未验证** | [docs/rtx4090/roadmap.md](docs/rtx4090/roadmap.md) |

## 代表性结果（定稿 vs 原部署）

- 峰值输出吞吐（并发 64，in1k/out256）：284 → 681 tok/s（2.4×）。主因是原 `max-num-seqs 6` 远低于 KV 容量。
- 真实文本单流 TPOT P50：30.6 → 17.4 ms（MTP2，接受长度约 2.17）。
- 200k 上下文经 Router：0.1.12 全部 413 → 0.1.15 可用。
- 编译缓存三层全部持久化后，单副本重建到就绪：317s（冷）→ 136–170s。

## 目录

| 路径 | 内容 |
| --- | --- |
| [scripts/](scripts/) | `deploy_llm.sh` 重建后端；`suite.sh`/`bench.sh` 压测套件（基于 `vllm bench serve`）；`experiments.sh` 批量参数实验；`finalize.sh`/`failsafe.sh` 定稿与超时回滚；`sampler.sh` GPU 与指标采样；`summarize.py` 汇总表格；`deploy_exp.sh`/`guard_single_backend.sh` 单副本实验容器与自动恢复；`exp_prefill.sh` prefill 场景；`prefill_probe.py` 纯 prefill 探针 |
| [experiments/](experiments/) | 参数实验列表 |
| [data/](data/) | 压测用合成提示词（非业务数据） |
| [results/](results/) | 原始结果 JSON、GPU/指标采样 |
| [docs/](docs/) | 实验记录与计划 |

## 复现

```bash
export MODEL_DIR=/path/to/models   # 其下有 Qwen/Qwen3.8-27B-FP8 等目录
scripts/deploy_llm.sh "--max-num-seqs 64 --enable-prefix-caching \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' --gpu-memory-utilization 0.78"
scripts/suite.sh my-run quick
scripts/summarize.py ~/loadtest/results/my-run
```

脚本假设 Router 在 `:8000`、两个后端在 `:8003/:8004`、embedding/reranker 在 `:8001/:8002`，按需修改。

## 许可

[Apache-2.0](LICENSE)。
