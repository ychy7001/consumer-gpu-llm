# RTX 4090 48G LLM Serving

在 4×RTX 4090 48G（PCIe，无 NVLink）上用 vLLM 部署 Qwen3.8-27B-FP8 的调优记录：部署脚本、压测套件、原始结果与结论。所有结论附原始数据，并明确区分「已验证」与「未验证」。

> 本项目不是 NVIDIA、Qwen 或 vLLM 官方发布。数据来自一台生产混部机器（同机还有其他业务），主机名与路径已替换。

叙述版本写在知乎专栏（筹备中），按基线调优、并行方式对比、长上下文等分篇；仓库这边是原始数据和复现入口。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | 4× RTX 4090 48G，PCIe 互联 |
| 模型 | Qwen3.8-27B-FP8，`max-model-len 262144`，`kv-cache-dtype fp8` |
| 推理 | vLLM 0.25.1（`vllm/vllm-openai:v0.25.1`） |
| 拓扑 | 2 个 TP2 副本（GPU 0,1 / 2,3）+ [vllm-router](https://github.com/vllm-project/router) 0.1.15，`power_of_two` |
| 同卡混部 | Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B |

## 状态

| 主题 | 状态 | 入口 |
| --- | --- | --- |
| 基线调优：max-num-seqs、prefix caching、MTP、gmu | 已验证（2026-09-29～30） | [docs/baseline.md](docs/baseline.md) |
| Router：策略、故障摘除、请求体上限 | 已验证 | [docs/baseline.md#router](docs/baseline.md#router) |
| 编译缓存三层持久化 | 已验证 | [docs/baseline.md#编译缓存](docs/baseline.md#编译缓存) |
| DP4（4 个单卡副本）vs TP2×2 | **未验证** | [docs/roadmap.md](docs/roadmap.md) |
| prefill 调度：max-num-batched-tokens 四档；与 RTX 3090 对比 | 已验证（2026-10-30） | [docs/prefill.md](docs/prefill.md) |
| PP 对照、其他推理引擎同条件对比 | **未验证** | [docs/roadmap.md](docs/roadmap.md) |

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
