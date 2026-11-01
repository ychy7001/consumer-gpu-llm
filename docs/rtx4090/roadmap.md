# 后续实验

10-29 到 11-01 的计划已全部完成或关闭，结论汇总在 [decision.md](decision.md)。

| 实验 | 状态 |
| --- | --- |
| prefill 调度 | 已完成，[prefill.md](prefill.md) |
| 长短请求隔离（`--long-prefill-token-threshold`） | 已完成，[final.md#长-prefill-切块](final.md#长-prefill-切块) |
| MTP 步数 | 已完成，[engine-tp.md](engine-tp.md)、[profiles.md](../profiles.md) |
| 其他推理引擎：SGLang、HyperQwen | 已完成，[engine-tp.md](engine-tp.md)、[tp1-hyperqwen.md](tp1-hyperqwen.md) |
| DP4 vs TP2×2 | 已完成，[dp4.md](dp4.md) |
| Router 恢复时间 | 已完成，[final.md#故障切换](final.md#故障切换) |
| PP2 vs TP2、TP4 单副本 | 关闭：PCIe 卡上 TP2 已输给单卡，理由见 [decision.md#未覆盖](decision.md#未覆盖) |
| 就绪后预热 | 关闭：切换流程先过视觉冒烟再接流，已包含预热 |

## 仍未回答

- 真实业务流量下 single 32 槽的表现（等内测反馈）。
- 负载感知路由：后端补 `/get_load` 或 Router 改按在途数均衡，解决长文档叠卡。
- 单个长文档 TTFT：128k 约 60 s，瓶颈在单卡 prefill 算力，需要 context parallel 或更快的 prefill kernel。
