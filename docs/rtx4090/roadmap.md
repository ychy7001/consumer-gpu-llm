# 后续实验

全部**未验证**。完成一项就单独写一篇记录（`docs/<主题>.md`）并更新 README 状态表。

## 约束

机器在生产服务中：

- 白天只停一个后端（两张卡）做实验，另一个后端继续经 Router 服务。压测直连被停后端的端口，不走 Router。另一个后端在服务，CPU/PCIe 共享，白天数据只用于横向比较参数，结论以晚上复测为准。
- 需要四张卡或改 Router 的实验在 18:00 以后做。改 Router 的 systemd 配置先 dry-run，脚本带白名单、备份和自动恢复。
- 每轮实验结束恢复原部署，给 Router 留约 2 分钟重新接流。

## 白天（两张卡）

| 实验 | 问题 | 判据 |
| --- | --- | --- |
| ~~prefill 调度~~ | 已完成，见 [prefill.md](prefill.md) | |
| 长短请求隔离 | `--max-num-partial-prefills`/`--long-prefill-token-threshold` 能否在 mbt 8192 下保住短请求 TPOT P50 | mix-long TTFT、mix-chat TPOT P50 |
| ~~MTP 步数~~ | 批处理并发 24 下 MTP3 不占优，见 [engine-tp.md](engine-tp.md)；低并发真实文本未测 | |
| PP2 vs TP2 | 两张 PCIe 卡上 PP 的流水线气泡是否抵得过 TP 的 all-reduce 开销 | 单流 TPOT、c16/c64 吞吐 |
| 其他推理引擎 | SGLang 已测，见 [engine-tp.md](engine-tp.md)；HyperQwen（W4A16 + KVarN）待测 | quick 套件全场景 |

## 晚上（四张卡）

| 实验 | 问题 | 判据 |
| --- | --- | --- |
| DP4 vs TP2×2 | 两卡已验证 2×TP1 吞吐 +63%、262k 可启动（[engine-tp.md](engine-tp.md)）；待四卡含 embedding/reranker 混部实测 | 批处理吞吐、ctx200k 是否可调度、Router 4 worker |
| TP4 单副本 | 长上下文 prefill 是否明显变快 | ctx128k/ctx200k TTFT |
| Router 恢复时间 | 调小 `--cb-timeout-duration-secs` 能否缩短约 104s 的重新接流 | failover 时间线 |
| 就绪后预热 | 先预热再接流能否消除首批 13–16s | 重启后首批请求延迟 |
