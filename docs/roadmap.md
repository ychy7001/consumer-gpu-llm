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
| prefill 调度 | `max-num-batched-tokens`、chunked prefill 对长输入 TTFT 与并发 decode 长尾的取舍 | rag/summ/ctx64k 的 TTFT 与 TPOT P99 |
| MTP 步数 | MTP3 是否在真实文本上继续收益 | real-c1/c16 TPOT、接受长度 |
| PP2 vs TP2 | 两张 PCIe 卡上 PP 的流水线气泡是否抵得过 TP 的 all-reduce 开销 | 单流 TPOT、c16/c64 吞吐 |
| 其他推理引擎 | 同模型同参数下与原版 vLLM 对比（候选：HyperQwen） | quick 套件全场景 |

## 晚上（四张卡）

| 实验 | 问题 | 判据 |
| --- | --- | --- |
| DP4 vs TP2×2 | 4 个单卡副本吞吐是否更高；单卡 KV 能否支撑 262k 上下文 | sweep-c64 吞吐、ctx200k 是否可调度、单卡 KV 池大小 |
| TP4 单副本 | 长上下文 prefill 是否明显变快 | ctx128k/ctx200k TTFT |
| Router 恢复时间 | 调小 `--cb-timeout-duration-secs` 能否缩短约 104s 的重新接流 | failover 时间线 |
| 就绪后预热 | 先预热再接流能否消除首批 13–16s | 重启后首批请求延迟 |
