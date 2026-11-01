#!/usr/bin/env bash
# 11-01 切换：四卡逐张切到 single+MTP+PTH，MAX_SEQS 32、CG 128（E 公开评测 + F 槽位测试的结论）
# 顺序 g1 → g2 → g0 → g3，每张摘流 → 重建 → 视觉冒烟 → 接回；任何一张失败，swap 回退该卡 batch 并停止。
# 回滚单卡：source 本目录 c-rotate.sh 的函数后 `swap <gpu> batch`。
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-switch; mkdir -p "$OUT"; L=$OUT/run.log
D=$(dirname "$0")
source <(sed -n '/^declare -A PORT/p;/^run_c()/,/^}/p;/^ready()/,/^}/p;/^swap()/,/^}/p' "$D/c-rotate.sh")
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"

log "workers $(curl -s 127.0.0.1:8000/list_workers)"
for g in 1 2 0 3; do SEQS=32 swap $g cg128; done
log "workers $(curl -s 127.0.0.1:8000/list_workers)"

# 切换后 Router 层验收
sleep 20
for c in 64 128; do waitidle; vb post-chat-c$c 8000 $CHAT --max-concurrency $c --num-prompts $((c*6)); done
waitidle
python3 "$S/gsm_probe.py" http://127.0.0.1:8000 $M "$T/data/gsm200.jsonl" "$OUT/post-gsm.json" --conc 32 2>&1 | tail -1 | tee -a "$L"
summ | tee "$OUT/summary.md"
log "G done"
