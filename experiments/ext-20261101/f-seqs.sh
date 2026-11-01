#!/usr/bin/env bash
# 11-01 扩展 F：single CG 模式槽位数（MAX_SEQS 16 vs 32），GPU1 单卡直连对比
#   s16cg64（现候选）/ s32cg64 / s32cg128（CG 覆盖 32 槽 × 4 token）
# 每档：GPU1 摘出 Router → 重建 → 视觉冒烟 → 直连 :8003 压测 → 接回 Router。
# 结束后 GPU1 留在 s16cg64（与 GPU0 一致），按结果再决定全切用哪档。
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-seqs; mkdir -p "$OUT"; L=$OUT/run.log
D=$(dirname "$0")
source <(sed -n '/^declare -A PORT/p;/^run_c()/,/^}/p;/^ready()/,/^}/p;/^swap()/,/^}/p' "$D/c-rotate.sh")
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
DOC="--dataset-name random --ignore-eos --random-range-ratio 0.2 --random-input-len 16384 --random-output-len 512"

for cfg in s16cg64 s32cg64 s32cg128; do
  s=${cfg#s}; s=${s%cg*}; cg=${cfg#*cg}
  SEQS=$s swap 1 cg$cg   # 失败时 swap 自己回退 batch 并 exit 1
  curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
  sleep 20
  for c in 8 16 24 32 48; do waitidle; vb $cfg-chat-c$c 8003 $CHAT --max-concurrency $c --num-prompts $((c*6)); done
  for c in 4 8 16; do waitidle; vb $cfg-doc-c$c 8003 $DOC --max-concurrency $c --num-prompts $((c*3)); done
  waitidle
  python3 "$S/decode_probe.py" http://127.0.0.1:8003 $M "$OUT/$cfg-decode.jsonl" --prompts "$T/data/real_prompts.jsonl" \
    --corpus-dir "$T/corpus" --grid "0:1,16,32" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
  log "$cfg done"
done
SEQS=16 swap 1 cg64
summ | tee "$OUT/summary.md"
log "F done"
