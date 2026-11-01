#!/usr/bin/env bash
# 11-01 扩展 A：整机容量曲线（经生产 Router :8000）、混部干扰、长文档并发
# 只压测不改部署；每组前等无真实流量
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-capacity; mkdir -p "$OUT"; L=$OUT/run.log
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
DOC="--dataset-name random --ignore-eos --random-range-ratio 0.2 --random-input-len 16384 --random-output-len 512"
client
log "workers $(curl -s 127.0.0.1:8000/list_workers)"

# 1. 容量曲线：短对话 / 16k 文档 / 混合（对话与文档 3:1 并发同时跑）
for c in 16 32 64 96 128 192; do waitidle; vb chat-c$c 8000 $CHAT --max-concurrency $c --num-prompts $((c*6)); log "chat-c$c done"; done
for c in 4 8 16 32 64; do waitidle; vb doc-c$c 8000 $DOC --max-concurrency $c --num-prompts $((c*3)); log "doc-c$c done"; done
for c in 32 64 96 128; do
  waitidle; d=$((c/4))
  vb mix$c-chat 8000 $CHAT --max-concurrency $((c-d)) --num-prompts $(((c-d)*6)) &
  vb mix$c-doc 8000 $DOC --max-concurrency $d --num-prompts $((d*3)) &
  wait; log "mix-c$c done"
done

# 6. 混部干扰：embedding / reranker 在空闲与 LLM 满载（chat c96）时的延迟
B="env RESULT_DIR=$OUT MODEL_DIR=/models bash $S/bench.sh"
EMB="--dataset-name random --random-input-len 512 --random-output-len 1 --max-concurrency 8 --num-prompts 400"
RR="--dataset-name random-rerank --random-input-len 512 --max-concurrency 8 --num-prompts 200"
waitidle
$B emb-idle 8001 emb $EMB >>"$L" 2>&1
$B rr-idle 8002 rerank $RR >>"$L" 2>&1
waitidle
vb bg-chat-c96 8000 $CHAT --max-concurrency 96 --num-prompts 1536 &
sleep 60
$B emb-busy 8001 emb $EMB >>"$L" 2>&1
$B rr-busy 8002 rerank $RR >>"$L" 2>&1
wait; log "colocation done"

# 2. 长文档并发：128k / 200k 纯 prefill（TTFT），再测 128k 上下文下的 decode
waitidle
python3 "$S/prefill_probe.py" http://127.0.0.1:8000 $M "$OUT/longdoc-prefill.jsonl" --lens 131072,204800 --conc 1,4,8,12 --rounds 1 >>"$L" 2>&1
log "longdoc prefill done"
waitidle
python3 "$S/decode_probe.py" http://127.0.0.1:8000 $M "$OUT/longdoc-decode.jsonl" --prompts "$T/data/real_prompts.jsonl" \
  --corpus-dir "$T/corpus" --grid "131072:1,4,8" --max-tokens 256 --rounds 1 >>"$L" 2>&1
log "longdoc decode done"

summ | tee -a "$L"
log "A done"
