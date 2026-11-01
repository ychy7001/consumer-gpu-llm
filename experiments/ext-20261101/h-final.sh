#!/usr/bin/env bash
# 11-01 扩展 H：最终配置（四卡 single+MTP+PTH，32 槽 CG128）复测
#   H1 经 Router 的容量曲线，与 A（3 batch + 1 single）同负载，直接对比
#   H2 GPU1 单卡：长 prefill 切块（mbt 8192 + long-prefill-token-threshold 2048）对混合负载的效果，
#      D 在 batch 模式下测过（chat P99 54 s → 12 s），这里在 single 模式复核；结束后 GPU1 回到默认
# 只压测、不改其余三卡；每组前等无真实流量
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-final; mkdir -p "$OUT"; L=$OUT/run.log
D=$(dirname "$0")
source <(sed -n '/^declare -A PORT/p;/^run_c()/,/^}/p;/^ready()/,/^}/p;/^swap()/,/^}/p' "$D/c-rotate.sh")
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
DOC="--dataset-name random --ignore-eos --random-range-ratio 0.2 --random-input-len 16384 --random-output-len 512"
DOC64="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 65536 --random-output-len 256"
client
log "workers $(curl -s 127.0.0.1:8000/list_workers)"

# H1
for c in 16 32 64 96 128 192; do waitidle; vb chat-c$c 8000 $CHAT --max-concurrency $c --num-prompts $((c*6)); log "chat-c$c done"; done
for c in 4 8 16 32 64; do waitidle; vb doc-c$c 8000 $DOC --max-concurrency $c --num-prompts $((c*3)); log "doc-c$c done"; done
for c in 32 64 128; do
  waitidle; d=$((c/4))
  vb mix$c-chat 8000 $CHAT --max-concurrency $((c-d)) --num-prompts $(((c-d)*6)) &
  vb mix$c-doc 8000 $DOC --max-concurrency $d --num-prompts $((d*3)) &
  wait; log "mix-c$c done"
done
waitidle
python3 "$S/prefill_probe.py" http://127.0.0.1:8000 $M "$OUT/longdoc-prefill.jsonl" --lens 131072,204800 --conc 1,4,8 --rounds 1 >>"$L" 2>&1
log "longdoc prefill done"
waitidle
python3 "$S/gsm_probe.py" http://127.0.0.1:8000 $M "$T/data/gsm200.jsonl" "$OUT/gsm.json" --conc 32 2>&1 | tail -1 | tee -a "$L"
log "H1 done"

# H2：GPU1 直连 :8003，默认 vs 长 prefill 切块
h2(){ # 名字
  local n=$1
  curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
  sleep 20; waitidle
  python3 "$S/prefill_probe.py" http://127.0.0.1:8003 $M "$OUT/$n-prefill.jsonl" --lens 16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  waitidle
  vb $n-mixdoc 8003 $DOC64 --max-concurrency 2 --num-prompts 12 &
  sleep 5
  vb $n-mixchat 8003 $CHAT --max-concurrency 24 --num-prompts 480
  wait
  waitidle; vb $n-chat-c24 8003 $CHAT --max-concurrency 24 --num-prompts 144
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
  log "$n done"
}
h2 lp-off
XARGS="--max-num-batched-tokens 8192 --long-prefill-token-threshold 2048" SEQS=32 swap 1 cg128
sudo docker logs vllm-hq-g1 2>&1 | grep -oE "max_num_batched_tokens=[0-9]+|long_prefill_token_threshold=[0-9]+" | sort -u | tr "\n" " " | tee -a "$L"; echo >> "$L"
h2 lp-on
SEQS=32 swap 1 cg128
summ | tee "$OUT/summary.md"
log "H done"
