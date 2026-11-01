#!/usr/bin/env bash
# 11-01 扩展 C：整机 profile 对比，逐卡轮换（每次只摘一张卡，Router 始终 ≥3 个后端）
#   阶段 1 四卡全 batch → 测；阶段 2 四卡全 single+MTP+PTH CG64 16 槽 → 测；
#   阶段 3 恢复：GPU0/1/2 batch，GPU3 single+MTP+PTH CG32（与实验前一致）
# 任一卡起不来或视觉冒烟不过：该卡按 batch 重建后退出
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-rotate; mkdir -p "$OUT"; L=$OUT/run.log
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
DOC="--dataset-name random --ignore-eos --random-range-ratio 0.2 --random-input-len 16384 --random-output-len 512"
declare -A PORT=([0]=8006 [1]=8003 [2]=8005 [3]=8004) UTIL=([0]=0.80 [1]=0.92 [2]=0.82 [3]=0.92) NAME=([0]=vllm-hq-g0 [1]=vllm-hq-g1 [2]=vllm-hq-g2 [3]=vllm-hq-g3)

run_c(){ # GPU 模式(batch|cg32|cg64)
  local g=$1 mode=$2 p=${PORT[$1]}
  if [ "$mode" = batch ]; then
    sudo docker run -d --name ${NAME[$g]} --restart always --ipc host --gpus "\"device=$g\"" -p "127.0.0.1:$p:8000" \
      -v "$ROOT/models":/app/models -v "$ROOT/cache-gpu$g":/cache \
      -e HOME=/cache -e HOST=0.0.0.0 -e PORT=8000 -e PREPARE=0 -e VERIFY=0 \
      -e KV=fp8 -e INT8_ACT=int8 -e MAX_LEN=262144 -e GPU_UTIL=${UTIL[$g]} -e PREFIX_CACHE=1 \
      -e VISION=1 -e EXTRA_ARGS="--served-model-name $M $VIS_ARGS ${XARGS:-}" $IMAGE batch >/dev/null
  else
    local cg=${mode#cg}
    [ -d "$ROOT/cache-gpu${g}s" ] || sudo cp -a "$ROOT/cache-exp" "$ROOT/cache-gpu${g}s"
    sudo docker run -d --name ${NAME[$g]} --restart always --ipc host --gpus "\"device=$g\"" -p "127.0.0.1:$p:8000" \
      -v "$ROOT/models":/app/models -v "$ROOT/cache-gpu${g}s":/cache \
      -e HOME=/cache -e HOST=0.0.0.0 -e PORT=8000 -e PREPARE=0 -e VERIFY=0 -e PREFIX_CACHE=1 -e VISION=1 \
      -e SPEC=mtp -e CTX=long -e PTH_FA=1 -e DRAFT_TOKENS=3 -e MTP_DRAFT_VOCAB=0 \
      -e MAX_LEN=262144 -e MAX_SEQS=${SEQS:-16} -e CG=$cg -e INT8_ACT=int8 -e PREFILL_ATTN=int8 \
      -e GPU_UTIL=${UTIL[$g]} -e KV_MEM= \
      -e EXTRA_ARGS="--served-model-name $M $VIS_ARGS ${XARGS:-}" $IMAGE single >/dev/null
  fi
}
ready(){ # 端口 → 0 就绪且过视觉冒烟
  local p=$1 i n
  for i in $(seq 1 72); do curl -sf -m3 127.0.0.1:$p/v1/models >/dev/null && break; sleep 10; done
  curl -s -m3 127.0.0.1:$p/v1/models | grep -q "\"$M\"" || return 1
  for n in 64 2000; do
    curl -s -m120 127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$(printf '测试%.0s' $(seq 1 $n))，回复好\"}],\"max_tokens\":16}" >/dev/null
  done
  python3 "$HOME/loadtest/vision/ocr_probe.py" http://127.0.0.1:$p $M "$HOME/loadtest/vision/ocrset" "$OUT/ocr-$p-$(date +%H%M).json" \
    --conc 2 --manifest manifest-smoke.jsonl 2>&1 | tail -1 | tee -a "$L" | grep -q "ok 2/2 .*multi5 ok"
}
swap(){ # GPU 模式
  local g=$1 mode=$2 p=${PORT[$1]} i r
  curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:$p" >> "$L"; echo >> "$L"
  for i in $(seq 1 90); do
    r=$(curl -s -m5 127.0.0.1:$p/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
    [[ ${r:-0} == 0 ]] && break; sleep 10
  done
  sudo docker rm -f ${NAME[$g]} >/dev/null 2>&1
  run_c $g $mode
  if ! ready $p; then
    log "GPU$g $mode failed, falling back to batch"
    sudo docker logs ${NAME[$g]} 2>&1 | grep -iE "error|out of memory" | tail -5 | cut -c1-200 >> "$L"
    sudo docker rm -f ${NAME[$g]} >/dev/null 2>&1; run_c $g batch
    ready $p && curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:$p" >> "$L"
    exit 1
  fi
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:$p" >> "$L"; echo >> "$L"
  log "GPU$g -> $mode on :$p, $(sudo docker logs ${NAME[$g]} 2>&1 | grep -o 'GPU KV cache size: [0-9,]* tokens' | tail -1)"
}
suite(){ # 阶段名
  sleep 20
  waitidle; vb $1-chat-c64 8000 $CHAT --max-concurrency 64 --num-prompts 384
  waitidle; vb $1-chat-c128 8000 $CHAT --max-concurrency 128 --num-prompts 768
  waitidle; vb $1-doc-c32 8000 $DOC --max-concurrency 32 --num-prompts 96
  waitidle
  python3 "$S/decode_probe.py" http://127.0.0.1:8000 $M "$OUT/$1-decode.jsonl" --prompts "$T/data/real_prompts.jsonl" \
    --corpus-dir "$T/corpus" --grid "0:1,16,64;32768:4" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  waitidle
  python3 "$S/gsm_probe.py" http://127.0.0.1:8000 $M "$T/data/gsm200.jsonl" "$OUT/$1-gsm.json" --conc 32 2>&1 | tail -1 | tee -a "$L"
  log "$1 suite done"
}

client
# g3 灰度原容器保留为回滚目标（rollback-g3-20261101.sh 依赖 vllm-hq-g3-batch，不动）
log "phase 1: all batch"
waitidle; swap 3 batch
suite allbatch
log "phase 2: all single+MTP+PTH CG64"
for g in 3 1 2 0; do waitidle; swap $g cg64; done
suite allsingle64
log "phase 3: restore (GPU0/1/2 batch, GPU3 cg32)"
for g in 0 1 2; do waitidle; swap $g batch; done
waitidle; swap 3 cg32
log "workers $(curl -s 127.0.0.1:8000/list_workers)"
summ | tee -a "$L"
log "C done"
