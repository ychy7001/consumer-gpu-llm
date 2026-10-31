#!/usr/bin/env bash
# 10-31 晚：RTX 4090 单卡 HyperQwen single / batch 模式完整数据（GPU3，端口 8006）
# 摘流停 hq-g3（8004），guard 盯 a(8003)+hq-g2(8005)；结束写 guard.restore 恢复 hq-g3
set -u
T=$HOME/loadtest/tool
S=$T/scripts; DATA=$T/data; CORPUS=$T/corpus; OCRSET=$HOME/loadtest/vision/ocrset
OUT=$HOME/loadtest/results/20261031-cards/rtx4090
L=$OUT/run.log; mkdir -p "$OUT"
PORT=8006; URL=http://127.0.0.1:$PORT; M=Qwen3.8-27B-FP8
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
ROOT=/models/hyperqwen
source "$T/cards/cardsuite.sh"
bench_run(){ RESULT_DIR=$OUT/$1 MODEL_DIR=/models bash "$S/bench.sh" batch-c$2 $PORT llm $BATCH_ARGS \
  --max-concurrency $2 --num-prompts $3 >>"$L" 2>&1; }

[ -d "$ROOT/models/Qwen3.8-27B-W4A16-AutoRound-fast" ] && [ -d "$ROOT/models/Qwen3.8-27B-DFlash2-W4A16" ] || { log "fast/dflash2 missing"; exit 1; }
sudo mkdir -p "$ROOT/cache-exp"

curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8004" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do
  r=$(curl -s -m5 127.0.0.1:8004/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
  [[ ${r:-0} == 0 ]] && break; sleep 10
done
log "hq-g3 running=${r:-0} after $((i*10))s, stopping"
sudo docker update --restart=no vllm-hq-g3 >/dev/null
sudo docker stop -t 120 vllm-hq-g3 >/dev/null
(cd "$S" && WAIT_MIN=24 WAIT_LIMIT=20 setsid nohup bash guard_single_backend.sh "$(date -d '+200 minutes' '+%F %H:%M')" 8003,8005 vllm-hq-g3 8004 \
  "vllm-exp" >/dev/null 2>&1 </dev/null &)
sleep 2

arm(){ # 名字 模式(single|batch) EXTRA_ARGS 环境变量...
  local name=$1 mode=$2 xa=$3; shift 3
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; return 1; }
  [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-hq-g3 2>/dev/null) == true ]] && { log "hq-g3 restored by guard, abort"; return 1; }
  mkdir -p "$OUT/$name"
  sudo docker rm -f vllm-exp >/dev/null 2>&1
  local envs=(); for e in "$@"; do envs+=(-e "$e"); done
  log "== $name [$mode]: $* EXTRA_ARGS=$xa"
  sudo docker run -d --name vllm-exp --network host --ipc host --gpus '"device=3"' \
    -v "$ROOT/models":/app/models -v "$ROOT/cache-exp":/cache \
    -e HOME=/cache -e PREPARE=0 -e VERIFY=0 -e HOST=127.0.0.1 -e PORT=$PORT -e PREFIX_CACHE=1 -e VISION=1 \
    -e EXTRA_ARGS="--served-model-name $M $VIS_ARGS $xa" "${envs[@]}" $IMAGE $mode >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m3 $URL/v1/models >/dev/null && break
    if [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-exp 2>/dev/null) != true ]]; then
      sudo docker logs vllm-exp > "$OUT/$name/deploy-fail.log" 2>&1
      log "$name exited"; grep -E "Error|OutOfMemory|exceed|refus" "$OUT/$name/deploy-fail.log" | grep -v Traceback | tail -5 | tee -a "$L"; return 1
    fi
    sleep 10
  done
  curl -sf -m3 $URL/v1/models >/dev/null || { sudo docker logs vllm-exp > "$OUT/$name/deploy-fail.log" 2>&1; log "$name timeout"; return 1; }
  log "ready after $((i*10))s"
  sudo docker logs vllm-exp 2>&1 | grep -E "GPU KV cache size" | tail -1 | grep -oE "[0-9,]+ tokens" > "$OUT/$name/kv.txt"
  sudo docker logs vllm-exp 2>&1 | grep -m1 "non-default args" | grep -oE "'(model|max_model_len|max_num_seqs|speculative_config|kv_cache_dtype|kv_cache_memory_bytes|gpu_memory_utilization)': [^,}]*(\}[^,}]*\})?" | tee -a "$L"
  suite "$name"
  sudo docker logs vllm-exp 2>&1 | grep -E "SpecDecoding metrics" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+" | tee -a "$L"
}

# single 模式：fast 变体自动选中；KV_MEM 置空按 GPU_UTIL 分配（默认值按 24G 卡固定字节数）
arm s-mtp     single "" SPEC=mtp CTX=long INT8_ACT=int8 PREFILL_ATTN=int8 MTP_DRAFT_VOCAB=0 MAX_LEN=262144 GPU_UTIL=0.92 KV_MEM=
arm s-mtp-16  single "" SPEC=mtp CTX=long INT8_ACT=int8 PREFILL_ATTN=int8 MTP_DRAFT_VOCAB=0 MAX_LEN=262144 GPU_UTIL=0.92 KV_MEM= MAX_SEQS=16
arm s-off     single "" SPEC=off CTX=long INT8_ACT=int8 PREFILL_ATTN=int8 MAX_LEN=262144 GPU_UTIL=0.92 KV_MEM=
arm s-dflash2 single "" SPEC=dflash2 CTX=long INT8_ACT=int8 PREFILL_ATTN=int8 GPU_UTIL=0.92 KV_MEM=
# batch 模式：线上同款，同一套件复测
arm b-prod    batch  "" KV=fp8 INT8_ACT=int8 MAX_LEN=262144 GPU_UTIL=0.92
sudo docker rm -f vllm-exp >/dev/null 2>&1
log "cards 4090 done"
touch "$HOME/loadtest/guard.restore"
