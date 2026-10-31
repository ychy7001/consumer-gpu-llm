#!/usr/bin/env bash
# 10-31 晚：RTX 3090 单卡 HyperQwen single / batch 模式完整数据，与 card-4090.sh 同一套件、同一镜像
# 用本机测试实例 hq-pc 的卡（GPU0，端口 18020），结束恢复 hq-pc 原配置
set -u
B=$HOME/llmbench
S=$B/cards; DATA=$B/cards/data; CORPUS=$B/corpus; OCRSET=$HOME/ocrset
OUT=$B/results/20261031-cards/rtx3090
L=$OUT/run.log; mkdir -p "$OUT"
PORT=18020; URL=http://127.0.0.1:$PORT; M=qwen3.8-27b
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
MROOT=/models/hyperqwen
source "$S/cardsuite.sh"
bench_run(){ # 与 4090 相同的 vllm bench serve 参数，在同一镜像里跑客户端
  local d=$OUT/$1; mkdir -p "$d"
  podman run --rm --network host --entrypoint /app/venv/bin/vllm -v "$MROOT":/m:ro -v "$d":/r $IMAGE bench serve \
    --backend vllm --endpoint /v1/completions --base-url $URL --model $M --tokenizer /m/Qwen3.8-27B-W4A16-AutoRound \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /r --result-filename batch-c$2.json \
    $BATCH_ARGS --max-concurrency $2 --num-prompts $3 > "$d/batch-c$2.log" 2>&1 || log "bench $1 c$2 failed"; }

run_hq(){ # 模式 -e...
  local mode=$1; shift
  podman rm -f hq-pc >/dev/null 2>&1
  podman run -d --name hq-pc --device nvidia.com/gpu=0 --network host --ipc host \
    -v hq-cache:/cache -v "$MROOT":/app/models -e VISION=1 -e PREFIX_CACHE=1 "$@" $IMAGE $mode >/dev/null
  for i in $(seq 1 180); do
    curl -sf $URL/health >/dev/null && { log "ready after $((i*5))s"; return 0; }
    podman ps --format '{{.Names}}' | grep -qx hq-pc || { log "container exited"; return 1; }
    sleep 5
  done
  log "health timeout"; return 1
}

arm(){ # 名字 模式 EXTRA_ARGS -e...
  local name=$1 mode=$2 xa=$3; shift 3
  [ -f "$OUT/$name/ocr.json" ] && { log "skip $name (done)"; return 0; }
  rm -rf "$OUT/$name"; mkdir -p "$OUT/$name"
  log "== $name [$mode]: $* EXTRA_ARGS=$xa"
  run_hq $mode -e EXTRA_ARGS="$VIS_ARGS $xa" "$@" || { podman logs hq-pc > "$OUT/$name/deploy-fail.log" 2>&1
    grep -E "Error|OutOfMemory|exceed|refus" "$OUT/$name/deploy-fail.log" | grep -v Traceback | tail -5 | tee -a "$L"; return 1; }
  podman logs hq-pc 2>&1 | grep -E "GPU KV cache size" | tail -1 | grep -oE "[0-9,]+ tokens" > "$OUT/$name/kv.txt"
  podman logs hq-pc 2>&1 | grep -m1 "non-default args" | grep -oE "'(model|max_model_len|max_num_seqs|speculative_config|kv_cache_dtype|kv_cache_memory_bytes|gpu_memory_utilization)': [^,}]*(\}[^,}]*\})?" | tee -a "$L"
  suite "$name"
  podman logs hq-pc 2>&1 | grep -E "SpecDecoding metrics" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+" | tee -a "$L"
}

# single 模式：与 10-30 推荐配置相同（150k）
arm s-mtp     single "" -e SPEC=mtp -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8 -e MTP_DRAFT_VOCAB=0
arm s-off     single "" -e SPEC=off -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8
arm s-dflash2 single "" -e SPEC=dflash2 -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8
# batch 模式：与 4090 线上同款参数，上下文按 150k
arm b-prod    batch  "--served-model-name $M" -e KV=fp8 -e INT8_ACT=int8 -e MAX_LEN=150000

log "== restore hq-pc (old image, original env)"
podman rm -f hq-pc >/dev/null 2>&1
podman run -d --name hq-pc --device nvidia.com/gpu=0 --network host --ipc host \
  -v hq-cache:/cache -v "$MROOT":/app/models -e MTP_DRAFT_VOCAB=0 -e VISION=1 -e PREFIX_CACHE=1 \
  -e SPEC=mtp -e CTX=long -e INT8_ACT=int8 -e PREFILL_ATTN=int8 ghcr.io/syv-ai/hyperqwen:latest single >/dev/null
for i in $(seq 1 180); do curl -sf $URL/health >/dev/null && { log "restored"; break; }; sleep 5; done
log "cards 3090 done"
