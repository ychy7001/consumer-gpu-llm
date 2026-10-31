#!/usr/bin/env bash
# 10-31 白天：GPU3 单卡 HyperQwen batch 模式（W4A16 权重），与 vLLM FP8 TP1 同一套负载 + GSM
# 先摘流停 b，结束写 guard.restore 恢复 b
set -u
D=$HOME/loadtest/tool/scripts
RESULT_DIR=$HOME/loadtest/results/20261031-hq
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
URL=http://127.0.0.1:8005; M=Qwen3.8-27B-FP8   # 用 served-model-name 覆盖成同名，探针与 bench 不用改
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
HQ=$HOME/loadtest/models/hyperqwen
BATCH="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"

sudo docker image inspect $IMAGE >/dev/null 2>&1 && [ -f "$HQ/.ok" ] || { log "no image or model, skip hq"; log "hq done"; exit 1; }
mkdir -p "$HOME/loadtest/hq-cache"

# 摘流、等在途清空（最多 15 分钟）、停 b、起 guard（同 tp1-tune）
curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8004" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do
  r=$(curl -s -m5 127.0.0.1:8004/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
  [[ ${r:-0} == 0 ]] && break; sleep 10
done
log "b running=${r:-0} after $((i*10))s, stopping"
sudo docker stop -t 120 vllm-tp2-b >/dev/null
(cd "$D" && WAIT_MIN=16 WAIT_LIMIT=20 setsid nohup bash guard_single_backend.sh "$(date -d '+150 minutes' '+%F %H:%M')" 8003 vllm-tp2-b 8004 \
  "vllm-exp vllm-tune-0 vllm-tune-1" >/dev/null 2>&1 </dev/null &)
sleep 2

arm(){ # 名字 环境变量...
  local name=$1; shift
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; return 1; }
  [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-tp2-b 2>/dev/null) == true ]] && { log "b restored by guard, abort"; return 1; }
  mkdir -p "$RESULT_DIR/$name"
  sudo docker rm -f vllm-exp >/dev/null 2>&1
  local envs=(); for e in "$@"; do envs+=(-e "$e"); done
  log "== $name: $*"
  sudo docker run -d --name vllm-exp --network host --ipc host --gpus '"device=3"' \
    -v "$HQ":/app/models -v "$HOME/loadtest/hq-cache":/cache \
    -e PREPARE=0 -e VERIFY=0 -e PORT=8005 -e MAX_LEN=262144 -e GPU_UTIL=0.92 -e PREFIX_CACHE=1 \
    -e EXTRA_ARGS="--served-model-name $M" "${envs[@]}" $IMAGE batch >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m3 $URL/v1/models >/dev/null && break
    if [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-exp 2>/dev/null) != true ]]; then
      sudo docker logs vllm-exp > "$RESULT_DIR/$name/deploy-fail.log" 2>&1
      log "$name exited"; grep -E "Error|OutOfMemory|exceed" "$RESULT_DIR/$name/deploy-fail.log" | grep -v Traceback | tail -5 | tee -a "$L"; return 1
    fi
    sleep 10
  done
  curl -sf -m3 $URL/v1/models >/dev/null || { sudo docker logs vllm-exp > "$RESULT_DIR/$name/deploy-fail.log" 2>&1; log "$name timeout"; return 1; }
  log "ready after $((i*10))s"
  sudo docker logs vllm-exp 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|non-default args" | cut -c1-300 | tail -3 | tee -a "$L"
  python3 "$D/prefill_probe.py" $URL $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  python3 "$D/decode_probe.py" $URL $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
    --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1,16" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch 8005 llm $BATCH --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch-c48 8005 llm $BATCH --max-concurrency 48 --num-prompts 288 >>"$L" 2>&1
  for b in batch batch-c48; do python3 - "$RESULT_DIR/$name/$b.json" $b <<'EOF' | tee -a "$L"
import json,sys
d=json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} total {d['total_token_throughput']:.0f} tok/s, ttft p50/p99 {d['median_ttft_ms']:.0f}/{d['p99_ttft_ms']:.0f} ms, tpot p50/p99 {d['median_tpot_ms']:.0f}/{d['p99_tpot_ms']:.0f} ms, ok {d['completed']}")
EOF
  done
  python3 "$D/gsm_probe.py" $URL $M "$HOME/loadtest/tool/data/gsm200.jsonl" "$RESULT_DIR/$name/gsm.json" --conc 32 2>&1 | tee -a "$L"
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill-long.jsonl" --lens 126976 --conc 1 --rounds 1 >>"$L" 2>&1
}

arm h1-int8-fp8kv  INT8_ACT=int8 KV=fp8
arm h2-w4a16-fp8kv INT8_ACT=     KV=fp8
arm h3-int8-kvarn  INT8_ACT=int8 KV=kvarn
sudo docker rm -f vllm-exp >/dev/null 2>&1
log "hq done"
touch "$HOME/loadtest/guard.restore"
