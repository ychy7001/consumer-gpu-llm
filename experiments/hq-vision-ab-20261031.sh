#!/usr/bin/env bash
# 10-31 下午：GPU3 单卡 HyperQwen batch，补视觉（图像 token 上限、每请求 5 图）与 MTP 调优
# 先摘流停 b、起 guard；结束写 guard.restore 恢复 b
set -u
D=$HOME/loadtest/tool/scripts
V=$HOME/loadtest/vision
RESULT_DIR=$HOME/loadtest/results/20261031-vision
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
URL=http://127.0.0.1:8005; M=Qwen3.8-27B-FP8
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
ROOT=/models/hyperqwen
BATCH="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"
MM5='{"image":{"count":5}}'
px(){ echo "{\"size\":{\"shortest_edge\":65536,\"longest_edge\":$(( $1 * 1024 ))}}"; }   # 图像 token 上限 -> 像素上限（32x32 每 token）

sudo docker image inspect $IMAGE >/dev/null 2>&1 && [ -d "$ROOT/models" ] || { log "no image or model"; log "vision done"; exit 1; }
sudo mkdir -p "$ROOT/cache-exp"; [ -d "$ROOT/cache-exp/.triton" ] || sudo cp -a "$ROOT/cache-gpu3/." "$ROOT/cache-exp/"

curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8004" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do
  r=$(curl -s -m5 127.0.0.1:8004/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
  [[ ${r:-0} == 0 ]] && break; sleep 10
done
log "b running=${r:-0} after $((i*10))s, stopping"
sudo docker stop -t 120 vllm-tp2-b >/dev/null
(cd "$D" && WAIT_MIN=16 WAIT_LIMIT=20 setsid nohup bash guard_single_backend.sh "$(date -d '+160 minutes' '+%F %H:%M')" 8003 vllm-tp2-b 8004 \
  "vllm-exp vllm-tune-0 vllm-tune-1" >/dev/null 2>&1 </dev/null &)
sleep 2

arm(){ # 名字 套餐(full|vis|text) EXTRA_ARGS 环境变量...
  local name=$1 suite=$2 xa=$3; shift 3
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; return 1; }
  [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-tp2-b 2>/dev/null) == true ]] && { log "b restored by guard, abort"; return 1; }
  mkdir -p "$RESULT_DIR/$name"
  sudo docker rm -f vllm-exp >/dev/null 2>&1
  local envs=(); for e in "$@"; do envs+=(-e "$e"); done
  log "== $name [$suite]: $* EXTRA_ARGS=$xa"
  sudo docker run -d --name vllm-exp --network host --ipc host --gpus '"device=3"' \
    -v "$ROOT/models":/app/models -v "$ROOT/cache-exp":/cache \
    -e HOME=/cache -e PREPARE=0 -e VERIFY=0 -e PORT=8005 -e MAX_LEN=262144 -e GPU_UTIL=0.92 -e PREFIX_CACHE=1 \
    -e KV=fp8 -e INT8_ACT=int8 -e EXTRA_ARGS="--served-model-name $M $xa" "${envs[@]}" $IMAGE batch >/dev/null
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
  sudo docker logs vllm-exp 2>&1 | grep -E "GPU KV cache size" | tail -1 | cut -c1-200 | tee -a "$L"
  sudo docker logs vllm-exp 2>&1 | grep -m1 "non-default args" | grep -oE "'(limit_mm_per_prompt|mm_processor_kwargs|speculative_config|language_model_only)': [^}]*\}?\}?" | tee -a "$L"
  python3 "$D/prefill_probe.py" $URL $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  if [[ $suite != vis ]]; then
    python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
    python3 "$D/decode_probe.py" $URL $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
      --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1,16" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  fi
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch 8005 llm $BATCH --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
  [[ $suite != vis ]] && RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch-c48 8005 llm $BATCH --max-concurrency 48 --num-prompts 288 >>"$L" 2>&1
  for b in batch batch-c48; do [ -f "$RESULT_DIR/$name/$b.json" ] && python3 - "$RESULT_DIR/$name/$b.json" $b <<'EOF' | tee -a "$L"
import json,sys
d=json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} total {d['total_token_throughput']:.0f} tok/s, ttft p50/p99 {d['median_ttft_ms']:.0f}/{d['p99_ttft_ms']:.0f} ms, tpot p50/p99 {d['median_tpot_ms']:.0f}/{d['p99_tpot_ms']:.0f} ms, ok {d['completed']}")
EOF
  done
  [[ $suite != vis ]] && python3 "$D/gsm_probe.py" $URL $M "$HOME/loadtest/tool/data/gsm200.jsonl" "$RESULT_DIR/$name/gsm.json" --conc 32 2>&1 | tee -a "$L"
  [[ $suite != text ]] && ocr "$name"
}
ocr(){ python3 "$V/ocr_probe.py" $URL $M "$V/ocrset" "$RESULT_DIR/$1/ocr.json" --conc 4 --manifest manifest12.jsonl 2>&1 | tail -1 | tee -a "$L"; }

# 对照：线上同款 vLLM 0.25.1 FP8（单卡），只测 OCR 与多图，作为视觉质量基线（不压线上 a）
mkdir -p "$RESULT_DIR/base-fp8"
log "== base-fp8: vLLM 0.25.1 FP8 TP1 (vision baseline)"
if IMAGE=vllm/vllm-openai:v0.25.1 TP=1 GPUS=3 MODEL_DIR=/models FAIL_LOG=$RESULT_DIR/base-fp8/deploy-fail.log \
    BASE="--max-num-seqs 32 --enable-prefix-caching --compilation-config {\"max_cudagraph_capture_size\":32} --gpu-memory-utilization 0.92" \
    bash "$D/deploy_exp.sh" "--max-model-len 262144" >>"$L" 2>&1; then
  ocr base-fp8
else
  log "deploy base-fp8 failed"
fi
sudo docker rm -f vllm-exp >/dev/null 2>&1

arm v2k    vis  "--limit-mm-per-prompt $MM5"                                  VISION=1   # 启动脚本默认 2048 图像 token
arm v4k    full "--limit-mm-per-prompt $MM5 --mm-processor-kwargs $(px 4096)" VISION=1   # 同时作 MTP 对照
arm v8k    vis  "--limit-mm-per-prompt $MM5 --mm-processor-kwargs $(px 8192)" VISION=1
arm m1     full "--limit-mm-per-prompt $MM5 --mm-processor-kwargs $(px 4096) --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}" VISION=1
arm m2     full "--limit-mm-per-prompt $MM5 --mm-processor-kwargs $(px 4096) --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" VISION=1
sudo docker rm -f vllm-exp >/dev/null 2>&1
log "vision done"
touch "$HOME/loadtest/guard.restore"
