#!/usr/bin/env bash
# 10-31 凌晨：GPU 2,3 上测 SGLang FP8（TP2 MTP / TP2 DFlash2 / TP1 MTP），负载与 kernel-ab 相同，外加 GSM8K 200 题
set -u
D=$HOME/loadtest/tool/scripts
RESULT_DIR=$HOME/loadtest/results/20261031-sglang
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
URL=http://127.0.0.1:8005; M=Qwen3.8-27B-FP8; NAME=vllm-exp   # 沿用 vllm-exp 名字，guard 恢复时会一并删掉
IMAGE=${SGL_IMAGE:-docker.m.daocloud.io/lmsysorg/sglang:v0.5.22}
MD=/models
DRAFT=$HOME/loadtest/models/Qwen3.8-27B-DFlash2
MTP="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
DFLASH="--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-num-draft-tokens 8"

gsm(){ python3 "$D/gsm_probe.py" $URL $M "$HOME/loadtest/tool/data/gsm200.jsonl" "$RESULT_DIR/$1/gsm.json" --conc 32 2>&1 | tee -a "$L"; }

deploy(){ # 名字 GPU tp 额外参数...
  local name=$1 gpus=$2 tp=$3; shift 3
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; exit 1; }
  [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-tp2-b 2>/dev/null) == true ]] && { log "vllm-tp2-b running, abort"; exit 3; }
  mkdir -p "$RESULT_DIR/$name"
  sudo docker rm -f $NAME >/dev/null 2>&1
  log "== $name: GPU $gpus tp$tp $*"
  # shellcheck disable=SC2068
  sudo docker run -d --name $NAME --ipc host --shm-size 16g --gpus "\"device=$gpus\"" -p 127.0.0.1:8005:8000 \
    -v $MD:$MD:ro -v "$DRAFT":/draft:ro -v "$HOME/loadtest/sglang-cache":/root/.cache \
    -e HF_HUB_OFFLINE=1 "$IMAGE" python3 -m sglang.launch_server \
    --model-path $MD/Qwen/Qwen3.8-27B-FP8 --served-model-name $M --host 0.0.0.0 --port 8000 \
    --tp-size "$tp" --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --disable-custom-all-reduce \
    --chunked-prefill-size 2048 --max-running-requests 64 --enable-metrics $@ >/dev/null
  for i in $(seq 1 120); do
    if curl -sf -m3 $URL/health >/dev/null && curl -sf -m3 $URL/v1/models >/dev/null; then
      log "ready after $((i*10))s"
      sudo docker logs $NAME 2>&1 | grep -E "max_total_num_tokens|max_running_requests|context_len|mamba|KV Cache is allocated|accept" | tail -8 | tee -a "$L"
      return 0
    fi
    if [[ $(sudo docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null) != true ]]; then
      sudo docker logs $NAME > "$RESULT_DIR/$name/deploy-fail.log" 2>&1
      log "$name exited"; grep -E "Error|error:|OutOfMemory|exceed|not enough" "$RESULT_DIR/$name/deploy-fail.log" | grep -v Traceback | tail -6 | tee -a "$L"
      return 1
    fi
    sleep 10
  done
  sudo docker logs $NAME > "$RESULT_DIR/$name/deploy-fail.log" 2>&1; log "$name timeout"; return 1
}

probe(){ # 名字
  local name=$1
  python3 "$D/prefill_probe.py" $URL $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  log "prefill $name"
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  log "decode $name"
  python3 "$D/decode_probe.py" $URL $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
    --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1,16" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  curl -s $URL/metrics | grep -iE "^sglang:(spec_accept|avg_spec)" | head -4 >> "$L"
  log "batch-shape $name"
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch 8005 llm --dataset-name random --ignore-eos --random-range-ratio 0.1 \
    --random-input-len 2048 --random-output-len 160 --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
  log "gsm $name"; gsm "$name"
  log "long $name"   # 124k 单请求（TP1 上下文 131072 也能放下），记录能否完成与 TTFT
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill-long.jsonl" --lens 126976 --conc 1 --rounds 1 >>"$L" 2>&1
}

# 基线：生产配置的 vLLM（默认 kernel、MTP2、mbt 默认 2048），同一套负载 + GSM
mkdir -p "$RESULT_DIR/v0-vllm-prod"
log "== v0-vllm-prod"
if IMAGE=vllm/vllm-openai:v0.25.1 MODEL_DIR=$MD FAIL_LOG=$RESULT_DIR/v0-vllm-prod/deploy-fail.log bash "$D/deploy_exp.sh" "" >>"$L" 2>&1; then probe v0-vllm-prod; else log "deploy v0 failed"; fi

# 镜像还在拉就等，最多 90 分钟
for i in $(seq 1 90); do sudo docker image inspect "$IMAGE" >/dev/null 2>&1 && break; [ $i = 1 ] && log "waiting for $IMAGE"; sleep 60; done
sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "no $IMAGE, skip sglang"; exit 1; }

# 每请求 L≈2.2k 时 cookbook 公式给出 ratio≈9.6，但那样 KV 池装不下 262k；取 2，状态槽仍够 64 路
deploy s1-tp2-mtp  2,3 2 --context-length 262144 --mem-fraction-static 0.80 --mamba-full-memory-ratio 2 $MTP && probe s1-tp2-mtp
deploy s2-tp2-dflash 2,3 2 --context-length 262144 --mem-fraction-static 0.80 --mamba-full-memory-ratio 2 $DFLASH && probe s2-tp2-dflash
# 单卡：GPU3（无 reranker），上下文 131072
deploy s3-tp1-mtp  3 1 --context-length 131072 --mem-fraction-static 0.88 --mamba-full-memory-ratio 2 $MTP && probe s3-tp1-mtp
sudo docker rm -f $NAME >/dev/null 2>&1
log "all done"
