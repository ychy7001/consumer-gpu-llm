#!/usr/bin/env bash
# 在空出的两张卡上起实验容器（不经 Router，压测直连），参数默认与定稿一致
# 用法: deploy_exp.sh "<额外 vllm 参数>"   环境变量: GPUS=2,3 TP=2 PORT=8005 NAME=vllm-exp IMAGE=... BASE="<替换定稿参数>"
#   例: deploy_exp.sh "--max-num-batched-tokens 4096"
#       BASE="--max-num-seqs 64 --enable-prefix-caching" deploy_exp.sh "--pipeline-parallel-size 2 --tensor-parallel-size 1"
set -euo pipefail
EXTRA=${1:-}
GPUS=${GPUS:-2,3}; PORT=${PORT:-8005}; NAME=${NAME:-vllm-exp}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.25.1}
MODEL_DIR=${MODEL_DIR:-/models}
M=$MODEL_DIR/Qwen/Qwen3.8-27B-FP8
CACHE=${CACHE:-$HOME/loadtest/vllm-cache}
# JSON 里的 } 会提前结束 ${BASE:-...}，所以用 if 赋默认值
if [[ -z ${BASE:-} ]]; then
  BASE='--max-num-seqs 64 --enable-prefix-caching --speculative-config {"method":"mtp","num_speculative_tokens":2} --gpu-memory-utilization 0.78'
fi

# 被腾出卡的线上后端（如守护脚本已恢复）还在运行就不起实验容器
HELD=${HELD:-vllm-tp2-b}
if [[ $(sudo docker inspect -f '{{.State.Running}}' "$HELD" 2>/dev/null) == true ]]; then
  echo "$HELD is running on the target GPUs, refuse to deploy"; exit 3
fi
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
sudo docker run -d --name "$NAME" --ipc host \
  --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:8000" \
  -v "$MODEL_DIR":"$MODEL_DIR" \
  -v "$CACHE":/root/.cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/inductor \
  -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
  "$IMAGE" "$M" --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 --tensor-parallel-size "${TP:-2}" --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --served-model-name Qwen3.8-27B-FP8 $BASE $EXTRA >/dev/null
echo "[$(date +%T)] $NAME on GPU $GPUS :$PORT : $BASE $EXTRA"

for i in $(seq 1 90); do
  if curl -sf -m3 "127.0.0.1:$PORT/health" >/dev/null; then
    echo "[$(date +%T)] ready after $((i*10))s"
    sudo docker logs "$NAME" 2>&1 | grep -E 'max_num_batched_tokens|chunked prefill|GPU KV cache size|Maximum concurrency|init engine' | tail -6
    exit 0
  fi
  if [[ $(sudo docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null) != true ]]; then
    echo "$NAME exited, full log: ${FAIL_LOG:=$HOME/loadtest/deploy-fail-$(date +%H%M%S).log}"
    sudo docker logs "$NAME" > "$FAIL_LOG" 2>&1
    grep -E 'Error|error:|ValueError|OutOfMemory|exceeds|not enough|insufficient' "$FAIL_LOG" | grep -v Traceback | tail -8
    exit 1
  fi
  sleep 10
done
echo "timeout"; exit 1
