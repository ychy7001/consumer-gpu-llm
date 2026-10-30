#!/usr/bin/env bash
# 在 gpu-host 上用临时容器执行一次 vllm bench serve，结果写入 $RESULT_DIR/<name>.json
# 用法: bench.sh <name> <port> <kind> [额外 bench 参数...]
#   kind: llm | emb | rerank
set -euo pipefail

NAME=$1; PORT=$2; KIND=$3; shift 3
RESULT_DIR=${RESULT_DIR:?需要设置 RESULT_DIR}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.25.1}
MODEL_DIR=${MODEL_DIR:-/models}
MODELS=$MODEL_DIR/Qwen
mkdir -p "$RESULT_DIR"

case $KIND in
  llm)    MODEL=Qwen3.8-27B-FP8;     BACKEND=vllm;              ENDPOINT=/v1/completions; PCT=ttft,tpot,itl,e2el ;;
  chat)   MODEL=Qwen3.8-27B-FP8;     BACKEND=openai-chat;       ENDPOINT=/v1/chat/completions; PCT=ttft,tpot,itl,e2el ;;
  emb)    MODEL=Qwen3-Embedding-0.6B; BACKEND=openai-embeddings; ENDPOINT=/v1/embeddings; PCT=e2el ;;
  rerank) MODEL=Qwen3-Reranker-0.6B;  BACKEND=vllm-rerank;       ENDPOINT=/v1/rerank; PCT=e2el ;;
  *) echo "unknown kind $KIND" >&2; exit 2 ;;
esac

# 已存在则跳过，便于中断后续跑
if [[ -f "$RESULT_DIR/$NAME.json" ]]; then echo "skip $NAME"; exit 0; fi

echo "[$(date +%T)] start $NAME $*" | tee -a "$RESULT_DIR/timeline.log"
# 常驻客户端容器，避免每次冷启动开销
if ! sudo docker ps --format '{{.Names}}' | grep -qx lt-client; then
  sudo docker rm -f lt-client >/dev/null 2>&1 || true
  sudo docker run -d --name lt-client --network host \
    -v "$MODEL_DIR":"$MODEL_DIR":ro \
    -v "$HOME/loadtest/results":/results \
    --entrypoint sleep "$IMAGE" infinity >/dev/null
fi
REL=${RESULT_DIR#$HOME/loadtest/results/}
sudo docker exec lt-client vllm bench serve \
  --backend "$BACKEND" --endpoint "$ENDPOINT" \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" --tokenizer "$MODELS/$MODEL" \
  --percentile-metrics "$PCT" --metric-percentiles 50,90,99 \
  --save-result --result-dir "/results/$REL" --result-filename "$NAME.json" \
  "$@" > "$RESULT_DIR/$NAME.log" 2>&1 || echo "[$(date +%T)] FAIL $NAME" | tee -a "$RESULT_DIR/timeline.log"
# 全部失败的结果视为无效，删除以便重跑
python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['completed']>0 else 1)" "$RESULT_DIR/$NAME.json" 2>/dev/null \
  || { mv -f "$RESULT_DIR/$NAME.json" "$RESULT_DIR/$NAME.failed.json" 2>/dev/null; echo "[$(date +%T)] INVALID $NAME" | tee -a "$RESULT_DIR/timeline.log"; }
echo "[$(date +%T)] end $NAME" | tee -a "$RESULT_DIR/timeline.log"
