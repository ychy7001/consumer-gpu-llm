#!/usr/bin/env bash
# 单后端实验守护：在线后端持续排队或到达截止时间时，停实验容器、恢复被停后端并接回 Router
# 用法: guard_single_backend.sh <截止时间 "HH:MM"> [在线端口=8003] [被停容器=vllm-tp2-b] [被停端口=8004] [实验容器=vllm-exp]
#   连续 WAIT_LIMIT 次（每 15s 一次）waiting>0 视为持续排队；创建 $STOP_FILE 可手动结束守护（不恢复），
#   创建 $RESTORE_FILE 立即恢复（实验脚本跑完时用）
set -u
DEADLINE=$(date -d "$1" +%s); LIVE=${2:-8003}; HELD=${3:-vllm-tp2-b}; HELD_PORT=${4:-8004}; EXP=${5:-vllm-exp}
WAIT_LIMIT=${WAIT_LIMIT:-8}; STOP_FILE=${STOP_FILE:-$HOME/loadtest/guard.stop}
RESTORE_FILE=${RESTORE_FILE:-$HOME/loadtest/guard.restore}
LOG=$HOME/loadtest/guard.log
rm -f "$STOP_FILE" "$RESTORE_FILE"
log() { echo "[$(date +%T)] $*" >> "$LOG"; }

restore() {
  log "restore: $1"
  sudo docker rm -f "$EXP" >/dev/null 2>&1 || true
  sudo docker start "$HELD" >/dev/null
  for _ in $(seq 1 90); do curl -sf -m3 "127.0.0.1:$HELD_PORT/health" >/dev/null && break; sleep 10; done
  # 预热后再接回 Router
  for _ in 1 2 3; do
    curl -s -m60 -o /dev/null -H content-type:application/json \
      -d '{"model":"Qwen3.8-27B-FP8","prompt":"hi","max_tokens":8}' "127.0.0.1:$HELD_PORT/v1/completions"
  done
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:$HELD_PORT" >> "$LOG"; echo >> "$LOG"
  log "workers: $(curl -s -m5 127.0.0.1:8000/list_workers)"
  exit 0
}

log "guard start: live=$LIVE held=$HELD deadline=$1"
n=0
while true; do
  [[ -f $STOP_FILE ]] && { log "stop file, exit without restore"; exit 0; }
  [[ -f $RESTORE_FILE ]] && { rm -f "$RESTORE_FILE"; restore "restore file"; }
  (( $(date +%s) >= DEADLINE )) && restore "deadline"
  w=$(curl -s -m5 "127.0.0.1:$LIVE/metrics" | awk '/^vllm:num_requests_waiting[ {]/ && !/by_reason/{s+=$2}END{print int(s)}')
  if [[ ${w:-0} -gt 0 ]]; then n=$((n+1)); else n=0; fi
  (( n >= WAIT_LIMIT )) && restore "live backend waiting ${w} for $((n*15))s"
  sleep 15
done
