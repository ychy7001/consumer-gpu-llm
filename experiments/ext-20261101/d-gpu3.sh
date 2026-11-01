#!/usr/bin/env bash
# 11-01 扩展 D：借 GPU3（摘 hq-g3，Router 剩 3 个后端；guard 盯排队、到点恢复）
#   D1 Router 故障恢复：旁路 Router :8010（3 个生产后端 + 实验实例 :8007），c32 持续负载，
#      t=60s kill 实验实例、t=150s 重启，1 s 采样实验实例健康与成功数；默认参数 vs 快速健康检查
#   D2 batch 调度参数：mbt 2048(默认)/4096/8192、8192+long_prefill 2048；64k 文档 c2 与短对话 c24 同时跑，
#      看短对话 TTFT/TPOT 长尾与文档 TTFT；另测纯 prefill 16k/64k 和 KV 池
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-gpu3; mkdir -p "$OUT"; L=$OUT/run.log
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
DOC64="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 65536 --random-output-len 256"
EP=8007
exp(){ # 额外参数...
  sudo docker rm -f vllm-exp >/dev/null 2>&1
  sudo docker run -d --name vllm-exp --restart no --ipc host --gpus '"device=3"' -p "127.0.0.1:$EP:8000" \
    -v "$ROOT/models":/app/models -v "$ROOT/cache-gpu3":/cache \
    -e HOME=/cache -e HOST=0.0.0.0 -e PORT=8000 -e PREPARE=0 -e VERIFY=0 \
    -e KV=fp8 -e INT8_ACT=int8 -e MAX_LEN=262144 -e GPU_UTIL=0.92 -e PREFIX_CACHE=1 \
    -e VISION=1 -e EXTRA_ARGS="--served-model-name $M $VIS_ARGS $*" $IMAGE batch >/dev/null
  local i; for i in $(seq 1 72); do curl -sf -m3 127.0.0.1:$EP/v1/models >/dev/null && break; sleep 10; done
  curl -sf -m3 127.0.0.1:$EP/v1/models >/dev/null || { log "exp not ready: $*"; return 1; }
  curl -s -m60 -o /dev/null 127.0.0.1:$EP/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"prompt\":\"hi\",\"max_tokens\":8}"
  log "exp up ($*): $(sudo docker logs vllm-exp 2>&1 | grep -o 'GPU KV cache size: [0-9,]* tokens' | tail -1)"
}
gpu3_live(){ [ "$(sudo docker inspect -f '{{.State.Running}}' vllm-hq-g3 2>/dev/null)" = true ]; }
finish(){ touch "$HOME/loadtest/guard.restore"; log "guard.restore touched"; }

# 摘 GPU3，guard 兜底（盯 8003/8005/8006 排队，4 小时后到点恢复）
curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8004" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do r=$(curl -s -m5 127.0.0.1:8004/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}'); [[ ${r:-0} == 0 ]] && break; sleep 10; done
sudo docker update --restart=no vllm-hq-g3 >/dev/null; sudo docker stop -t 120 vllm-hq-g3 >/dev/null
(cd "$S" && WAIT_MIN=24 WAIT_LIMIT=20 setsid nohup bash guard_single_backend.sh "$(date -d '+240 minutes' '+%F %H:%M')" 8003,8005,8006 vllm-hq-g3 8004 vllm-exp >/dev/null 2>&1 </dev/null &)
trap finish EXIT
PORTS="8003 8005 8006"
client
log "GPU3 free"

# D1 Router 故障恢复
sampler(){ # 输出文件
  local t0=$(date +%s)
  while :; do
    h=$(curl -s -o /dev/null -w '%{http_code}' -m2 127.0.0.1:$EP/health)
    m=$(curl -s -m2 127.0.0.1:$EP/metrics | awk '/^vllm:request_success_total/{s+=$2}/^vllm:time_to_first_token_seconds_sum/{ts=$2}/^vllm:time_to_first_token_seconds_count/{tc=$2}END{print int(s), ts+0, tc+0}')
    echo "$(( $(date +%s) - t0 )) $h $m" >> "$1"; sleep 1
  done
}
failover(){ # 名字 Router参数...
  local n=$1; shift
  exp || return 1
  setsid nohup ~/vllm-router-venv-0.1.15/bin/vllm-router --host 127.0.0.1 --port 8010 --prometheus-port 29010 \
    --worker-urls http://127.0.0.1:8003 http://127.0.0.1:8005 http://127.0.0.1:8006 http://127.0.0.1:$EP \
    --policy power_of_two --max-payload-size 67108864 "$@" > "$OUT/router-$n.out" 2>&1 < /dev/null &
  local rp=$!
  for i in $(seq 1 30); do curl -sf -m3 127.0.0.1:8010/list_workers >/dev/null && break; sleep 2; done
  waitidle
  sampler "$OUT/fo-$n.samples" & local sp=$!
  vb fo-$n 8010 $CHAT --max-concurrency 32 --num-prompts 2400 &
  local bp=$!
  sleep 60; sudo docker kill vllm-exp >/dev/null; log "$n: killed exp"
  sleep 90; sudo docker start vllm-exp >/dev/null; log "$n: restarted exp"
  wait $bp; kill $sp; kill $rp; sleep 2; kill -9 $rp 2>/dev/null
  grep -iE "circuit|unhealthy|healthy|removed|failed" "$OUT/router-$n.out" | grep -vi "api.key" | sed 's/\x1b\[[0-9;]*m//g' | tail -30 > "$OUT/router-$n.events"
  rm -f "$OUT/router-$n.out"
  log "$n done: $(grep -E 'Failed requests|Successful' "$OUT/fo-$n.log" | tr -s ' ' | tr '\n' ' ')"
}
failover default
gpu3_live && exit 0
failover fast --health-check-interval-secs 5 --health-failure-threshold 2 --health-success-threshold 1 --cb-timeout-duration-secs 10
gpu3_live && exit 0

# D2 batch 调度参数
for cfg in "m2048:" "m4096:--max-num-batched-tokens 4096" "m8192:--max-num-batched-tokens 8192" "m8192lp:--max-num-batched-tokens 8192 --long-prefill-token-threshold 2048"; do
  n=${cfg%%:*}; a=${cfg#*:}
  gpu3_live && exit 0
  exp $a || continue
  python3 "$S/prefill_probe.py" http://127.0.0.1:$EP $M "$OUT/$n-prefill.jsonl" --lens 16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  vb $n-mixdoc $EP $DOC64 --max-concurrency 2 --num-prompts 12 &
  sleep 5
  vb $n-mixchat $EP $CHAT --max-concurrency 24 --num-prompts 480
  wait
  vb $n-chat-c24 $EP $CHAT --max-concurrency 24 --num-prompts 144
  log "$n done"
done
sudo docker rm -f vllm-exp >/dev/null 2>&1
summ | tee -a "$L"
log "D done"
