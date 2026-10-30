#!/usr/bin/env bash
# 15s 粒度采样 GPU 与后端 metrics，写入 $RESULT_DIR/samples/
# 用法: sampler.sh <端口列表，如 "8003 8004">；后台运行，kill 结束
set -u
RESULT_DIR=${RESULT_DIR:?}; PORTS=${1:-"8003 8004"}
mkdir -p "$RESULT_DIR/samples"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
  --format=csv,noheader -l 15 >> "$RESULT_DIR/samples/gpu.csv" &
GPID=$!
trap 'kill $GPID 2>/dev/null' EXIT
while true; do
  ts=$(date +%s)
  for p in $PORTS; do
    curl -s -m3 "127.0.0.1:$p/metrics" | grep -E '^vllm:(num_requests_(running|waiting)|kv_cache_usage_perc|num_preemptions_total|prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)' \
      | sed "s/^/$ts $p /" >> "$RESULT_DIR/samples/metrics.txt"
  done
  # Router 指标（熔断、worker 健康、重试、错误）
  curl -s -m3 127.0.0.1:29000/metrics | grep -E '^vllm_router_(cb_state|worker_health|request_errors_total|retries_total|requests_total|active_workers)' \
    | sed "s/^/$ts router /" >> "$RESULT_DIR/samples/metrics.txt"
  sleep 15
done
