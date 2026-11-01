#!/usr/bin/env bash
# E 补充（e2 因 --tag 参数写法失败，vllm-ref 已拆）：重新在 GPU1 起 FP8 基准，
# LongBench v2 short 以 max_tokens 8192 重跑三种部署，然后恢复 GPU1 batch 接回 8003。
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-pubeval; L=$OUT/run.log
D=$(dirname "$0")
source <(sed -n '/^declare -A PORT/p;/^run_c()/,/^}/p;/^ready()/,/^}/p' "$D/c-rotate.sh")
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
PE="$HOME/evalenv/bin/python $D/pubeval.py"

curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do r=$(curl -s -m5 127.0.0.1:8003/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}'); [[ ${r:-0} == 0 ]] && break; sleep 10; done
sudo docker update --restart=no vllm-hq-g1 >/dev/null; sudo docker stop -t 60 vllm-hq-g1 >/dev/null
IMAGE=vllm/vllm-openai:v0.25.1 TP=1 GPUS=1 PORT=8007 NAME=vllm-ref HELD=vllm-hq-g1 MODEL_DIR=/models \
  FAIL_LOG=$OUT/ref-deploy-fail.log \
  BASE="--max-num-seqs 32 --enable-prefix-caching --compilation-config {\"max_cudagraph_capture_size\":32} --gpu-memory-utilization 0.92" \
  bash "$S/deploy_exp.sh" "$VIS_ARGS" >>"$L" 2>&1 || log "FP8 ref deploy failed"

log "longbench 8k rerun start"
for x in 8007:fp8-ref 8005:hq-batch 8006:hq-single64; do
  p=${x%%:*}; n=${x#*:}
  curl -sf -m3 127.0.0.1:$p/v1/models >/dev/null || { log "skip $n, :$p down"; continue; }
  $PE http://127.0.0.1:$p $n "$OUT" --only longbench_v2 --max-tokens 8192 --tag=-8k > "$OUT/$n.lb8k.out" 2>&1 &
done
wait
log "longbench 8k reports: $(ls $OUT/*/longbench_v2-8k/reports/*/*.json 2>/dev/null | wc -l)/3"

sudo docker rm -f vllm-ref >/dev/null 2>&1
sudo docker update --restart=always vllm-hq-g1 >/dev/null; sudo docker start vllm-hq-g1 >/dev/null
if ready 8003; then curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"; else log "hq-g1 restore failed"; fi
log "workers $(curl -s 127.0.0.1:8000/list_workers)"
log "E3 done"
