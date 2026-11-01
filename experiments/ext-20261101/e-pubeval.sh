#!/usr/bin/env bash
# 11-01 扩展 E：公开测试集质量门槛（evalscope），三种部署并行、同题对比
#   GPU1 :8007  vLLM 0.25.1 FP8 TP1（原生产权重，质量基准；临时停 hq-g1，摘 8003）
#   GPU2 :8005  HyperQwen batch（现生产配置）
#   GPU0 :8006  HyperQwen single+MTP+PTH CG64 16 槽（候选）
# Router 期间挂 8004/8005/8006（GPU3 灰度不动）；结束后 GPU1 恢复 batch 接回 8003，GPU0 留在 CG64 等切换决定
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-pubeval; mkdir -p "$OUT"; L=$OUT/run.log
D=$(dirname "$0")
source <(sed -n '/^declare -A PORT/p;/^run_c()/,/^}/p;/^ready()/,/^}/p;/^swap()/,/^}/p' "$D/c-rotate.sh")
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'
PE="$HOME/evalenv/bin/python $D/pubeval.py"

# 等 D 结束、guard 恢复 GPU3
until grep -q "D exit" $HOME/loadtest/ext-20261101.chain.log 2>/dev/null; do sleep 30; done
for i in $(seq 1 60); do [ "$(curl -s 127.0.0.1:8000/list_workers | grep -c 8004)" = 1 ] && break; sleep 10; done
log "workers $(curl -s 127.0.0.1:8000/list_workers)"

# GPU0 -> single CG64
swap 0 cg64
# GPU1 -> FP8 TP1 基准
curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do r=$(curl -s -m5 127.0.0.1:8003/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}'); [[ ${r:-0} == 0 ]] && break; sleep 10; done
sudo docker update --restart=no vllm-hq-g1 >/dev/null; sudo docker stop -t 60 vllm-hq-g1 >/dev/null
IMAGE=vllm/vllm-openai:v0.25.1 TP=1 GPUS=1 PORT=8007 NAME=vllm-ref HELD=vllm-hq-g1 MODEL_DIR=/models \
  FAIL_LOG=$OUT/ref-deploy-fail.log \
  BASE="--max-num-seqs 32 --enable-prefix-caching --compilation-config {\"max_cudagraph_capture_size\":32} --gpu-memory-utilization 0.92" \
  bash "$S/deploy_exp.sh" "$VIS_ARGS" >>"$L" 2>&1 || log "FP8 ref deploy failed"
log "workers $(curl -s 127.0.0.1:8000/list_workers)"

for x in 8007:fp8-ref 8005:hq-batch 8006:hq-single64; do
  p=${x%%:*}; n=${x#*:}
  curl -sf -m3 127.0.0.1:$p/v1/models >/dev/null || { log "skip $n, :$p down"; continue; }
  $PE http://127.0.0.1:$p $n "$OUT" > "$OUT/$n.out" 2>&1 &
done
wait
log "evals done"

# GPU1 恢复 batch，接回 8003
sudo docker rm -f vllm-ref >/dev/null 2>&1
sudo docker update --restart=always vllm-hq-g1 >/dev/null; sudo docker start vllm-hq-g1 >/dev/null
if ready 8003; then curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"; else log "hq-g1 restore failed"; fi
log "workers $(curl -s 127.0.0.1:8000/list_workers)"
log "E done"
