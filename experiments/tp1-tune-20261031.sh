#!/usr/bin/env bash
# 10-31 白天：GPU3 单卡 TP1 细调（kernel 配置、mbt、max-num-seqs、MTP 步数），之后 vLLM 0.31 同参数对比
# 先从 Router 摘下 8004、等在途清空再停 b；结束写 guard.restore 恢复
set -u
D=$HOME/loadtest/tool/scripts
export MODEL_DIR=/models
RESULT_DIR=$HOME/loadtest/results/20261031-tp1-tune
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
URL=http://127.0.0.1:8005; M=Qwen3.8-27B-FP8
K=$HOME/loadtest/tune/out
BATCH="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"
export TP=1 GPUS=3

# 1. 摘流、等在途清空（最多 15 分钟）、停 b、起 guard
curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8004" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do
  r=$(curl -s -m5 127.0.0.1:8004/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
  [[ ${r:-0} == 0 ]] && break; sleep 10
done
log "b running=${r:-0} after $((i*10))s, stopping"
sudo docker stop -t 120 vllm-tp2-b >/dev/null
(cd "$D" && WAIT_MIN=16 WAIT_LIMIT=20 setsid nohup bash guard_single_backend.sh "$(date -d '+5 hours' '+%F %H:%M')" 8003 vllm-tp2-b 8004 \
  "vllm-exp vllm-tune-0 vllm-tune-1" >/dev/null 2>&1 </dev/null &)
sleep 2

cfg(){ # 名字 seqs mtp(0=关) mbt kernel(1/0) [镜像]
  local name=$1 seqs=$2 k=$3 mbt=$4 kern=$5 image=${6:-vllm/vllm-openai:v0.25.1}
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; return 1; }
  [[ $(sudo docker inspect -f '{{.State.Running}}' vllm-tp2-b 2>/dev/null) == true ]] && { log "b restored by guard, abort"; return 1; }
  mkdir -p "$RESULT_DIR/$name"
  local spec=""; [[ $k != 0 ]] && spec="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$k}"
  local base="--max-num-seqs $seqs --enable-prefix-caching $spec --compilation-config {\"max_cudagraph_capture_size\":$seqs} --gpu-memory-utilization 0.92"
  local kc=""; [[ $kern == 1 ]] && kc=$K
  log "== $name: seqs=$seqs mtp=$k mbt=$mbt kernel=$kern image=$image"
  if ! IMAGE=$image KERNEL_CFG=$kc FAIL_LOG=$RESULT_DIR/$name/deploy-fail.log BASE="$base" \
      bash "$D/deploy_exp.sh" "--max-model-len 262144 --max-num-batched-tokens $mbt" >>"$L" 2>&1; then
    log "deploy $name failed"; return 1
  fi
  sudo docker logs vllm-exp 2>&1 | grep -E "GPU KV cache size|Using default W8A8 Block FP8|Using configuration from .*W8A8" | sed -E 's/.*(GPU KV cache size.*|Using default.*N=[0-9]+,K=[0-9]+|Using configuration from.*json).*/\1/' | sort | uniq -c | tee -a "$L"
  python3 "$D/prefill_probe.py" $URL $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384 --conc 1 --rounds 2 >>"$L" 2>&1
  python3 "$D/decode_probe.py" $URL $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
    --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1" --max-tokens 512 --rounds 3 >>"$L" 2>&1
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch 8005 llm $BATCH --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
  python3 - "$RESULT_DIR/$name/batch.json" <<'EOF' | tee -a "$L"
import json,sys
d=json.load(open(sys.argv[1]))
print(f"batch total {d['total_token_throughput']:.0f} tok/s, out {d['output_throughput']:.0f}, ttft p50/p99 {d['median_ttft_ms']:.0f}/{d['p99_ttft_ms']:.0f} ms, tpot p50/p99 {d['median_tpot_ms']:.0f}/{d['p99_tpot_ms']:.0f} ms")
EOF
}

cfg p0-base       32 2 2048 0     # 昨晚 t1 的配置，复测看噪声
cfg p1-kernel     32 2 2048 1
cfg p2-mbt4k      32 2 4096 1
cfg p3-seqs48     48 2 2048 1
cfg p4-mtp1       32 1 2048 1
cfg p5-nospec     32 0 2048 1
# vLLM 0.31（镜像已拉到才测）；kernel 配置目录在包内，0.31 若换了 python 版本会挂不上，日志里看是否加载
V31=docker.m.daocloud.io/vllm/vllm-openai:v0.31.0
if sudo docker image inspect $V31 >/dev/null 2>&1; then
  cfg q0-v031       32 2 2048 0 $V31
  cfg q1-v031-kern  32 2 2048 1 $V31
else
  log "no $V31, skip"
fi
sudo docker rm -f vllm-exp >/dev/null 2>&1
log "sweep done"
