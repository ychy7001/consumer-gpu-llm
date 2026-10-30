#!/usr/bin/env bash
# 10-31 凌晨：vLLM TP1。SGLang TP1 单卡 prefill 比 TP2 高 ~45%，验证 vLLM TP1 单卡与 2×TP1 并行能否胜过 1×TP2
set -u
D=$HOME/loadtest/tool/scripts
export MODEL_DIR=/models
RESULT_DIR=$HOME/loadtest/results/20261031-tp1b
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
M=Qwen3.8-27B-FP8
export IMAGE=vllm/vllm-openai:v0.25.1 TP=1
# 第一次（seqs 64）在 cudagraph 显存预估时 OOM：权重 29G + 64 路捕获超出单卡，seqs 与捕获上限降到 32
BASE0='--max-num-seqs 32 --enable-prefix-caching --speculative-config {"method":"mtp","num_speculative_tokens":2} --compilation-config {"max_cudagraph_capture_size":32}'
BATCH="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"

sudo docker rm -f vllm-exp >/dev/null 2>&1

dep(){ # 名字 GPU 端口 容器名 gmu max-model-len
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; exit 1; }
  mkdir -p "$RESULT_DIR/$1"
  GPUS=$2 PORT=$3 NAME=$4 FAIL_LOG=$RESULT_DIR/$1/deploy-fail-$4.log BASE="$BASE0 --gpu-memory-utilization $5" \
    bash "$D/deploy_exp.sh" "--max-model-len $6" >>"$L" 2>&1
}

probe(){ # 名字 端口
  local name=$1 url=http://127.0.0.1:$2
  python3 "$D/prefill_probe.py" $url $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  log "prefill $name"
  python3 "$D/prefill_probe.py" $url $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  log "decode $name"
  python3 "$D/decode_probe.py" $url $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
    --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1,16" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  log "batch-shape $name"
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch $2 llm $BATCH --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
  log "gsm $name"
  python3 "$D/gsm_probe.py" $url $M "$HOME/loadtest/tool/data/gsm200.jsonl" "$RESULT_DIR/$name/gsm.json" --conc 32 2>&1 | tee -a "$L"
  log "long $name"
  python3 "$D/prefill_probe.py" $url $M "$RESULT_DIR/$name/prefill-long.jsonl" --lens 126976 --conc 1 --rounds 1 >>"$L" 2>&1
}

# t1：单卡 GPU3。先试 262144，起不来退到 131072
log "== t1-tp1 GPU3"
if dep t1-tp1 3 8005 vllm-exp 0.92 262144; then CTX=262144; elif dep t1-tp1 3 8005 vllm-exp 0.92 131072; then CTX=131072; else log "t1 deploy failed"; exit 1; fi
log "t1 ctx=$CTX"; probe t1-tp1 8005
sudo docker rm -f vllm-exp >/dev/null 2>&1

# t2：GPU2、GPU3 各一个 TP1（GPU2 上有 reranker，gmu 取 0.88），两路同时压 批处理形态。
# 第二个容器借用 guard 的 EXP 名单里的 vllm-tune-0，guard 恢复时会一并删掉
log "== t2-2xtp1 ctx=$CTX"
mkdir -p "$RESULT_DIR/t2-2xtp1"
dep t2-2xtp1 3 8005 vllm-exp 0.88 $CTX && dep t2-2xtp1 2 8006 vllm-tune-0 0.88 $CTX || { log "t2 deploy failed"; exit 1; }
for c in 12 24; do
  log "t2 batch 2x conc $c"
  for p in 8005 8006; do
    RESULT_DIR=$RESULT_DIR/t2-2xtp1 bash "$D/bench.sh" batch-c$c-$p $p llm $BATCH --max-concurrency $c --num-prompts $((c*6)) >>"$L" 2>&1 &
  done
  wait
done
sudo docker rm -f vllm-exp vllm-tune-0 >/dev/null 2>&1
log "all done"
