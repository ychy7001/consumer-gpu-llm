#!/usr/bin/env bash
# 10-31 凌晨：GPU 2,3 上 TP2 对比默认 / 调优后的 W8A8 block FP8 kernel，以及 MTP3、mbt 8192
# 负载：纯 prefill、真实提示词 decode、批处理形态（in 2048 / out 160，并发 24）
set -u
D=$HOME/loadtest/tool/scripts
export MODEL_DIR=/models
export RESULT_DIR=$HOME/loadtest/results/20261031-kernel-ab
L=$RESULT_DIR/run.log
mkdir -p "$RESULT_DIR"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
URL=http://127.0.0.1:8005; M=Qwen3.8-27B-FP8
spec(){ curl -s $URL/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{' | awk '{print $1, $2}'; }
K=$HOME/loadtest/tune/out

cfg(){ # 名字 kernel配置目录或空 mtp步数 mbt
  local name=$1 kcfg=$2 k=$3 mbt=$4
  [ -f "$HOME/loadtest/guard.stop" ] && { log "guard.stop found, abort"; exit 1; }
  log "== $name: kernel=${kcfg:-default} mtp=$k mbt=$mbt"
  mkdir -p "$RESULT_DIR/$name"
  local base="--max-num-seqs 64 --enable-prefix-caching --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$k} --gpu-memory-utilization 0.78"
  if ! KERNEL_CFG=$kcfg FAIL_LOG=$RESULT_DIR/$name/deploy-fail.log BASE="$base" bash "$D/deploy_exp.sh" "--max-num-batched-tokens $mbt" >>"$L" 2>&1; then
    log "deploy $name failed"; return 1
  fi
  sudo docker logs vllm-exp 2>&1 | grep -cE "Using configuration from .*W8A8 Block FP8" | sed 's/^/tuned kernel configs loaded: /' | tee -a "$L"
  python3 "$D/prefill_probe.py" $URL $M /tmp/warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  log "prefill $name"
  python3 "$D/prefill_probe.py" $URL $M "$RESULT_DIR/$name/prefill.jsonl" --lens 2048,16384,65536 --conc 1 --rounds 2 >>"$L" 2>&1
  spec > "$RESULT_DIR/$name/spec_before.txt"
  log "decode $name"
  python3 "$D/decode_probe.py" $URL $M "$RESULT_DIR/$name/decode.jsonl" --prompts "$HOME/loadtest/tool/data/real_prompts.jsonl" \
    --corpus-dir "$HOME/loadtest/tool/corpus" --grid "0:1,16" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  spec > "$RESULT_DIR/$name/spec_after.txt"
  log "batch-shape $name"
  RESULT_DIR=$RESULT_DIR/$name bash "$D/bench.sh" batch 8005 llm --dataset-name random --ignore-eos --random-range-ratio 0.1 \
    --random-input-len 2048 --random-output-len 160 --max-concurrency 24 --num-prompts 144 >>"$L" 2>&1
}

cfg k0-default-mtp2 ""  2 4096
cfg k1-tuned-mtp2   "$K" 2 4096
cfg k2-tuned-mtp3   "$K" 3 4096
cfg k3-tuned-mbt8k  "$K" 2 8192
log "all done (vllm-exp left running with last config)"
