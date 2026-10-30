#!/usr/bin/env bash
# 定稿：部署选定 LLM 参数 → 升级 Router → 全套件 → soak → 写 FINALIZED
# 用法: finalize.sh <tag> "<额外 vllm 参数>" <soak 请求速率 req/s> <soak 分钟数> ["<回退参数>"]
#   回退参数：定稿配置起不来时使用的已验证配置（默认原始 seqs=6）
set -u
D=$(cd "$(dirname "$0")" && pwd)
TAG=$1; EXTRA=$2; RATE=$3; MIN=$4; FALLBACK=${5:---max-num-seqs 6}
LOG=$HOME/loadtest/results/experiments.log
ROUTER_VENV=$HOME/vllm-router-venv-0.1.15

while pgrep -f "[s]cripts/experiments.sh" >/dev/null; do sleep 20; done
echo "[$(date +%T)] === finalize $TAG : $EXTRA" >> "$LOG"

# 1. Router 升级到 0.1.15（修复约 2MB 请求体上限导致的 413），旧 venv 保留用于回滚
if [[ ! -x $ROUTER_VENV/bin/vllm-router ]]; then
  python3 -m venv "$ROUTER_VENV" && "$ROUTER_VENV/bin/pip" install -q "vllm-router==0.1.15" >> "$LOG" 2>&1
fi
printf '[Service]\nExecStart=\nExecStart=%s/bin/vllm-router --host 0.0.0.0 --port 8000 --worker-urls http://127.0.0.1:8003 http://127.0.0.1:8004 --policy power_of_two --max-payload-size 67108864\n' \
  "$ROUTER_VENV" | sudo tee /etc/systemd/system/vllm-router.service.d/override.conf >/dev/null
sudo systemctl daemon-reload

# 2. LLM 定稿部署（deploy_llm 内含 Router 端到端就绪检查，故先重启 Router）
sudo systemctl restart vllm-router
if ! "$D/deploy_llm.sh" "$EXTRA" >> "$LOG" 2>&1; then
  echo "[$(date +%T)] finalize deploy FAILED, fallback: $FALLBACK" >> "$LOG"
  "$D/deploy_llm.sh" "$FALLBACK" >> "$LOG" 2>&1 || { "$D/deploy_llm.sh" "--max-num-seqs 6" >> "$LOG" 2>&1; exit 1; }
  TAG=$TAG-fallback; EXTRA=$FALLBACK
fi

# 3. 全套件（含长上下文、辅助模型）
"$D/suite.sh" "$TAG" full >> "$LOG" 2>&1

# 4. soak：Poisson 到达，LLM + 辅助模型同时施压（同时也是混部干扰场景）
export RESULT_DIR=$HOME/loadtest/results/$TAG
N=$(python3 -c "print(int($RATE*60*$MIN))")
"$D/sampler.sh" "8003 8004" & SPID=$!
"$D/bench.sh" soak-emb 8001 emb --dataset-name random --random-input-len 512 --random-output-len 1 \
  --request-rate 20 --num-prompts $((20*60*MIN)) --max-concurrency 16 & EPID=$!
"$D/bench.sh" soak-rr 8002 rerank --dataset-name random-rerank --random-input-len 512 \
  --request-rate 10 --num-prompts $((10*60*MIN)) --max-concurrency 16 & RPID=$!
"$D/bench.sh" soak-llm 8000 llm --dataset-name random --ignore-eos --random-range-ratio 0.5 \
  --random-input-len 2048 --random-output-len 384 --request-rate "$RATE" --num-prompts "$N" --max-concurrency 96
wait $EPID $RPID 2>/dev/null
kill $SPID 2>/dev/null

touch "$HOME/loadtest/FINALIZED"
echo "[$(date +%T)] finalize done" >> "$LOG"
