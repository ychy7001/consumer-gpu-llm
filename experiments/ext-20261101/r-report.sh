#!/usr/bin/env bash
# 11-02 汇报补测：最终部署（四卡 single 32 槽 CG128）按 09-30 suite.sh 同场景复测，便于同条件对比
#   R1–R3 三轮：吞吐扫描 c1–c64（另加 c96/c128）+ 典型长度 + 共享前缀 + 真实中文；R1 另测长上下文与向量/重排
#   SOAK 50 分钟混合长稳：LLM 1 req/s Poisson（2k±50% / 384）+ 向量 20 req/s + 重排 10 req/s
# 只对生产 Router 施压，不改部署；每轮前等无真实流量
set -u
source "$(dirname "$0")/lib.sh"
BASE=$HOME/loadtest/results/20261102-report; mkdir -p "$BASE"; L=$BASE/run.log
B=$S/bench.sh
R="--dataset-name random --ignore-eos --random-range-ratio 0.1"
vbc(){ # 名字 端口 bench参数...（chat 接口，真实提示词）
  local n=$1 p=$2; shift 2
  [ -f "$OUT/$n.json" ] && { log "skip $n"; return; }
  sudo docker exec hq-client /app/venv/bin/vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://127.0.0.1:$p --model $M --tokenizer $TOK \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "/results/${OUT#$HOME/loadtest/results/}" --result-filename "$n.json" "$@" > "$OUT/$n.log" 2>&1 \
    || log "FAIL $n"
}
restarts(){ sudo docker ps --format '{{.Names}}' | grep -E '^vllm-' | sort | while read c; do printf "%s=%s " "$c" "$(sudo docker inspect -f '{{.RestartCount}}' "$c")"; done; }
router_err(){ curl -s -m3 127.0.0.1:29000/metrics | grep -E '^vllm_router_(request_errors_total|retries_total)' | awk '{s+=$2}END{print int(s)}'; }
client
cp "$T/data/real_prompts.jsonl" "$HOME/loadtest/results/_data/" 2>/dev/null

for r in 1 2 3; do
  OUT=$BASE/r$r; mkdir -p "$OUT"; export RESULT_DIR=$OUT
  waitidle; log "round $r start"
  "$S/sampler.sh" "$PORTS" & SPID=$!
  for c in 1 4 8 16 32 64 96 128; do
    n=$(( c*4 < 16 ? 16 : c*4 ))
    vb sweep-c$c 8000 $R --random-input-len 1024 --random-output-len 256 --max-concurrency $c --num-prompts $n
    sleep 10
  done
  vb chat 8000 $R --random-input-len 512 --random-output-len 256 --max-concurrency 16 --num-prompts 96
  vb rag 8000 $R --random-input-len 4096 --random-output-len 512 --max-concurrency 16 --num-prompts 64
  vb summ 8000 $R --random-input-len 16384 --random-output-len 1024 --max-concurrency 8 --num-prompts 24
  vb longout 8000 $R --random-input-len 1024 --random-output-len 2048 --max-concurrency 16 --num-prompts 32
  vb prefix-c16 8000 $R --random-prefix-len 4096 --random-input-len 512 --random-output-len 256 --max-concurrency 16 --num-prompts 96
  vbc real-c16 8000 --dataset-name custom --dataset-path /results/_data/real_prompts.jsonl --custom-output-len 1024 --max-concurrency 16 --num-prompts 64 --temperature 0.7
  vbc real-c1 8000 --dataset-name custom --dataset-path /results/_data/real_prompts.jsonl --custom-output-len 512 --max-concurrency 1 --num-prompts 6 --temperature 0.7
  if [ $r = 1 ]; then
    vb ctx64k 8000 $R --random-input-len 65536 --random-output-len 256 --max-concurrency 4 --num-prompts 8
    vb ctx128k 8000 $R --random-input-len 131072 --random-output-len 256 --max-concurrency 2 --num-prompts 4
    vb ctx200k 8000 $R --random-input-len 200000 --random-output-len 256 --max-concurrency 1 --num-prompts 2
    # 新部署四卡，加测长上下文并发 4
    vb ctx128k-c4 8000 $R --random-input-len 131072 --random-output-len 256 --max-concurrency 4 --num-prompts 8
    for c in 1 8 32; do
      $B emb512-c$c 8001 emb --dataset-name random --random-input-len 512 --random-output-len 1 --max-concurrency $c --num-prompts $(( c*40 < 80 ? 80 : c*40 )) >>"$L" 2>&1
    done
    $B emb4k-c8 8001 emb --dataset-name random --random-input-len 4096 --random-output-len 1 --max-concurrency 8 --num-prompts 200 >>"$L" 2>&1
    for c in 1 8 32; do
      $B rr512-c$c 8002 rerank --dataset-name random-rerank --random-input-len 512 --max-concurrency $c --num-prompts $(( c*20 < 40 ? 40 : c*20 )) >>"$L" 2>&1
    done
  fi
  kill $SPID 2>/dev/null
  log "round $r done"; summ >> "$L"
done

# 50 分钟混合长稳
OUT=$BASE/soak; mkdir -p "$OUT"; export RESULT_DIR=$OUT
waitidle
log "soak start; restarts: $(restarts); router errors: $(router_err)"
"$S/sampler.sh" "$PORTS" & SPID=$!
vb soak-llm 8000 --dataset-name random --ignore-eos --random-input-len 2048 --random-range-ratio 0.5 --random-output-len 384 \
  --request-rate 1 --num-prompts 3000 --seed 7 &
P1=$!
$B soak-emb 8001 emb --dataset-name random --random-input-len 512 --random-output-len 1 --request-rate 20 --num-prompts 60000 >>"$L" 2>&1 &
P2=$!
$B soak-rr 8002 rerank --dataset-name random-rerank --random-input-len 512 --request-rate 10 --num-prompts 30000 >>"$L" 2>&1 &
P3=$!
wait $P1 $P2 $P3
kill $SPID 2>/dev/null
log "soak done; restarts: $(restarts); router errors: $(router_err)"
summ >> "$L"
log "REPORT ALL DONE"
