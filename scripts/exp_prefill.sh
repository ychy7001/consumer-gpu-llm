#!/usr/bin/env bash
# prefill 调度实验：直连实验后端，比较不同 max-num-batched-tokens 下长输入 TTFT 与对短请求 decode 的干扰
# 用法: exp_prefill.sh <tag> [端口=8005]
set -u
TAG=$1; PORT=${2:-8005}
D=$(cd "$(dirname "$0")" && pwd)
export RESULT_DIR=$HOME/loadtest/results/$TAG
mkdir -p "$RESULT_DIR"
B="$D/bench.sh"
R="--dataset-name random --ignore-eos --random-range-ratio 0.1"

"$D/sampler.sh" "$PORT" & SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

# 单一负载
$B chat     "$PORT" llm $R --random-input-len 512   --random-output-len 256  --max-concurrency 16 --num-prompts 96
$B rag      "$PORT" llm $R --random-input-len 4096  --random-output-len 512  --max-concurrency 16 --num-prompts 64
$B summ     "$PORT" llm $R --random-input-len 16384 --random-output-len 1024 --max-concurrency 8  --num-prompts 24
$B ctx64k   "$PORT" llm $R --random-input-len 65536 --random-output-len 256  --max-concurrency 4  --num-prompts 8

# 混合：长输入持续进来，同时测短对话的 TPOT 长尾（prefill 抢占 decode 的程度）
$B mix-long "$PORT" llm $R --random-input-len 16384 --random-output-len 128  --max-concurrency 4  --num-prompts 32 & LPID=$!
sleep 20
$B mix-chat "$PORT" llm $R --random-input-len 512   --random-output-len 512  --max-concurrency 8  --num-prompts 48
wait $LPID
