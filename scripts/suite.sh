#!/usr/bin/env bash
# 一轮标准场景套件，用于对比不同部署配置
# 用法: suite.sh <tag> [quick|full]
#   quick: 仅吞吐扫描 + 典型长度（参数实验用）；full: 另含长上下文、辅助模型
set -u
TAG=$1; MODE=${2:-full}
D=$(cd "$(dirname "$0")" && pwd)
export RESULT_DIR=$HOME/loadtest/results/$TAG
mkdir -p "$RESULT_DIR"
B="$D/bench.sh"
R="--dataset-name random --ignore-eos --random-range-ratio 0.1"

# 开跑前确认无外部在途请求
for p in 8003 8004; do
  run=$(curl -s -m3 127.0.0.1:$p/metrics | awk '/^vllm:num_requests_running/{s+=$2}END{print s+0}')
  [[ ${run%.*} -gt 0 ]] && echo "WARN: $p 有 $run 个在途请求" | tee -a "$RESULT_DIR/timeline.log"
done

"$D/sampler.sh" "8003 8004" & SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

# S1/S2 吞吐-并发扫描（Router 入口，in1k/out256）
for c in 1 4 8 16 32 64; do
  n=$(( c*4 < 16 ? 16 : c*4 ))
  $B sweep-c$c 8000 llm $R --random-input-len 1024 --random-output-len 256 --max-concurrency $c --num-prompts $n
  sleep 10
done

# S3 典型长度组合，并发 16
$B chat 8000 llm $R --random-input-len 512  --random-output-len 256  --max-concurrency 16 --num-prompts 96
$B rag  8000 llm $R --random-input-len 4096 --random-output-len 512  --max-concurrency 16 --num-prompts 64
$B summ 8000 llm $R --random-input-len 16384 --random-output-len 1024 --max-concurrency 8 --num-prompts 24
$B longout 8000 llm $R --random-input-len 1024 --random-output-len 2048 --max-concurrency 16 --num-prompts 32
# 共享 4k 前缀（同一 system prompt / RAG 文档），评估 prefix caching
$B prefix-c16 8000 llm $R --random-prefix-len 4096 --random-input-len 512 --random-output-len 256 --max-concurrency 16 --num-prompts 96

# 真实文本（chat 模板 + thinking），评估 MTP 等与内容相关的优化
mkdir -p "$HOME/loadtest/results/_data" && cp "$D/../data/real_prompts.jsonl" "$HOME/loadtest/results/_data/"
$B real-c16 8000 chat --dataset-name custom --dataset-path /results/_data/real_prompts.jsonl --custom-output-len 1024 --max-concurrency 16 --num-prompts 64 --temperature 0.7
$B real-c1  8000 chat --dataset-name custom --dataset-path /results/_data/real_prompts.jsonl --custom-output-len 512 --max-concurrency 1 --num-prompts 6 --temperature 0.7

[[ $MODE == quick ]] && exit 0

# S4 长上下文
$B ctx64k  8000 llm $R --random-input-len 65536  --random-output-len 256 --max-concurrency 4 --num-prompts 8
$B ctx128k 8000 llm $R --random-input-len 131072 --random-output-len 256 --max-concurrency 2 --num-prompts 4
$B ctx200k 8000 llm $R --random-input-len 200000 --random-output-len 256 --max-concurrency 1 --num-prompts 2

# S6/S7 辅助模型
for c in 1 8 32; do
  $B emb512-c$c 8001 emb --dataset-name random --random-input-len 512 --random-output-len 1 --max-concurrency $c --num-prompts $(( c*40 < 80 ? 80 : c*40 ))
done
$B emb4k-c8 8001 emb --dataset-name random --random-input-len 4096 --random-output-len 1 --max-concurrency 8 --num-prompts 200
for c in 1 8 32; do
  $B rr512-c$c 8002 rerank --dataset-name random-rerank --random-input-len 512 --max-concurrency $c --num-prompts $(( c*20 < 40 ? 40 : c*20 ))
done
