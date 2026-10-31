#!/usr/bin/env bash
# 11-01：DP4（四个单卡 HyperQwen）Router 级压测
# power_of_two 走生产 Router :8000；cache_aware 起旁路 Router :8010（prometheus 29010），测完即删，不动生产 Router
# 负载与 10-31 同：random 2048 入 / 160 出（批处理），另加 16k 入 / 256 出长 prefill
# 第二轮：先 cache_aware 后 power_of_two（排除顺序影响），加共享前缀场景（09-30 两后端时 cache_aware 把共享前缀全压到一台）
# 对比基线：旧 2×TP2 合计约 4700 tok/s（docs/rtx4090/engine-tp.md，2×TP1 各并发 24 为 4774）
set -u
T=$HOME/loadtest/tool; S=$T/scripts
OUT=$HOME/loadtest/results/20261101-dp4b; mkdir -p "$OUT"
L=$OUT/run.log
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
BATCH="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"
LONG="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 16384 --random-output-len 256"
W="http://127.0.0.1:8003 http://127.0.0.1:8004 http://127.0.0.1:8005 http://127.0.0.1:8006"
succ(){ for p in 8003 8004 8005 8006; do printf "%s:%s " $p "$(curl -s -m5 127.0.0.1:$p/metrics | awk '/^vllm:request_success_total/{s+=$2}END{print int(s)}')"; done; }
b(){ # 名字 端口 参数...
  local n=$1 p=$2; shift 2
  log "start $n  succ $(succ)"
  RESULT_DIR=$OUT MODEL_DIR=/models bash "$S/bench.sh" "$n" "$p" llm "$@" >>"$L" 2>&1
  log "end   $n  succ $(succ)"
}
PREFIX="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-prefix-len 8192 --random-input-len 1024 --random-output-len 160"
suite(){ # 前缀 端口
  b "$1-c96" "$2" $BATCH --max-concurrency 96 --num-prompts 576
  b "$1-long-c16" "$2" $LONG --max-concurrency 16 --num-prompts 64
  b "$1-prefix-c48" "$2" $PREFIX --max-concurrency 48 --num-prompts 288
}

log "workers $(curl -s 127.0.0.1:8000/list_workers)"
setsid nohup vllm-router --host 127.0.0.1 --port 8010 --prometheus-port 29010 \
  --worker-urls $W --policy cache_aware --max-payload-size 67108864 > "$OUT/router-ca.out" 2>&1 < /dev/null &
RP=$!
for i in $(seq 1 30); do curl -sf -m3 127.0.0.1:8010/list_workers >/dev/null && break; sleep 2; done
log "side router pid $RP workers $(curl -s 127.0.0.1:8010/list_workers)"
suite ca 8010
kill $RP; sleep 2; kill -9 $RP 2>/dev/null
suite p2c 8000
grep -v -i "api.key" "$OUT/router-ca.out" | tail -5 > "$OUT/router-ca.tail"; rm -f "$OUT/router-ca.out"

python3 - "$OUT" <<'EOF' | tee -a "$L"
import json, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    tot = (d["total_input_tokens"] + d["total_output_tokens"]) / d["duration"]
    print(f"{os.path.basename(f)[:-5]:14s} ok {d['completed']}/{d['num_prompts']} tot {tot:7.0f} out {d['output_throughput']:6.0f} "
          f"ttft p50/p99 {d['median_ttft_ms']:7.0f}/{d['p99_ttft_ms']:7.0f} tpot p50/p99 {d['median_tpot_ms']:5.0f}/{d['p99_tpot_ms']:5.0f}")
EOF
log "dp4 done"
