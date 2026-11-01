# 11-01 扩展实验公共函数（source 使用）
# 安全：每步前等无真实流量（四个后端 running+waiting=0 且本机没有压测进程），最多等 15 分钟，否则整体退出；
# 创建 ~/loadtest/ext.stop 随时中止
T=$HOME/loadtest/tool; S=$T/scripts
ROOT=/models/hyperqwen
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
M=Qwen3.8-27B-FP8
TOK=/m/Qwen3.8-27B-W4A16-AutoRound
STOPF=$HOME/loadtest/ext.stop
PORTS="8003 8004 8005 8006"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
load(){ local s=0 p; for p in $PORTS; do s=$((s + $(curl -s -m5 127.0.0.1:$p/metrics | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$2}END{print int(s)}'))); done; echo $s; }
succ(){ local p; for p in $PORTS; do printf "%s:%s " $p "$(curl -s -m5 127.0.0.1:$p/metrics | awk '/^vllm:request_success_total/{s+=$2}END{print int(s)}')"; done; }
waitidle(){ # 本脚本无在途压测时调用
  local i
  [ -f "$STOPF" ] && { log "stop file, exit"; exit 3; }
  for i in $(seq 1 90); do
    [ "$(load)" = 0 ] && return 0
    [ $i = 1 ] && log "real traffic seen (load $(load)), waiting"
    sleep 10
  done
  log "traffic not cleared in 15 min, abort"; exit 2
}
client(){ # 压测客户端：HyperQwen 镜像自带的 vllm bench 已处理 SSE keep-alive
  sudo docker ps --format '{{.Names}}' | grep -qx hq-client && return
  sudo docker rm -f hq-client >/dev/null 2>&1
  sudo docker run -d --name hq-client --network host --entrypoint sleep \
    -v "$ROOT/models":/m:ro -v "$HOME/loadtest/results":/results "$IMAGE" infinity >/dev/null
}
vb(){ # 名字 端口 bench参数...（结果 $OUT/名字.json，已存在则跳过）
  local n=$1 p=$2; shift 2
  [ -f "$OUT/$n.json" ] && { log "skip $n"; return; }
  sudo docker exec hq-client /app/venv/bin/vllm bench serve --backend vllm --endpoint /v1/completions \
    --base-url http://127.0.0.1:$p --model $M --tokenizer $TOK \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "/results/${OUT#$HOME/loadtest/results/}" --result-filename "$n.json" "$@" > "$OUT/$n.log" 2>&1 \
    || log "FAIL $n"
}
summ(){ # 汇总 $OUT 下 bench json
  python3 - "$OUT" <<'EOF'
import json, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if "duration" not in d: continue
    if "median_ttft_ms" not in d:  # embedding / rerank：只有端到端延迟
        print(f"{os.path.basename(f)[:-5]:22s} ok {d['completed']}/{d['num_prompts']} req/s {d['request_throughput']:6.1f} "
              f"e2el p50/p99 {d.get('median_e2el_ms', 0):6.0f}/{d.get('p99_e2el_ms', 0):6.0f}")
        continue
    tot = (d["total_input_tokens"] + d["total_output_tokens"]) / d["duration"]
    print(f"{os.path.basename(f)[:-5]:22s} ok {d['completed']}/{d['num_prompts']} tot {tot:7.0f} out {d['output_throughput']:6.0f} "
          f"ttft p50/p99 {d['median_ttft_ms']:7.0f}/{d['p99_ttft_ms']:7.0f} tpot p50/p99 {d['median_tpot_ms']:5.0f}/{d['p99_tpot_ms']:5.0f}")
EOF
}
