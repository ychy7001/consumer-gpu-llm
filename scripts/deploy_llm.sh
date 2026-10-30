#!/usr/bin/env bash
# 以指定额外参数重建 LLM 后端 A/B（保持原端口、GPU、挂载、restart 策略）
# 用法: deploy_llm.sh "<额外 vllm 参数>"
#   基础参数与原部署一致，仅 max-num-seqs 等由 EXTRA 覆盖
# 回滚: deploy_llm.sh "--max-num-seqs 6"  即恢复原始配置
# ONLY=vllm-tp2-a deploy_llm.sh ...  只重建一个后端（另一个继续服务）
set -euo pipefail
EXTRA=${1:?需要额外参数，原始配置传 "--max-num-seqs 6"}
IMAGE=vllm/vllm-openai:v0.25.1
MODEL_DIR=${MODEL_DIR:-/models}
M=$MODEL_DIR/Qwen/Qwen3.8-27B-FP8
CACHE=$HOME/loadtest/vllm-cache   # 持久化 torch.compile 缓存，缩短重启时间
ONLY=${ONLY:-}
mkdir -p "$CACHE"

start() { # name gpus port
  [[ -n $ONLY && $ONLY != "$1" ]] && return 0
  # inductor/triton 缓存默认在容器可写层，重建即丢失（warmup 多约 70s）；首次改挂载前从旧容器拷出
  for d in /tmp/torchinductor_root:inductor /root/.triton/cache:triton; do
    [[ -d $CACHE/${d#*:} ]] || sudo docker cp "$1:${d%%:*}" "$CACHE/${d#*:}" 2>/dev/null || true
  done
  sudo docker rm -f "$1" >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  sudo docker run -d --name "$1" --restart always --ipc host \
    --gpus "\"device=$2\"" -p "127.0.0.1:$3:8000" \
    -v "$MODEL_DIR":"$MODEL_DIR" \
    -v "$CACHE":/root/.cache/vllm \
    -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/inductor \
    -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
    "$IMAGE" "$M" --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.82 --max-model-len 262144 \
    --tensor-parallel-size 2 --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --served-model-name Qwen3.8-27B-FP8 $EXTRA >/dev/null
}

echo "[$(date +%T)] deploy: $EXTRA"
start vllm-tp2-a 0,1 8003
start vllm-tp2-b 2,3 8004

# 等待两个后端就绪（最长 20 分钟）
for i in $(seq 1 120); do
  ok=0
  for p in 8003 8004; do curl -sf -m3 "127.0.0.1:$p/health" >/dev/null && ok=$((ok+1)); done
  if [[ $ok == 2 ]]; then
    echo "[$(date +%T)] backends ready after $((i*10))s"
    # Router 熔断/健康检查恢复需要时间，端到端连续 4 次成功才算就绪
    good=0
    for j in $(seq 1 60); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -m30 -H content-type:application/json \
        -d '{"model":"Qwen3.8-27B-FP8","prompt":"hi","max_tokens":2}' 127.0.0.1:8000/v1/completions)
      if [[ $code == 200 ]]; then good=$((good+1)); [[ $good -ge 4 ]] && { echo "[$(date +%T)] router ready"; exit 0; }
      else good=0; sleep 5; fi
    done
    echo "router not ready"; exit 1
  fi
  for c in vllm-tp2-a vllm-tp2-b; do
    [[ $(sudo docker inspect -f '{{.RestartCount}}' $c) -gt 0 ]] && { echo "$c crashed"; sudo docker logs --tail 30 $c; exit 1; }
  done
  sleep 10
done
echo "timeout"; exit 1
