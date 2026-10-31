#!/usr/bin/env bash
# 11-01：a（GPU0,1 TP2 FP8，:8003）换成两个单卡 HyperQwen batch，四卡全部单卡 DP
# GPU1 -> 127.0.0.1:8003（沿用端口），GPU0 -> 127.0.0.1:8006（GPU0 上有 embedding ~5.3G，GPU_UTIL 0.80）
# 旧 a 只停不删、restart=no；Router 静态列表改为 8003-8006（原 override 备份为 override.conf.bak-20261101）
# 回滚见 rollback-a-20261101.sh
set -u
L=$HOME/loadtest/swap-a-20261101.log
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }
IMAGE=ghcr.io/syv-ai/hyperqwen:latest
ROOT=/models/hyperqwen
M=Qwen3.8-27B-FP8
VIS_ARGS='--limit-mm-per-prompt {"image":{"count":5}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":8388608}}'

[ "$(sudo docker image inspect $IMAGE --format '{{.Id}}' | cut -c8-19)" = e69714eab977 ] || { log "image id mismatch"; exit 1; }
for g in 0 1; do [ -d "$ROOT/cache-gpu$g" ] || sudo cp -a "$ROOT/cache-gpu2" "$ROOT/cache-gpu$g"; done

curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
for i in $(seq 1 90); do
  r=$(curl -s -m5 127.0.0.1:8003/metrics | awk '/^vllm:num_requests_running\{/{print int($2)}')
  [[ ${r:-0} == 0 ]] && break; sleep 10
done
log "old a running=${r:-0} after $((i*10))s, stopping"
sudo docker update --restart=no vllm-tp2-a >/dev/null
sudo docker stop -t 120 vllm-tp2-a >/dev/null

start(){ # 名字 GPU 主机端口 GPU_UTIL
  sudo docker rm -f "$1" >/dev/null 2>&1
  sudo docker run -d --name "$1" --restart always --ipc host --gpus "\"device=$2\"" -p "127.0.0.1:$3:8000" \
    -v "$ROOT/models":/app/models -v "$ROOT/cache-gpu$2":/cache \
    -e HOME=/cache -e HOST=0.0.0.0 -e PORT=8000 -e PREPARE=0 -e VERIFY=0 \
    -e KV=fp8 -e INT8_ACT=int8 -e MAX_LEN=262144 -e GPU_UTIL="$4" -e PREFIX_CACHE=1 \
    -e VISION=1 -e EXTRA_ARGS="--served-model-name $M $VIS_ARGS" $IMAGE batch >/dev/null
  log "started $1 on GPU$2 -> :$3 util $4"
}
start vllm-hq-g1 1 8003 0.92
start vllm-hq-g0 0 8006 0.80

for p in 8003 8006; do
  for i in $(seq 1 60); do curl -sf -m3 127.0.0.1:$p/v1/models >/dev/null && break; sleep 10; done
  if ! curl -sf -m3 127.0.0.1:$p/v1/models >/dev/null; then log ":$p not ready after 10 min"; exit 1; fi
  curl -s 127.0.0.1:$p/v1/models | grep -q "\"$M\"" || { log ":$p wrong model name"; exit 1; }
  for n in 64 2000; do
    curl -s -m120 127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$(printf '测试%.0s' $(seq 1 $n))，回复好\"}],\"max_tokens\":16}" >/dev/null
  done
  if ! python3 "$HOME/loadtest/vision/ocr_probe.py" http://127.0.0.1:$p $M "$HOME/loadtest/vision/ocrset" "$HOME/loadtest/swap-a-ocr-$p.json" \
      --conc 2 --manifest manifest-smoke.jsonl 2>&1 | tail -1 | tee -a "$L" | grep -q "ok 2/2 .*multi5 ok"; then
    log ":$p vision smoke failed, not adding"; exit 1
  fi
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:$p" >> "$L"; echo >> "$L"
  log ":$p ready, warmed, added"
done
for c in vllm-hq-g1 vllm-hq-g0; do sudo docker logs $c 2>&1 | grep "GPU KV cache size" | tail -1 | cut -c60-200 | sed "s/^/$c /" | tee -a "$L"; done

# Router 静态列表补全四个端口（之前 8005 只是动态加的，Router 重启会丢）
O=/etc/systemd/system/vllm-router.service.d/override.conf
[ -f $O.bak-20261101 ] || sudo cp -a $O $O.bak-20261101
sudo sed -i 's#--worker-urls http://127.0.0.1:8003 http://127.0.0.1:8004 #--worker-urls http://127.0.0.1:8003 http://127.0.0.1:8004 http://127.0.0.1:8005 http://127.0.0.1:8006 #' $O
grep -q "8006" $O || { log "override edit failed, router not restarted"; exit 1; }
sudo systemctl daemon-reload && sudo systemctl restart vllm-router
for i in $(seq 1 30); do curl -sf -m3 127.0.0.1:8000/list_workers >/dev/null && break; sleep 2; done
log "workers $(curl -s 127.0.0.1:8000/list_workers)"
log "swap done"
