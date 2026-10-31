#!/usr/bin/env bash
# 回滚：摘下 GPU0/1 两个 HyperQwen，恢复旧 a（GPU0,1 TP2 FP8，:8003），Router override 还原
set -u
L=$HOME/loadtest/swap-a-20261101.log
log(){ echo "[$(date +%T)] rollback: $*" | tee -a "$L"; }
for p in 8003 8006; do curl -s -m10 -X POST "127.0.0.1:8000/remove_worker?url=http://127.0.0.1:$p" >> "$L"; echo >> "$L"; done
for c in vllm-hq-g1 vllm-hq-g0; do
  sudo docker inspect "$c" >/dev/null 2>&1 || continue
  sudo docker update --restart=no "$c" >/dev/null 2>&1; sudo docker stop -t 60 "$c" >/dev/null
done
sudo docker update --restart=always vllm-tp2-a >/dev/null
sudo docker start vllm-tp2-a >/dev/null
for i in $(seq 1 60); do curl -sf -m3 127.0.0.1:8003/v1/models >/dev/null && break; sleep 10; done
O=/etc/systemd/system/vllm-router.service.d/override.conf
if [ -f $O.bak-20261101 ]; then
  sudo cp -a $O.bak-20261101 $O && sudo systemctl daemon-reload && sudo systemctl restart vllm-router
  sleep 3; curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8005" >> "$L"; echo >> "$L"
else
  curl -s -m10 -X POST "127.0.0.1:8000/add_worker?url=http://127.0.0.1:8003" >> "$L"; echo >> "$L"
fi
log "workers $(curl -s 127.0.0.1:8000/list_workers)"
