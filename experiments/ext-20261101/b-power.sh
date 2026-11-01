#!/usr/bin/env bash
# 11-01 扩展 B：功耗上限 450 / 350 / 300 / 250 W（四卡同时设置），退出时无论成败恢复 450 W
# 每档：整机批处理 chat c96（经 Router）、GPU3 单路 decode、GPU1 16k 单请求 prefill；1 s 采样功耗与 SM 频率
set -u
source "$(dirname "$0")/lib.sh"
OUT=$HOME/loadtest/results/20261101-ext-power; mkdir -p "$OUT"; L=$OUT/run.log
CHAT="--dataset-name random --ignore-eos --random-range-ratio 0.3 --random-input-len 1024 --random-output-len 256"
restore(){ sudo nvidia-smi -pl 450 >/dev/null; log "power restored: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"; }
trap restore EXIT
client
for w in 450 350 300 250; do
  waitidle
  sudo nvidia-smi -pl $w >/dev/null || { log "set $w failed"; exit 1; }
  log "limit $w: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"
  nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/smi-$w.csv" &
  SMI=$!
  vb chat-c96-$w 8000 $CHAT --max-concurrency 96 --num-prompts 768
  python3 "$S/decode_probe.py" http://127.0.0.1:8004 $M "$OUT/decode-$w.jsonl" --prompts "$T/data/real_prompts.jsonl" \
    --corpus-dir "$T/corpus" --grid "0:1" --max-tokens 512 --rounds 3 >>"$L" 2>&1
  python3 "$S/prefill_probe.py" http://127.0.0.1:8003 $M "$OUT/prefill-$w.jsonl" --lens 16384 --conc 1 --rounds 3 >>"$L" 2>&1
  kill $SMI
  log "limit $w done"
done
summ | tee -a "$L"
python3 - "$OUT" <<'EOF' | tee -a "$L"
import csv, glob, json, os, sys
o = sys.argv[1]
for w in (450, 350, 300, 250):
    f = f"{o}/smi-{w}.csv"
    if not os.path.exists(f): continue
    rows = [r for r in csv.reader(open(f)) if len(r) >= 6 and int(r[5].strip().split()[0]) > 50]
    p = [float(r[2].split()[0]) for r in rows]; c = [float(r[3].split()[0]) for r in rows]
    dec = [json.loads(l) for l in open(f"{o}/decode-{w}.jsonl")] if os.path.exists(f"{o}/decode-{w}.jsonl") else []
    pre = [json.loads(l) for l in open(f"{o}/prefill-{w}.jsonl")] if os.path.exists(f"{o}/prefill-{w}.jsonl") else []
    print(f"{w}W busy-samples {len(p)} avg power/GPU {sum(p)/max(len(p),1):.0f} W, avg SM {sum(c)/max(len(c),1):.0f} MHz | "
          f"decode {dec[-1].get('decode_tok_s_p50') if dec else None} | prefill16k {pre[-1].get('per_req_prefill_tok_s') if pre else None}")
EOF
log "B done"
