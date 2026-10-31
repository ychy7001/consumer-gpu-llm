#!/usr/bin/env bash
# 两张卡（4090 / 3090）共用的单实例测试套件，被 card-*.sh source。
# 调用方需设置：URL M OUT S(探针脚本目录) DATA(real_prompts/gsm200) CORPUS OCRSET L 与 bench_run(名字 并发 条数)
# 每组：prefill 2k/16k/64k/120k，decode 短上下文 1/4/16 与 32k 上下文 1/4，批处理 c8/c24，GSM8K-200，OCR 12 张 + 5 图
BATCH_ARGS="--dataset-name random --ignore-eos --random-range-ratio 0.1 --random-input-len 2048 --random-output-len 160"
MM5='{"image":{"count":5}}'
PX8K='{"size":{"shortest_edge":65536,"longest_edge":8388608}}'   # 单图 8192 token
VIS_ARGS="--limit-mm-per-prompt $MM5 --mm-processor-kwargs $PX8K"
log(){ echo "[$(date +%T)] $*" | tee -a "$L"; }

suite(){ # 名字
  local d=$OUT/$1; mkdir -p "$d"
  python3 "$S/prefill_probe.py" $URL $M /tmp/card-warm.jsonl --lens 2048 --conc 1 --rounds 1 >/dev/null 2>&1
  python3 "$S/prefill_probe.py" $URL $M "$d/prefill.jsonl" --lens 2048,16384,65536,120000 --conc 1 --rounds 2 >>"$L" 2>&1
  python3 "$S/decode_probe.py" $URL $M "$d/decode.jsonl" --prompts "$DATA/real_prompts.jsonl" --corpus-dir "$CORPUS" \
    --grid "0:1,4,16;32768:1,4" --max-tokens 512 --rounds 2 >>"$L" 2>&1
  bench_run "$1" 8 48
  bench_run "$1" 24 144
  python3 "$S/gsm_probe.py" $URL $M "$DATA/gsm200.jsonl" "$d/gsm.json" --conc 16 >>"$L" 2>&1
  python3 "$S/ocr_probe.py" $URL $M "$OCRSET" "$d/ocr.json" --conc 4 --manifest manifest12.jsonl 2>&1 | tail -1 >>"$L"
  python3 "$S/card_summary.py" "$d" | tee -a "$L"
}
