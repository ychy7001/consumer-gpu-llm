#!/usr/bin/env python3
"""汇总 cardsuite 一组结果为一行（或多目录为表格）。用法: card_summary.py <组目录...>"""
import json, os, sys


def rows(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def one(d):
    p = sorted(rows(f"{d}/prefill.jsonl"), key=lambda r: r["input_len"])
    pre = "/".join(f"{r['input_len'] // 1000}k:{r['per_req_prefill_tok_s']:.0f}" for r in p)
    dec = {(r.get("ctx_nominal", 0), r["concurrency"]): r for r in rows(f"{d}/decode.jsonl")}
    def dv(k):
        r = dec.get(k)
        return f"{r['decode_tok_s_p50']:.0f}({r['total_out_tok_s']:.0f})" if r else "-"
    decs = " ".join(dv(k) for k in ((0, 1), (0, 4), (0, 16), (32768, 1), (32768, 4)))
    bt = []
    for c in (8, 24):
        b = load(f"{d}/batch-c{c}.json")
        bt.append(f"{b['total_token_throughput']:.0f}/p99 {b['p99_ttft_ms'] / 1000:.1f}s" if b else "-")
    g = load(f"{d}/gsm.json"); o = load(f"{d}/ocr.json")
    gs = f"{g['acc'] * 100:.1f}%" if g else "-"
    os_ = (f"cer {o['summary']['cer_all'] * 100:.2f}% multi5 {'ok' if o['summary'].get('multi_ok') else 'FAIL'}"
           if o else "-")
    kv = open(f"{d}/kv.txt").read().strip() if os.path.exists(f"{d}/kv.txt") else "-"
    return (f"{os.path.basename(d.rstrip('/'))} | prefill {pre} | decode 1/4/16 32k:1/4 {decs} "
            f"| batch c8 {bt[0]} c24 {bt[1]} | gsm {gs} | ocr {os_} | kv {kv}")


for d in sys.argv[1:]:
    print(one(d))
