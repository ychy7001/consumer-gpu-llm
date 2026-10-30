#!/usr/bin/env python3
"""GSM8K 快速质量探针：提示词与判分沿用 HyperQwen bench/quality_battery.py 的 run_gsm，只依赖标准库，
vLLM / SGLang 等 OpenAI 兼容服务都能直接测。
用法: gsm_probe.py <base_url> <model> <gsm.jsonl> <out.json> [--conc 32]
  gsm.jsonl 每行 {"q": 题目, "a": 原始答案(含 ####)}"""
import argparse, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--conc", type=int, default=32)
args = ap.parse_args()


def post(body):
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def extract_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "") if m else None


def one(row):
    gold = row["a"].split("####")[-1].strip().replace(",", "")
    try:
        r = post({"model": args.model, "max_tokens": 768, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": row["q"] + "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}]})
    except Exception as e:
        return False, 0, repr(e)
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = m.group(1).replace(",", "") if m else extract_num(txt)
    try:
        ok = abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        ok = False
    return ok, r["usage"]["completion_tokens"], None


rows = [json.loads(l) for l in open(args.data)]
t0 = time.perf_counter()
with ThreadPoolExecutor(args.conc) as ex:
    res = list(ex.map(one, rows))
errs = [e for _, _, e in res if e]
out = {"n": len(rows), "acc": sum(ok for ok, _, _ in res) / len(rows),
       "mean_tokens": sum(n for _, n, _ in res) / len(rows), "errors": len(errs), "secs": round(time.perf_counter() - t0)}
json.dump(out, open(args.out, "w"))
print(f"GSM8K n={out['n']} acc={out['acc']:.3f} mean_tokens={out['mean_tokens']:.0f} errors={out['errors']} {out['secs']}s")
