#!/usr/bin/env python3
"""prefill 专项：固定输入长度、输出 4 token，测 TTFT 与 prefill 速度。

用法: prefill_probe.py <base_url> <served_model> <结果 jsonl> [--lens 2048,8192,32768,65536] [--conc 1,4]

- 提示词直接传随机 token id（/v1/completions 支持），长度精确，不依赖 tokenizer；
  每个请求的 id 都不同，不会命中 prefix cache。
- 不用 vllm bench serve：部分引擎（如 HyperQwen）在 30s 后发 SSE 注释 `: keep-alive`，
  bench 客户端会把这类请求误判为失败。这里跳过注释行，以第一个带 text 的数据块计 TTFT。
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request


def one(url, model, n_tok, seed):
    rnd = random.Random(seed)
    body = json.dumps({
        "model": model, "prompt": [rnd.randint(1000, 100000) for _ in range(n_tok)],
        "max_tokens": 4, "ignore_eos": True, "stream": True, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/completions", body, {"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            ch = json.loads(s[5:]).get("choices") or []
            if ttft is None and ch and ch[0].get("text"):
                ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0


def run(url, model, n_tok, conc, rounds, base_seed):
    res, lock = [], threading.Lock()

    def worker(w):
        for i in range(rounds):
            try:
                r = one(url, model, n_tok, base_seed * 1000 + w * rounds + i)
            except Exception as e:  # noqa: BLE001 记录失败，不中断整轮
                r = (None, repr(e))
            with lock:
                res.append(r)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r[0] for r in res if r[0] is not None]
    return {
        "input_len": n_tok, "concurrency": conc, "requests": len(res), "ok": len(ok),
        "errors": [r[1] for r in res if r[0] is None][:3],
        "ttft_p50_s": statistics.median(ok) if ok else None,
        "ttft_max_s": max(ok) if ok else None,
        "per_req_prefill_tok_s": n_tok / statistics.median(ok) if ok else None,
        "agg_prefill_tok_s": n_tok * len(ok) / wall if ok else None,
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--lens", default="2048,8192,32768,65536")
    ap.add_argument("--conc", default="1,4")
    ap.add_argument("--rounds", type=int, default=3, help="每个并发线程的请求数")
    a = ap.parse_args()
    # 每次运行换一组种子：预热与正式测试若用同一组 id，正式第一轮会命中 prefix cache
    salt = random.randrange(1 << 20) * 1000
    rows = []
    for n in map(int, a.lens.split(",")):
        for c in map(int, a.conc.split(",")):
            print(f"[{time.strftime('%T')}] in{n // 1024}k-c{c}", flush=True)
            r = run(a.url, a.model, n, c, a.rounds, salt + n + c)
            r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            rows.append(r)
            with open(a.out, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("| 输入 | 并发 | 成功 | TTFT P50 s | 单请求 prefill tok/s | 总 prefill tok/s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        f = lambda k, p=0: "-" if r[k] is None else f"{r[k]:.{p}f}"  # noqa: E731
        print(f"| {r['input_len'] // 1024}k | {r['concurrency']} | {r['ok']}/{r['requests']} "
              f"| {f('ttft_p50_s', 2)} | {f('per_req_prefill_tok_s')} | {f('agg_prefill_tok_s')} |")


if __name__ == "__main__":
    main()
