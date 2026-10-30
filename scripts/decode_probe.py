#!/usr/bin/env python3
"""decode 专项：真实提示词（可前置长文档），测每路 decode 速度与总输出吞吐。

用法: decode_probe.py <base_url> <served_model> <结果 jsonl> --prompts real_prompts.jsonl
        [--grid "0:1,16,64;32768:1,4"] [--max-tokens 512] [--rounds 2] [--corpus-dir DIR]

- grid 为「上下文 token 数:并发列表」，分号分隔；上下文 >0 时在提示词前拼接 corpus-dir 下的文本，
  按约 1.75 字符/token 截断，实际长度以返回的 usage.prompt_tokens 为准。
- 走 /v1/chat/completions（thinking 按模型默认），temperature 0.6 / top_p 0.95，
  每个请求开头带随机盐，不命中 prefix cache。
- 单路 decode tok/s = (completion_tokens - 1) / (E2E - TTFT)；TTFT 取第一个带 content 或 reasoning 的块。
- 跳过 SSE 注释行（HyperQwen 30s 后发 `: keep-alive`）。
"""
import argparse
import glob
import json
import os
import random
import statistics
import threading
import time
import urllib.request


def load_corpus(d):
    parts = []
    for ext in ("md", "sh", "py", "txt"):
        for f in sorted(glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)):
            try:
                parts.append(open(f, encoding="utf-8").read())
            except (OSError, UnicodeDecodeError):
                pass
    return "\n\n".join(parts) or "lorem ipsum " * 1000


def build_doc(corpus, n_tok):
    want = int(n_tok * 1.75)
    s = corpus
    while len(s) < want:
        s += "\n\n" + corpus
    return s[:want]


def one(url, model, messages, max_tokens):
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True}, "temperature": 0.6, "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage = None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            d = json.loads(s[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ttft is None and ch:
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    ttft = time.perf_counter() - t0
    e2e = time.perf_counter() - t0
    out = (usage or {}).get("completion_tokens") or 0
    dec = (out - 1) / (e2e - ttft) if ttft and out > 1 and e2e > ttft else None
    return {"ttft": ttft, "e2e": e2e, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": out, "decode_tok_s": dec}


def run(url, model, prompts, doc, conc, rounds, max_tokens):
    res, lock = [], threading.Lock()
    rnd = random.Random()

    def worker(w):
        for _ in range(rounds):
            p = rnd.choice(prompts)
            salt = f"[req {rnd.randrange(1 << 40):x}]\n"
            text = salt + (f"以下是参考资料：\n{doc}\n\n参考资料结束。\n\n" if doc else "") + p
            try:
                r = one(url, model, [{"role": "user", "content": text}], max_tokens)
            except Exception as e:  # noqa: BLE001 记录失败，不中断整轮
                r = {"error": repr(e)}
            with lock:
                res.append(r)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in res if r.get("decode_tok_s")]
    med = lambda k: statistics.median(r[k] for r in ok) if ok else None  # noqa: E731
    return {
        "concurrency": conc, "requests": len(res), "ok": len(ok),
        "errors": [r["error"] for r in res if "error" in r][:3],
        "prompt_tokens_p50": med("prompt_tokens"), "completion_tokens_p50": med("completion_tokens"),
        "ttft_p50_s": med("ttft"), "decode_tok_s_p50": med("decode_tok_s"),
        "decode_tok_s_min": min((r["decode_tok_s"] for r in ok), default=None),
        "total_out_tok_s": sum(r["completion_tokens"] for r in ok) / wall if ok else None,
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--grid", default="0:1,16,64;32768:1,4")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2, help="每个并发线程的请求数（并发 1 时至少 3）")
    ap.add_argument("--corpus-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts, encoding="utf-8") if l.strip()]
    corpus = load_corpus(a.corpus_dir)
    rows = []
    for cell in a.grid.split(";"):
        ctx, concs = cell.split(":")
        doc = build_doc(corpus, int(ctx)) if int(ctx) > 0 else ""
        for c in map(int, concs.split(",")):
            print(f"[{time.strftime('%T')}] ctx{int(ctx) // 1024}k-c{c}", flush=True)
            r = run(a.url, a.model, prompts, doc, c, max(a.rounds, 3) if c == 1 else a.rounds, a.max_tokens)
            r.update(ctx_nominal=int(ctx), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            rows.append(r)
            with open(a.out, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("| 上下文 | 并发 | 成功 | 输入 tok P50 | TTFT P50 s | 单路 decode tok/s P50 | 最慢一路 | 总输出 tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda k, p=0: "-" if r[k] is None else f"{r[k]:.{p}f}"  # noqa: E731
        print(f"| {r['ctx_nominal'] // 1024}k | {r['concurrency']} | {r['ok']}/{r['requests']} | {f('prompt_tokens_p50')} "
              f"| {f('ttft_p50_s', 2)} | {f('decode_tok_s_p50', 1)} | {f('decode_tok_s_min', 1)} | {f('total_out_tok_s')} |")


if __name__ == "__main__":
    main()
