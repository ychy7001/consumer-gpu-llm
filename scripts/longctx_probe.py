#!/usr/bin/env python3
"""长上下文质量探针（RULER 风格，标准库实现，OpenAI 兼容服务通用）。
真实英文文本（wikitext-2 test）做干草堆，两类任务：
  mk  多键检索：随机深度插入 8 条 "The access code for <key> is <7位数>."，问其中 4 个，按命中键数计分
  vt  变量追踪：一条 4 跳赋值链 + 2 条干扰链交错插入，问哪些变量等于目标值，按 Jaccard 计分，完全一致记 exact
同一 seed 下题目确定，可在不同部署间逐题对比。thinking 关闭，temperature 0。
用法: longctx_probe.py <base_url> <model> <haystack.txt> <out.json> [--lengths 32000,64000,128000,200000]
      [--trials 5] [--conc 2] [--seed 0]"""
import argparse, json, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("haystack"); ap.add_argument("out")
ap.add_argument("--lengths", default="32000,64000,128000,200000")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--conc", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chars-per-token", type=float, default=4.3)
args = ap.parse_args()

HAY = re.sub(r"\n{2,}", "\n", open(args.haystack, encoding="utf-8").read())
WORDS = ("amber birch cobalt delta ember falcon garnet harbor indigo juniper kestrel lumen meadow nickel "
         "onyx pepper quartz raven saffron tundra umber violet willow xenon yarrow zephyr").split()


def chunk(rng, n_chars):
    start = rng.randrange(len(HAY))
    s = (HAY[start:] + HAY)[:n_chars]
    return s[s.find("\n") + 1:]  # 从段落边界开始


def insert(text, lines, rng):
    """把 lines 插到 text 的随机段落边界，返回新文本与各行深度（0~1）。"""
    cuts = [m.start() for m in re.finditer("\n", text)] or [len(text)]
    picks = sorted(rng.sample(cuts, len(lines)))
    order = list(range(len(lines)))  # 保持 lines 原有先后顺序（vt 链依赖顺序）
    out, last, depth = [], 0, []
    for i, p in zip(order, picks):
        out += [text[last:p], "\n" + lines[i]]
        depth.append(round(p / len(text), 3)); last = p
    out.append(text[last:])
    return "".join(out), depth


def make_mk(rng, n_chars):
    keys = rng.sample([f"{a}-{b}" for a in WORDS for b in WORDS if a != b], 8)
    vals = [str(rng.randrange(1_000_000, 9_999_999)) for _ in keys]
    lines = [f"The access code for {k} is {v}." for k, v in zip(keys, vals)]
    perm = rng.sample(range(8), 8)
    text, depth = insert(chunk(rng, n_chars), [lines[i] for i in perm], rng)
    ask = rng.sample(range(8), 4)
    q = ("Several access codes are hidden in the text above. Give the access codes for: "
         + ", ".join(keys[i] for i in ask)
         + ". Answer with one line per key in the form '<key>: <code>' and nothing else.")
    return {"prompt": text + "\n\n" + q, "gold": {keys[i]: vals[i] for i in ask},
            "depth": [depth[perm.index(i)] for i in ask]}


def make_vt(rng, n_chars):
    names = rng.sample([w.upper()[:3] + str(d) for w in WORDS for d in range(10)], 15)
    vals = rng.sample(range(10000, 99999), 3)
    chains, lines = [], []
    for c in range(3):  # c=0 为目标链，其余为干扰链
        vs = names[c * 5:(c + 1) * 5]
        chains.append(vs)
        lines.append([f"VAR {vs[0]} = {vals[c]}"] + [f"VAR {vs[j]} = VAR {vs[j - 1]}" for j in range(1, 5)])
    # 三条链交错，各自保持顺序
    seq, idx = [], [0, 0, 0]
    while any(i < 5 for i in idx):
        c = rng.choice([k for k in range(3) if idx[k] < 5]); seq.append(lines[c][idx[c]]); idx[c] += 1
    text, depth = insert(chunk(rng, n_chars), seq, rng)
    q = (f"Some variable assignments are hidden in the text above. Find all variables that are assigned "
         f"the value {vals[0]}, directly or through a chain of assignments. "
         "Answer with the variable names separated by spaces and nothing else.")
    return {"prompt": text + "\n\n" + q, "gold": sorted(chains[0]), "depth": [min(depth), max(depth)]}


def post(body):
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def score(task, item, txt):
    if task == "mk":
        hit = sum(1 for k, v in item["gold"].items()
                  if re.search(re.escape(k) + r"\W{0,6}" + v + r"\b", txt))
        return hit / len(item["gold"])
    got, gold = set(re.findall(r"\b[A-Z]{3}\d\b", txt)), set(item["gold"])
    return len(got & gold) / len(got | gold) if got else 0.0  # Jaccard；完全一致另记 exact


def run(job):
    task, L, t, item = job
    t0 = time.perf_counter()
    try:
        r = post({"model": args.model, "max_tokens": 160, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": item["prompt"]}]})
        txt = r["choices"][0]["message"]["content"] or ""
        ptok, err = r.get("usage", {}).get("prompt_tokens"), None
    except Exception as e:
        txt, ptok, err = "", None, repr(e)[:200]
    return {"task": task, "len": L, "trial": t, "prompt_tokens": ptok, "score": score(task, item, txt),
            "depth": item["depth"], "gold": item["gold"], "answer": txt[:300], "err": err,
            "sec": round(time.perf_counter() - t0, 1)}


jobs = []
for L in map(int, args.lengths.split(",")):
    for t in range(args.trials):
        for task, mk in (("mk", make_mk), ("vt", make_vt)):
            rng = random.Random(f"{args.seed}-{task}-{L}-{t}")
            jobs.append((task, L, t, mk(rng, int(L * args.chars_per_token))))
with ThreadPoolExecutor(args.conc) as ex:
    rows = list(ex.map(run, jobs))

summary = {}
for r in rows:
    s = summary.setdefault(f"{r['task']}@{r['len'] // 1000}k", {"n": 0, "score": 0.0, "exact": 0, "err": 0, "tok": []})
    s["n"] += 1; s["score"] += r["score"]; s["exact"] += r["score"] == 1.0; s["err"] += bool(r["err"])
    if r["prompt_tokens"]: s["tok"].append(r["prompt_tokens"])
for k, s in summary.items():
    s["score"] = round(s["score"] / s["n"], 3); s["tok"] = max(s["tok"]) if s["tok"] else None
json.dump({"model": args.model, "url": args.url, "summary": summary, "rows": rows},
          open(args.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary))
