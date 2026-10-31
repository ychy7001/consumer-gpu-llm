#!/usr/bin/env python3
"""OCR 探针：用 批处理客户端同款 OCR 提示词逐图请求，算 CER；另测 5 图单请求。仅标准库。
用法: ocr_probe.py <base_url> <model> <ocrset_dir> <out.json> [--conc 4] [--multi 5]
"""
import argparse, base64, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = ("<system_prompt>\nYou are an OCR assistant. Your task is to extract all body text content from this document "
          "image and output in pure Markdown format.\n</system_prompt>\n\n<instructions>\n1. Ignore headers and footers.\n"
          "2. Use Markdown table syntax for tables.\n3. Use LaTeX format for formulas (wrapped with $ or $$).\n"
          "4. Organize content in the original reading order.\n5. Output ONLY the extracted text content. Do NOT include "
          "any HTML tags, reasoning, or unrelated comments.\n6. If there is absolutely no recognizable text content in the "
          "image, reply ONLY with: No text content.\n</instructions>")
NORM = re.compile(r"[\s|*#>`\-:]+")


def norm(s):
    return NORM.sub("", s)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def call(url, model, images, prompt, timeout=600):
    parts = [{"type": "text", "text": prompt}]
    for p in images:
        mime = "image/jpeg" if p.suffix == ".jpg" else "image/png"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}", "detail": "auto"}})
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": parts}], "max_tokens": 5000, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        u = d.get("usage", {})
        return {"ok": True, "s": time.time() - t, "text": d["choices"][0]["message"].get("content") or "",
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "s": time.time() - t, "code": e.code, "err": e.read()[:300].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "s": time.time() - t, "err": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--conc", type=int, default=4); ap.add_argument("--multi", type=int, default=5)
    ap.add_argument("--manifest", default="manifest.jsonl")
    a = ap.parse_args()
    d = Path(a.dir)
    items = [json.loads(l) for l in open(d / a.manifest, encoding="utf-8")]
    t0 = time.time()
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda it: call(a.url, a.model, [d / it["file"]], PROMPT), items))
    wall = time.time() - t0
    rows = []
    for it, r in zip(items, res):
        row = {"file": it["file"], "kind": it["kind"], **{k: v for k, v in r.items() if k != "text"}}
        if r["ok"]:
            row["cer"] = round(cer(it["text"], r["text"]), 4)
        rows.append(row)
    allitems = [json.loads(l) for l in open(d / "manifest.jsonl", encoding="utf-8")]
    shots = [d / it["file"] for it in allitems if it["kind"] == "shot"][: a.multi]
    multi = call(a.url, a.model, shots, f"这里有 {len(shots)} 张截图。请依次输出每张图里表格的第二行（吞吐那一行），每张一行。")
    multi.pop("text", None)
    summ = {"wall_s": round(wall, 1), "ok": sum(r["ok"] for r in res), "n": len(res), "multi_ok": multi["ok"], "multi": multi}
    for k in ("scan", "shot", "photo", "all"):
        sel = [r for r in rows if r.get("cer") is not None and (k == "all" or r["kind"] == k)]
        if sel:
            summ[f"cer_{k}"] = round(sum(r["cer"] for r in sel) / len(sel), 4)
            summ[f"ptok_{k}"] = round(sum(r["prompt_tokens"] or 0 for r in sel) / len(sel))
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("OCR ok %d/%d wall %.0fs cer all/scan/shot/photo %s/%s/%s/%s ptok scan/shot/photo %s/%s/%s multi%d %s" % (
        summ["ok"], summ["n"], wall, summ.get("cer_all"), summ.get("cer_scan"), summ.get("cer_shot"), summ.get("cer_photo"),
        summ.get("ptok_scan"), summ.get("ptok_shot"), summ.get("ptok_photo"), len(shots),
        "ok" if multi["ok"] else f"FAIL {multi.get('code')} {multi.get('err', '')[:120]}"))


if __name__ == "__main__":
    sys.exit(main())
