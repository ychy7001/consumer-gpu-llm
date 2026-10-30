#!/usr/bin/env python3
"""汇总一个或多个结果目录下的 bench JSON，输出 markdown 表格。

用法: summarize.py <结果目录>... [--prefix sweep]
多个目录时按场景名对齐，便于对比不同部署配置。
"""
import argparse
import glob
import json
import os

COLS = [
    ("completed", "ok", "{:.0f}"),
    ("request_throughput", "req/s", "{:.2f}"),
    ("output_throughput", "out tok/s", "{:.0f}"),
    ("total_token_throughput", "total tok/s", "{:.0f}"),
    ("median_ttft_ms", "TTFT p50", "{:.0f}"),
    ("p99_ttft_ms", "TTFT p99", "{:.0f}"),
    ("median_tpot_ms", "TPOT p50", "{:.1f}"),
    ("p99_tpot_ms", "TPOT p99", "{:.1f}"),
    ("median_e2el_ms", "E2E p50", "{:.0f}"),
    ("p99_e2el_ms", "E2E p99", "{:.0f}"),
]


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            out[os.path.basename(f)[:-5]] = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    data = {os.path.basename(d.rstrip("/")): load(d) for d in a.dirs}
    names = sorted({n for v in data.values() for n in v if n.startswith(a.prefix)},
                   key=lambda n: (n.rstrip("0123456789"), int("".join(c for c in n if c.isdigit()) or 0)))
    print("| 场景 | 配置 | " + " | ".join(c[1] for c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for n in names:
        for tag, v in data.items():
            if n not in v:
                continue
            r = v[n]
            cells = [fmt.format(r[k]) if isinstance(r.get(k), (int, float)) else "-" for k, _, fmt in COLS]
            print(f"| {n} | {tag} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
