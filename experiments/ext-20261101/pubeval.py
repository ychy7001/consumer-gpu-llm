#!/usr/bin/env python3
"""公开测试集质量对比（evalscope，OpenAI 兼容接口，直连单个实例不走 Router）。
用法: pubeval.py <base_url> <名字> <输出目录> [--limit N] [--only gsm8k,ifeval]
thinking 关闭、temperature 0；各数据集抽样规模固定，三种部署同题对比。"""
import argparse
from evalscope import run_task
from evalscope.config import TaskConfig

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("name"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=None, help="冒烟用：每个子集只取 N 条")
ap.add_argument("--only", default=None)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--tag", default="", help="结果子目录后缀，如 -8k")
a = ap.parse_args()

# 规模：GSM8K 全量 1319；MMLU-Pro 14 科各 100；C-Eval 52 科全量（val 1346）；IFEval 全量 541；
# LongBench v2 short 180（约 1.5 万–25 万 token，超过 262k 的题三种部署同样失败）；DocVQA val 前 500
LIMITS = {"gsm8k": None, "mmlu_pro": 100, "ceval": None, "ifeval": None, "longbench_v2": None, "docvqa": 500}
ARGS = {"longbench_v2": {"subset_list": ["short"]}}
ds = a.only.split(",") if a.only else list(LIMITS)

for d in ds:  # 逐个跑，单个数据集出错不影响其他
    lim = a.limit if a.limit is not None else LIMITS[d]
    cfg = TaskConfig(
        model="Qwen3.8-27B-FP8", api_url=a.url.rstrip("/") + "/v1", api_key="EMPTY", eval_type="openai_api",
        datasets=[d], dataset_args={d: ARGS.get(d, {})}, limit=lim,
        eval_batch_size=4 if d == "longbench_v2" else a.batch,
        generation_config={"temperature": 0, "max_tokens": a.max_tokens, "timeout": 3600,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        work_dir=f"{a.out}/{a.name}/{d}{a.tag}",no_timestamp=True,
    )
    try:
        run_task(task_cfg=cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[pubeval] {a.name} {d} failed: {e!r}"[:500], flush=True)
