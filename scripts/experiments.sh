#!/usr/bin/env bash
# 依次部署多组 LLM 参数并跑 quick 套件；部署失败自动回滚原配置并继续下一组
# 用法: experiments.sh <实验列表文件>   每行: <tag>|<额外 vllm 参数>
set -u
D=$(cd "$(dirname "$0")" && pwd)
LIST=$1
LOG=$HOME/loadtest/results/experiments.log

# 等待其他套件结束，避免并发施压
while pgrep -f "scripts/suite.sh" >/dev/null; do sleep 20; done

while IFS='|' read -r tag extra; do
  [[ -z $tag || $tag == \#* ]] && continue
  if python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['completed']>0 else 1)" "$HOME/loadtest/results/$tag/longout.json" 2>/dev/null; then echo "skip $tag" >> "$LOG"; continue; fi
  echo "[$(date +%T)] === $tag : $extra" >> "$LOG"
  if "$D/deploy_llm.sh" "$extra" >> "$LOG" 2>&1; then
    "$D/suite.sh" "$tag" "${MODE:-quick}" >> "$LOG" 2>&1
  else
    echo "[$(date +%T)] $tag deploy FAILED, rollback" >> "$LOG"
    "$D/deploy_llm.sh" "--max-num-seqs 6" >> "$LOG" 2>&1
  fi
done < "$LIST"
echo "[$(date +%T)] batch done" >> "$LOG"
