#!/usr/bin/env bash
# 兜底：到指定时间仍未写入 FINALIZED 标记，则回滚到原始部署
# 用法: failsafe.sh "2026-09-29 07:00"
set -u
D=$(cd "$(dirname "$0")" && pwd)
DEADLINE=$(date -d "$1" +%s)
while (( $(date +%s) < DEADLINE )); do sleep 60; done
if [[ ! -f $HOME/loadtest/FINALIZED ]]; then
  pkill -f scripts/experiments.sh; pkill -f scripts/suite.sh
  echo "[$(date +%T)] failsafe rollback" >> "$HOME/loadtest/results/experiments.log"
  "$D/deploy_llm.sh" "--max-num-seqs 6" >> "$HOME/loadtest/results/experiments.log" 2>&1
fi
