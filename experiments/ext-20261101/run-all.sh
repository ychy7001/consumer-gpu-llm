#!/usr/bin/env bash
# 顺序执行 A（容量/混部/长文档）→ B（功耗）→ C（整机 profile 轮换）；任一步退出码非 0 则停止
d=$(dirname "$0")
bash $d/a-capacity.sh && bash $d/b-power.sh && bash $d/c-rotate.sh
echo "[$(date +%T)] chain exit $?" >> $HOME/loadtest/ext-20261101.chain.log
