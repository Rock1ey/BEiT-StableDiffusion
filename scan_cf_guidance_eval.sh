#!/bin/bash
# 批量评估 scan_cf_guidance 下所有 cfg 结果，csv 自动保存在各自目录

CONFIG="config/hemit_dino.yaml"

for OUTDIR in scan_cf_guidance/cfg_*
do
  if [ -d "$OUTDIR" ]; then
    echo "评估 $OUTDIR ..."
    python -m tools.evaluate_hemit \
      --config $CONFIG \
      --result-dir $OUTDIR
    echo "评估完成，结果已保存到 $OUTDIR/eval_metrics.csv"
  fi
done

echo "全部评估完成。"