#!/bin/bash
# 扫描不同cf_guidance_scale，快速推理5/10张，结果按scale命名

CONFIG="config/hemit_dino.yaml"
N=5  # 或10，快速测试
PATCHES_PER_GPU=8
DDIM_STEPS=1000

for SCALE in 1.0 1.5 2.0 2.5 3.0
  do
    OUTDIR="scan_cf_guidance/cfg_${SCALE}"
    mkdir -p "$OUTDIR"
    echo "\n==== cf_guidance_scale=$SCALE ===="
    python -m tools.sample_ddpm_hemit \
      --config $CONFIG \
      --full-image \
      --num-samples $N \
      --patches-per-gpu $PATCHES_PER_GPU \
      --ddim-steps $DDIM_STEPS \
      --cf-guidance-scale $SCALE \
      --out-dir $OUTDIR
  done

echo "全部推理完成，结果保存在 scan_cf_guidance/ 下，各子文件夹对应不同cfg"