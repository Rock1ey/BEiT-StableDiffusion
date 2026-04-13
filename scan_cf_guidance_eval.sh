#!/bin/bash
# 批量评估 scan_cf_guidance 下所有 cfg_* 结果，并生成汇总 CSV

CONFIG="config/hemit_dino.yaml"
OUT_ROOT="scan_cf_guidance"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"

echo "scale,ssim,psnr,ms_ssim,lpips" > "$SUMMARY_CSV"

for OUTDIR in ${OUT_ROOT}/cfg_*
do
  if [ -d "$OUTDIR" ]; then
    echo "评估 $OUTDIR ..."
    # 调用 evaluate_hemit：传入 config + gen-dir，脚本会从 config 中自动查找 gt-dir
    python -m tools.evaluate_hemit --config $CONFIG --gen-dir "$OUTDIR"

    METRICS_CSV="$OUTDIR/metrics.csv"
    if [ -f "$METRICS_CSV" ]; then
      # 取最后一行 mean,ssim,psnr,ms_ssim,lpips
      mean_line=$(tail -n 1 "$METRICS_CSV")
      # 解析 CSV（assume format: mean,ssim,psnr,ms_ssim,lpips）
      ssim=$(echo "$mean_line" | awk -F',' '{print $2}')
      psnr=$(echo "$mean_line" | awk -F',' '{print $3}')
      ms_ssim=$(echo "$mean_line" | awk -F',' '{print $4}')
      lpips=$(echo "$mean_line" | awk -F',' '{print $5}')
      scale_name=$(basename "$OUTDIR")
      scale_val=$(echo "$scale_name" | sed 's/^cfg_//')
      echo "${scale_val},${ssim},${psnr},${ms_ssim},${lpips}" >> "$SUMMARY_CSV"
      echo "Saved metrics to $METRICS_CSV and appended to $SUMMARY_CSV"
    else
      echo "Warning: metrics.csv not found in $OUTDIR"
    fi
  fi
done

echo "All evaluations complete. Summary: $SUMMARY_CSV"