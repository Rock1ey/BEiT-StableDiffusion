#!/bin/bash
# =============================================================================
# Phase 1: HEMIT Full Dataset Training (Baseline)
# 用法: tmux new -s hemit && bash train_hemit_phase1.sh 2>&1 | tee train_phase1.log
# =============================================================================
set -e

CONFIG="config/hemit_full.yaml"
TASK_DIR="hemit_full"
LATENT_DIR="${TASK_DIR}/vqvae_latents"

echo "=============================================="
echo "  HEMIT Phase 1: Full Dataset Baseline"
echo "  Config: ${CONFIG}"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Create task directory
mkdir -p "${TASK_DIR}"

# -----------------------------------------------
# Step 1/3: Train VQVAE autoencoder
# -----------------------------------------------
echo ""
echo "[Step 1/3] Training VQVAE autoencoder..."
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

python -m tools.train_vqvae --config ${CONFIG}

echo "[Step 1/3] VQVAE training complete: $(date '+%Y-%m-%d %H:%M:%S')"

# -----------------------------------------------
# Step 2/3: Cache latents (grid patches -> VQVAE latents)
# -----------------------------------------------
echo ""
echo "[Step 2/3] Caching VQVAE latents for all grid patches..."
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

# Clean old latents if any
if [ -d "${LATENT_DIR}" ] && [ "$(ls -A ${LATENT_DIR} 2>/dev/null)" ]; then
    echo "Removing existing latents in ${LATENT_DIR}..."
    rm -f ${LATENT_DIR}/*.pkl
fi

python -m tools.infer_vqvae --config ${CONFIG}

echo "[Step 2/3] Latent caching complete: $(date '+%Y-%m-%d %H:%M:%S')"

# -----------------------------------------------
# Step 3/3: Train LDM (conditional diffusion model)
# -----------------------------------------------
echo ""
echo "[Step 3/3] Training LDM with image conditioning..."
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

python -m tools.train_ddpm_cond --config ${CONFIG}

echo "[Step 3/3] LDM training complete: $(date '+%Y-%m-%d %H:%M:%S')"

# -----------------------------------------------
# Done
# -----------------------------------------------
echo ""
echo "=============================================="
echo "  Phase 1 Complete!"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Check VQVAE reconstruction: ${TASK_DIR}/reconstructed_samples.png"
echo "  2. Sample single patches:"
echo "     python -m tools.sample_ddpm_hemit --config ${CONFIG}"
echo "  3. Full-resolution stitched inference:"
echo "     python -m tools.sample_ddpm_hemit --config ${CONFIG} --full-image --stride 192"

# 1. 全图推理（滑动窗口拼接）
python -m tools.sample_ddpm_hemit --config config/hemit_full.yaml --full-image --stride 192

# 2. 评估
python -m tools.evaluate_hemit --config config/hemit_full.yaml

# 3. 训练 LDM（DINOv2 + image dual conditioning）
python -m tools.train_ddpm_cond --config config/hemit_dino.yaml