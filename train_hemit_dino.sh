#!/bin/bash
# =============================================================================
# HEMIT DINOv2+LDM Training (DDP, 2x RTX 4090)
# Usage: bash train_hemit_dino.sh
# =============================================================================
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sd-pytorch

CONFIG="config/hemit_dino.yaml"
TASK_DIR="hemit_dino"
LATENT_DIR="${TASK_DIR}/vqvae_latents"
NGPU=2

echo "=============================================="
echo "  HEMIT DINOv2+LDM Training"
echo "  Config: ${CONFIG}"
echo "  GPUs:   ${NGPU}"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Verify required VQVAE assets exist
if [ ! -f "${TASK_DIR}/vqvae_autoencoder_ckpt.pth" ]; then
    echo "ERROR: ${TASK_DIR}/vqvae_autoencoder_ckpt.pth not found!"
    exit 1
fi
if [ ! -d "${LATENT_DIR}" ] || [ -z "$(ls -A ${LATENT_DIR} 2>/dev/null)" ]; then
    echo "ERROR: ${LATENT_DIR} not found or empty!"
    exit 1
fi
echo "VQVAE checkpoint: OK"
echo "VQVAE latents:    OK"

# Clean stale LDM artifacts from previous runs
LDM_CKPT="${TASK_DIR}/ddpm_ckpt_hemit_dino.pth"
TRAIN_LOG="${TASK_DIR}/train_log.csv"
if [ -f "${LDM_CKPT}" ]; then
    echo "Removing old LDM checkpoint: ${LDM_CKPT}"
    rm -f "${LDM_CKPT}"
fi
if [ -f "${TRAIN_LOG}" ]; then
    echo "Removing old train log: ${TRAIN_LOG}"
    rm -f "${TRAIN_LOG}"
fi

# -----------------------------------------------
# Train LDM with DDP
# -----------------------------------------------
echo ""
echo "Starting DDP training with ${NGPU} GPUs..."
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

torchrun --nproc_per_node=${NGPU} -m tools.train_ddpm_cond --config ${CONFIG}

echo ""
echo "=============================================="
echo "  Training complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# # 3. 训练完成后推理（自动使用 EMA 权重）
# python -m tools.sample_ddpm_hemit --config config/hemit_dino.yaml --full-image --num-samples 9

# # 4. 评估
# python -m tools.evaluate_hemit --config config/hemit_dino.yaml