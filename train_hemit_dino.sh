#!/bin/bash
# =============================================================================
# HEMIT DINOv2+LDM Training (DDP, 2x RTX 4090)
# Usage: tmux new -s dino && bash train_hemit_dino.sh 2>&1 | tee train_dino.log
# =============================================================================
set -e

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

# Create task directory & symlink shared VQVAE assets from hemit_full
mkdir -p "${TASK_DIR}"

# Symlink VQVAE checkpoint if not already present
if [ ! -f "${TASK_DIR}/vqvae_autoencoder_ckpt.pth" ]; then
    if [ -f "hemit_full/vqvae_autoencoder_ckpt.pth" ]; then
        ln -sf "$(pwd)/hemit_full/vqvae_autoencoder_ckpt.pth" "${TASK_DIR}/vqvae_autoencoder_ckpt.pth"
        echo "Symlinked VQVAE checkpoint from hemit_full"
    else
        echo "ERROR: hemit_full/vqvae_autoencoder_ckpt.pth not found!"
        exit 1
    fi
fi

# Symlink latents directory if not already present
if [ ! -d "${LATENT_DIR}" ] || [ -z "$(ls -A ${LATENT_DIR} 2>/dev/null)" ]; then
    if [ -d "hemit_full/vqvae_latents" ] && [ -n "$(ls -A hemit_full/vqvae_latents 2>/dev/null)" ]; then
        rm -rf "${LATENT_DIR}"
        ln -sf "$(pwd)/hemit_full/vqvae_latents" "${LATENT_DIR}"
        echo "Symlinked VQVAE latents from hemit_full"
    else
        echo "ERROR: hemit_full/vqvae_latents not found or empty!"
        exit 1
    fi
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
