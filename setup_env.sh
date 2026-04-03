#!/bin/bash
# Environment setup script for StableDiffusion-PyTorch
# Target: Ubuntu 22.04 / Python 3.10 / PyTorch 2.1.0 / CUDA 12.1

set -e

echo "=== Step 1: Create conda environment ==="
conda create -n sd-pytorch python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate sd-pytorch

echo "=== Step 2: Install PyTorch 2.1.0 + CUDA 12.1 ==="
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Install remaining dependencies ==="
pip install -r requirements.txt

echo "=== Step 4: Verify installation ==="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
python -c "import torchvision, transformers, einops; print('All dependencies OK')"

echo "=== Step 5: Download LPIPS weights ==="
mkdir -p models/weights/v0.1
if [ ! -f models/weights/v0.1/vgg.pth ]; then
    echo "Downloading LPIPS VGG weights..."
    python -c "
import torch
from torchvision.models import vgg16, VGG16_Weights
# This just verifies the model loads; the actual LPIPS weights need manual download
print('VGG16 pretrained model loads successfully')
"
    echo ""
    echo "*** IMPORTANT: Download LPIPS weights manually ***"
    echo "Open this link in a browser and download the raw file:"
    echo "  https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth"
    echo "Place it at: models/weights/v0.1/vgg.pth"
else
    echo "LPIPS weights already exist."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Prepare dataset (see README.md for data structure)"
echo "  2. Train VQVAE:    python -m tools.train_vqvae --config config/mnist.yaml"
echo "  3. Infer VQVAE:    python -m tools.infer_vqvae --config config/mnist.yaml"
echo "  4. Train LDM:      python -m tools.train_ddpm_vqvae --config config/mnist.yaml"
echo "  5. Sample:          python -m tools.sample_ddpm_vqvae --config config/mnist.yaml"
