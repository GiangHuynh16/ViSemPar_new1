#!/bin/bash
# Install correct packages for A6000 GPU (48GB VRAM)
# Run this on the server

set -e

echo "=========================================="
echo "INSTALL PACKAGES FOR A6000 GPU"
echo "=========================================="
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating lora_py310 environment..."
    eval "$(conda shell.bash hook)"
    conda activate lora_py310
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Check CUDA version
echo "Step 1: Checking CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "  CUDA Version: $CUDA_VERSION"
echo ""

# Uninstall old packages
echo "Step 2: Uninstalling old packages..."
pip uninstall torch torchvision torchaudio transformers peft accelerate bitsandbytes triton -y 2>/dev/null || true
echo "  ✓ Old packages removed"
echo ""

# Install PyTorch 2.1.0 with CUDA 12.1
echo "Step 3: Installing PyTorch 2.1.0 with CUDA 12.1..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
echo "  ✓ PyTorch installed"
echo ""

# Install compatible versions
echo "Step 4: Installing Transformers, PEFT, and dependencies..."
pip install transformers==4.36.0
pip install peft==0.6.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.1
pip install triton==2.0.0
echo "  ✓ Core packages installed"
echo ""

# Install other dependencies
echo "Step 5: Installing other dependencies..."
pip install datasets==2.14.5
pip install smatch==1.0.4
pip install penman==1.3.0
pip install tqdm
echo "  ✓ Other packages installed"
echo ""

# Verify installation
echo "Step 6: Verifying installation..."
python << 'EOF'
import sys
print("\nPackage Versions:")
print("=" * 50)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU 0 VRAM: {total_mem:.1f} GB")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"✓ Accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"✗ Accelerate: {e}")

try:
    import bitsandbytes
    print(f"✓ Bitsandbytes: {bitsandbytes.__version__}")
except Exception as e:
    print(f"✗ Bitsandbytes: {e}")

try:
    import triton
    print(f"✓ Triton: {triton.__version__}")
except Exception as e:
    print(f"✗ Triton: {e}")

print("=" * 50)
print("\n✓ All critical packages installed successfully!")
print(f"\nWith {total_mem:.0f}GB VRAM, you can use FULL MTUP config:")
print("  MAX_SEQ_LENGTH = 2048")
print("  batch_size = 2")
print("  gradient_accumulation = 8")
EOF

echo ""
echo "=========================================="
echo "INSTALLATION COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update config for 48GB GPU (optional but recommended):"
echo "   nano config/config.py"
echo ""
echo "   Change to:"
echo "   MAX_SEQ_LENGTH = 2048  # Full MTUP config"
echo "   \"per_device_train_batch_size\": 2"
echo "   \"gradient_accumulation_steps\": 8"
echo ""
echo "2. Start training:"
echo "   bash VERIFY_AND_START.sh"
echo ""
