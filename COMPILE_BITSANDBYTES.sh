#!/bin/bash
# Compile bitsandbytes from source for CUDA 11.8
# This is the ONLY way to fix the libcusparse.so.11 error

set -e

echo "==========================================="
echo "COMPILE BITSANDBYTES FROM SOURCE"
echo "==========================================="
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating baseline_final environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Remove old bitsandbytes
echo "Step 1: Removing old bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || true
echo "  ✓ Old version removed"
echo ""

# Install build dependencies
echo "Step 2: Installing build dependencies..."
conda install -y cmake ninja
echo "  ✓ Build tools installed"
echo ""

# Clone bitsandbytes
echo "Step 3: Cloning bitsandbytes repository..."
cd /tmp
rm -rf bitsandbytes
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
echo "  ✓ Repository cloned"
echo ""

# Compile for CUDA 11.8
echo "Step 4: Compiling for CUDA 11.8..."
echo "  This may take 5-10 minutes..."
CUDA_VERSION=118 make cuda11x
echo "  ✓ Compilation complete"
echo ""

# Install
echo "Step 5: Installing compiled bitsandbytes..."
python setup.py install
echo "  ✓ Installed"
echo ""

# Verify
echo "Step 6: Verifying installation..."
cd ~
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python << 'EOF'
import bitsandbytes as bnb
print(f"  ✓ bitsandbytes: {bnb.__version__}")
print("  ✓ Compiled from source - should work now!")
EOF

echo ""
echo "==========================================="
echo "COMPILATION COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1"
echo "  git pull origin main"
echo "  bash START_TRAINING_NOW.sh"
echo ""
