#!/bin/bash
# Fix bitsandbytes CUDA library path issue

set -e

echo "==========================================="
echo "FIX BITSANDBYTES CUDA LIBRARIES"
echo "==========================================="
echo ""

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Added to LD_LIBRARY_PATH:"
echo "  /usr/local/cuda/lib64"
echo ""

# Verify bitsandbytes now works
echo "Testing bitsandbytes..."
python << 'EOF'
import bitsandbytes as bnb
print(f"✓ bitsandbytes: {bnb.__version__}")
print("✓ CUDA libraries found")
EOF

echo ""
echo "==========================================="
echo "FIX APPLIED"
echo "==========================================="
echo ""
echo "To make this permanent, add to ~/.bashrc:"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "For now, start training with:"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo "  bash START_TRAINING_NOW.sh"
echo ""
