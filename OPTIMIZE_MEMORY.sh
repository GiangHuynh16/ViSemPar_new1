#!/bin/bash
# Memory Optimization Script for Baseline 7B Training
# Run this BEFORE starting training to maximize available GPU memory

set -e

echo "=========================================="
echo "MEMORY OPTIMIZATION FOR BASELINE 7B"
echo "=========================================="
echo ""

# Step 1: Check current GPU usage
echo "Step 1: Checking GPU status..."
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# Step 2: Kill any existing Python processes (except this one)
echo "Step 2: Cleaning up old Python processes..."
echo ""
pkill -9 -f python || echo "  No Python processes to kill"
sleep 2
echo ""

# Step 3: Clear PyTorch cache
echo "Step 3: Clearing PyTorch cache..."
python3 << 'EOF'
import torch
import gc

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  ✓ Cleared CUDA cache")
    print(f"  ✓ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  ✓ GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("  ⚠️  CUDA not available")

gc.collect()
print(f"  ✓ Python garbage collector run")
EOF
echo ""

# Step 4: Set optimal CUDA memory allocation settings
echo "Step 4: Setting CUDA memory allocation optimizations..."
echo ""
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
echo "  ✓ Set PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Step 5: Disable unnecessary parallelism
echo "Step 5: Disabling unnecessary parallelism..."
echo ""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
echo "  ✓ Set TOKENIZERS_PARALLELISM=false"
echo "  ✓ Set OMP_NUM_THREADS=4"
echo ""

# Step 6: Clear system cache (requires sudo)
echo "Step 6: Clearing system cache (optional - requires sudo)..."
echo ""
if command -v sync &> /dev/null && [ "$EUID" -eq 0 ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
    echo "  ✓ System cache cleared"
else
    echo "  ⚠️  Skipped (run with sudo for system cache clearing)"
fi
echo ""

# Step 7: Verify final GPU state
echo "Step 7: Final GPU status..."
echo ""
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo ""
echo "  Free GPU memory: ${FREE_MEM} MB"
echo ""

# Step 8: Check if we have enough memory
echo "Step 8: Memory requirement check..."
echo ""
REQUIRED_MB=18000  # 18GB minimum for 7B training
if [ "$FREE_MEM" -ge "$REQUIRED_MB" ]; then
    echo "  ✅ PASS: ${FREE_MEM} MB >= ${REQUIRED_MB} MB (required)"
    echo ""
    echo "=========================================="
    echo "✅ OPTIMIZATION COMPLETE - READY TO TRAIN"
    echo "=========================================="
    echo ""
    echo "Run training with:"
    echo "  bash START_BASELINE_7B_TRAINING.sh"
    echo ""
    echo "Or with optimized environment variables:"
    echo "  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \\"
    echo "  TOKENIZERS_PARALLELISM=false \\"
    echo "  python train_baseline.py --epochs 15"
    echo ""
else
    echo "  ❌ FAIL: ${FREE_MEM} MB < ${REQUIRED_MB} MB (required)"
    echo ""
    echo "=========================================="
    echo "⚠️  INSUFFICIENT MEMORY"
    echo "=========================================="
    echo ""
    echo "Options:"
    echo "  1. Check for other GPU processes: nvidia-smi"
    echo "  2. Reboot server to clear all memory"
    echo "  3. Use smaller max_seq_length (already optimized to 1536)"
    echo ""
fi

# Export environment variables for current session
cat << 'ENVEOF'

To apply these optimizations to your current shell, run:
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=4

Or source this script:
  source OPTIMIZE_MEMORY.sh
ENVEOF
