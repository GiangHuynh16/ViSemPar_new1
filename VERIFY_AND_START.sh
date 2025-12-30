#!/bin/bash
# Comprehensive verification and training startup script
# Run this on server to ensure everything is correctly configured

set -e

echo "=========================================="
echo "BASELINE 7B TRAINING - VERIFICATION"
echo "=========================================="
echo ""

# Step 1: Pull latest code
echo "Step 1: Pulling latest code from git..."
git pull origin main
echo "  ✓ Code updated"
echo ""

# Step 2: Verify MAX_SEQ_LENGTH is 512
echo "Step 2: Verifying MAX_SEQ_LENGTH configuration..."
SEQ_LENGTH=$(grep "^MAX_SEQ_LENGTH = " config/config.py | awk '{print $3}')
echo "  Found: MAX_SEQ_LENGTH = $SEQ_LENGTH"
if [ "$SEQ_LENGTH" != "512" ]; then
    echo "  ✗ ERROR: MAX_SEQ_LENGTH should be 512, but found $SEQ_LENGTH"
    exit 1
fi
echo "  ✓ MAX_SEQ_LENGTH is correctly set to 512"
echo ""

# Step 3: Verify batch_size is 1
echo "Step 3: Verifying batch_size configuration..."
BATCH_SIZE=$(grep "per_device_train_batch_size" config/config.py | grep -oP '\d+')
echo "  Found: per_device_train_batch_size = $BATCH_SIZE"
if [ "$BATCH_SIZE" != "1" ]; then
    echo "  ✗ ERROR: batch_size should be 1, but found $BATCH_SIZE"
    exit 1
fi
echo "  ✓ Batch size is correctly set to 1"
echo ""

# Step 4: Verify max_memory is 14GB
echo "Step 4: Verifying max_memory configuration..."
MAX_MEM=$(grep "max_memory={0:" train_baseline.py | grep -oP '\d+GB' | head -1)
echo "  Found: max_memory GPU limit = $MAX_MEM"
if [ "$MAX_MEM" != "14GB" ]; then
    echo "  ✗ ERROR: max_memory should be 14GB, but found $MAX_MEM"
    exit 1
fi
echo "  ✓ max_memory is correctly set to 14GB GPU limit"
echo ""

# Step 5: Clear Python cache
echo "Step 5: Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Python cache cleared"
echo ""

# Step 6: Clear GPU memory
echo "Step 6: Clearing GPU memory..."
pkill -9 python 2>/dev/null || echo "  No Python processes to kill"
sleep 2

python3 << 'EOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("  ✓ GPU cache cleared")
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"  ✓ Free GPU memory: {free_mem / 1024**3:.2f} GB")
else:
    print("  ⚠️  CUDA not available")
gc.collect()
EOF
echo ""

# Step 7: Check GPU availability
echo "Step 7: Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

FREE_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
FREE_MEM_GB=$(echo "scale=1; $FREE_MEM_MB/1024" | bc)

echo "  Free VRAM: ${FREE_MEM_GB} GB"

if (( $(echo "$FREE_MEM_GB < 20" | bc -l) )); then
    echo "  ⚠️  WARNING: Less than 20GB free VRAM"
    echo "  Training may have tight memory margins"
fi
echo ""

# Step 8: Verify environment
echo "Step 8: Verifying Python environment..."
echo "  Python: $(python --version)"
echo "  Environment: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}"
echo ""

python -c "
import torch
import transformers
import peft
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"
echo ""

# Step 9: Check if in tmux
echo "Step 9: Checking tmux session..."
if [ -z "$TMUX" ]; then
    echo "  ⚠️  WARNING: You are NOT in a tmux session!"
    echo ""
    echo "  Training will take ~15 hours. Strongly recommended to run in tmux:"
    echo ""
    echo "    tmux new -s baseline_7b"
    echo "    bash VERIFY_AND_START.sh"
    echo ""
    read -p "  Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Cancelled. Please start tmux first."
        exit 1
    fi
else
    echo "  ✓ Running in tmux session"
fi
echo ""

# Summary
echo "=========================================="
echo "VERIFICATION COMPLETE - CONFIGURATION"
echo "=========================================="
echo ""
echo "Configuration verified:"
echo "  • Model: Qwen 2.5 7B"
echo "  • LoRA rank: 128"
echo "  • MAX_SEQ_LENGTH: 512 ✓"
echo "  • Batch size: 1 ✓"
echo "  • Gradient accumulation: 16"
echo "  • Effective batch: 16"
echo "  • Max GPU memory: 14GB ✓"
echo "  • CPU offload: 50GB"
echo "  • Epochs: 15"
echo "  • Estimated time: ~15 hours"
echo ""

read -p "Start training now? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cancelled. To start training later, run:"
    echo "  bash START_BASELINE_7B_TRAINING.sh"
    echo ""
    exit 0
fi

echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""

# Set memory optimizations
export PYTORCH_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "Memory optimizations applied:"
echo "  • PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "  • TOKENIZERS_PARALLELISM=false"
echo "  • OMP_NUM_THREADS=4"
echo ""

# Start training
echo "🚀 Starting baseline 7B training..."
echo ""
echo "Monitor with: watch -n 10 nvidia-smi"
echo "View logs: tail -f logs/training_baseline*.log"
echo ""

python train_baseline.py --epochs 15 --show-sample

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo ""
