#!/bin/bash
# Start training with the fixed gradient checkpointing
# Run this after applying the critical fix

set -e

echo "==========================================="
echo "BASELINE 7B TRAINING - START"
echo "==========================================="
echo ""

# Pull latest code
echo "Step 1: Pulling latest code..."
git pull origin main
echo "  ✓ Code updated"
echo ""

# Clear Python cache (CRITICAL to ensure fix is applied)
echo "Step 2: Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Cache cleared"
echo ""

# Clear GPU memory
echo "Step 3: Clearing GPU memory..."
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

# Show GPU status
echo "Step 4: GPU Status..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Show current config
echo "Step 5: Current Configuration..."
echo ""
python << 'EOF'
import sys
sys.path.insert(0, '.')
from config.config import MAX_SEQ_LENGTH, TRAINING_CONFIG, LORA_CONFIG, MODEL_NAME

print(f"  Model: {MODEL_NAME}")
print(f"  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
print(f"  Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
print(f"  Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
print(f"  Effective batch: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
print(f"  LoRA rank: {LORA_CONFIG['r']}")
print(f"  FP16: {TRAINING_CONFIG.get('fp16', False)}")
print(f"  BF16: {TRAINING_CONFIG.get('bf16', False)}")
print(f"  Epochs: {TRAINING_CONFIG['num_train_epochs']}")
EOF
echo ""

# Check if gradient checkpointing is in correct place
echo "Step 6: Verifying gradient checkpointing fix..."
CHECKPOINT_AFTER_LORA=$(grep -A 10 "get_peft_model" train_baseline.py | grep -c "gradient_checkpointing_enable" || echo "0")
if [ "$CHECKPOINT_AFTER_LORA" -gt 0 ]; then
    echo "  ✅ gradient_checkpointing_enable() is AFTER LoRA (CORRECT)"
else
    echo "  ⚠️  WARNING: Cannot verify gradient checkpointing position"
fi
echo ""

# Check if in tmux
echo "Step 7: Checking tmux session..."
if [ -z "$TMUX" ]; then
    echo "  ⚠️  WARNING: You are NOT in a tmux session!"
    echo ""
    echo "  Training will take ~10-15 hours. Recommended to run in tmux:"
    echo ""
    echo "    tmux new -s baseline_7b"
    echo "    bash START_TRAINING_NOW.sh"
    echo ""
    read -p "  Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Cancelled. Please start tmux first."
        exit 1
    fi
else
    echo "  ✓ Running in tmux session: $TMUX"
fi
echo ""

# Final confirmation
echo "==========================================="
echo "READY TO START TRAINING"
echo "==========================================="
echo ""
read -p "Start training now? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cancelled."
    exit 0
fi

echo ""
echo "==========================================="
echo "STARTING TRAINING"
echo "==========================================="
echo ""

# Set memory optimizations
# Note: expandable_segments requires PyTorch >= 2.1, we have 2.0.1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "Memory optimizations applied:"
echo "  • PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  • TOKENIZERS_PARALLELISM=false"
echo "  • OMP_NUM_THREADS=4"
echo ""

# Start training
echo "🚀 Starting baseline 7B training..."
echo ""
echo "Monitor with:"
echo "  watch -n 10 nvidia-smi"
echo ""
echo "View logs:"
echo "  tail -f logs/training_baseline*.log"
echo ""
echo "Detach from tmux: Ctrl+B, then D"
echo ""

python train_baseline.py --epochs 15 --show-sample

echo ""
echo "==========================================="
echo "TRAINING COMPLETED"
echo "==========================================="
echo ""
