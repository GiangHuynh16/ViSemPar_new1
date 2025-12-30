#!/bin/bash
# Start Baseline training with Qwen 2.5 7B + LoRA rank 128
# For fair comparison with MTUP 7B

set -e

echo "=========================================="
echo "BASELINE 7B TRAINING - SINGLE-TASK"
echo "=========================================="
echo ""
echo "Baseline approach (Single-Task):"
echo "  Model: Qwen 2.5 7B"
echo "  LoRA rank: 128"
echo "  Trainable params: ~239M (same as MTUP)"
echo "  Task: Direct Sentence → AMR mapping"
echo "  Purpose: Fair comparison with MTUP"
echo ""
echo "Training configuration:"
echo "  - Epochs: 15 (same as MTUP)"
echo "  - Batch size: 1 (per device, optimized for memory)"
echo "  - Gradient accumulation: 16 (increased to maintain effective batch)"
echo "  - Effective batch size: 16 (same as MTUP)"
echo "  - Max sequence length: 1536 (optimized from 2048)"
echo "  - Learning rate: 2e-4"
echo "  - Estimated time: ~12-15 hours"
echo "  - Peak VRAM usage: ~18-20 GB (optimized)"
echo ""

# Check VRAM
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ nvidia-smi not found"
    exit 1
fi

free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
free_gb=$(echo "scale=1; $free_vram/1024" | bc)

echo "  Free VRAM: ${free_gb} GB"

if (( $(echo "$free_gb < 18" | bc -l) )); then
    echo "  ✗ Not enough VRAM (need >= 18 GB for 7B model)"
    echo "  Current free: ${free_gb} GB"
    exit 1
fi

echo "  ✓ VRAM check passed"
echo ""

# Check if in tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  WARNING: You are NOT in a tmux session!"
    echo ""
    echo "Training will take ~12-15 hours. Recommended to run in tmux:"
    echo ""
    echo "  tmux new -s baseline_7b"
    echo "  bash START_BASELINE_7B_TRAINING.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Please start tmux first."
        exit 1
    fi
else
    echo "✓ Running in tmux session: $TMUX"
fi

echo ""
echo "🚀 Starting 7B baseline training..."
echo ""

# Set memory optimizations
echo "Applying memory optimizations..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
echo "  ✓ PYTORCH_CUDA_ALLOC_CONF set"
echo "  ✓ TOKENIZERS_PARALLELISM=false"
echo ""

# Clear GPU cache before training
python3 << 'EOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("  ✓ GPU cache cleared")
EOF
echo ""

# Run training
python train_baseline.py --epochs 15 --show-sample

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation:"
echo "     python evaluate_baseline_model.py \\"
echo "       --checkpoint outputs/checkpoints/baseline_7b_final \\"
echo "       --test-file data/public_test_ground_truth.txt \\"
echo "       --output results/baseline_7b_evaluation.json"
echo ""
echo "  2. Check F1 score:"
echo "     cat results/baseline_7b_evaluation.json"
echo ""
echo "  3. Compare with MTUP 7B:"
echo "     echo 'MTUP 7B: F1 = [from previous evaluation]'"
echo "     echo 'Baseline 7B: F1 = [check results]'"
echo ""
echo "  4. If satisfactory, push to HuggingFace:"
echo "     hf upload YOUR-USERNAME/vietnamese-amr-baseline-7b \\"
echo "       outputs/checkpoints/baseline_7b_final"
echo ""
