#!/bin/bash
# Start MTUP training with Qwen 2.5 7B + LoRA rank 128
# Expected improvement: F1 0.46 → 0.51-0.52

set -e

echo "=========================================="
echo "MTUP 7B TRAINING - PERFORMANCE UPGRADE"
echo "=========================================="
echo ""
echo "Previous model (3B):"
echo "  Model: Qwen 2.5 3B"
echo "  LoRA rank: 64"
echo "  Trainable params: ~7M"
echo "  F1 score: 0.4599"
echo "  Success rate: 125/150 (83.3%)"
echo ""
echo "New model (7B):"
echo "  Model: Qwen 2.5 7B"
echo "  LoRA rank: 128"
echo "  Trainable params: ~28M"
echo "  Expected F1: 0.51-0.52 (+0.05-0.06)"
echo "  Expected success: 135-140/150 (90%+)"
echo ""
echo "Training configuration:"
echo "  - Epochs: 15"
echo "  - Batch size: 2 (per device)"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch size: 16"
echo "  - Learning rate: 2e-4"
echo "  - Estimated time: ~12-15 hours"
echo "  - Peak VRAM usage: ~20-22 GB"
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
    echo "  tmux new -s mtup_7b"
    echo "  bash START_MTUP_7B_TRAINING.sh"
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
echo "🚀 Starting 7B training..."
echo ""

# Run training
python train_mtup.py --use-case full_training --epochs 15

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation:"
echo "     python evaluate_mtup_model.py \\"
echo "       --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \\"
echo "       --test-file data/public_test_ground_truth.txt \\"
echo "       --output results/mtup_7b_evaluation.json"
echo ""
echo "  2. Check F1 score:"
echo "     cat results/mtup_7b_evaluation.json"
echo ""
echo "  3. Compare with 3B baseline:"
echo "     echo '3B model: F1 = 0.4599'"
echo "     echo '7B model: F1 = [check results]'"
echo ""
echo "  4. If F1 >= 0.50, push to HuggingFace:"
echo "     huggingface-cli upload your-username/vietnamese-amr-mtup-7b \\"
echo "       outputs/checkpoints_mtup/mtup_full_training_final"
echo ""
