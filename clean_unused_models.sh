#!/bin/bash
################################################################################
# CLEAN UNUSED MODELS - Free up disk space
# Safe to run - removes only 14B model that's not being used
################################################################################

echo "========================================================================"
echo "🗑️  CLEANING UNUSED MODELS"
echo "========================================================================"
echo ""
echo "Models to remove:"
echo "  - Qwen 2.5 14B (11GB) - not used, training uses 3B model"
echo ""
echo "⚠️  This will free up ~11GB of disk space"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Remove 14B model from HuggingFace cache
echo "🗑️  Removing Qwen 2.5 14B model..."
rm -rf ~/.cache/huggingface/hub/models--unsloth--qwen2.5-14b-instruct-unsloth-bnb-4bit
rm -rf ~/.cache/huggingface/hub/models--unsloth--qwen2.5-14b-instruct

echo ""
echo "✅ Model removed!"
echo ""

# Show new disk usage
echo "New disk usage:"
du -sh ~/.cache/huggingface/hub/models--* | sort -rh | head -10

echo ""
echo "========================================================================"
echo "✅ CLEANUP COMPLETE"
echo "========================================================================"
echo ""
echo "Freed up: ~11GB"
echo ""
echo "⚠️  NOTE: This does NOT fix the OOM error!"
echo "   OOM is caused by GPU RAM (23.64GB), not disk space."
echo ""
echo "To fix OOM, you need to reduce GPU memory usage:"
echo "  1. Reduce batch_size to 1"
echo "  2. Reduce grad_accum to 4"
echo "  3. Reduce max_samples to 50"
echo ""
echo "Run this command on server:"
echo "  python train_mtup.py --use-case quick_test --show-sample \\"
echo "    --batch-size 1 --grad-accum 4 --max-samples 50"
echo ""
