#!/bin/bash
# Cleanup 3B model to make space for 7B model training

set -e

echo "=========================================="
echo "CLEANUP 3B MODEL"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - outputs/checkpoints_mtup/mtup_full_training_final/"
echo "  - outputs/checkpoints_mtup/ (all checkpoints)"
echo "  - results/mtup_evaluation_final.json"
echo ""
echo "Space to be freed: ~5-7 GB"
echo ""

# Confirm
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Removing 3B model files..."

# Remove final model
if [ -d "outputs/checkpoints_mtup/mtup_full_training_final" ]; then
    rm -rf outputs/checkpoints_mtup/mtup_full_training_final
    echo "  ✓ Removed mtup_full_training_final/"
else
    echo "  ⊘ mtup_full_training_final/ not found"
fi

# Remove all checkpoints
if [ -d "outputs/checkpoints_mtup" ]; then
    find outputs/checkpoints_mtup -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Removed checkpoint-* directories"
else
    echo "  ⊘ checkpoints_mtup/ not found"
fi

# Remove evaluation results
if [ -f "results/mtup_evaluation_final.json" ]; then
    rm -f results/mtup_evaluation_final.json
    echo "  ✓ Removed mtup_evaluation_final.json"
else
    echo "  ⊘ mtup_evaluation_final.json not found"
fi

echo ""
echo "=========================================="
echo "CLEANUP COMPLETED"
echo "=========================================="
echo ""
echo "Disk space freed:"
du -sh outputs/checkpoints_mtup 2>/dev/null || echo "  0 MB (directory cleaned)"
echo ""
echo "Ready to train 7B model!"
echo ""
echo "Next steps:"
echo "  1. Verify config updated:"
echo "     grep 'MODEL_NAME' config/config_mtup.py"
echo ""
echo "  2. Start 7B training:"
echo "     bash START_MTUP_7B_TRAINING.sh"
echo ""
