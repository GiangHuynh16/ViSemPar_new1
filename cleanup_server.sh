#!/bin/bash
################################################################################
# CLEANUP SERVER - Remove old models and prepare for fresh training
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "🧹 CLEANUP SERVER FOR FRESH MTUP TRAINING"
echo "========================================================================"
echo ""

BACKUP_DIR="outputs/backup_$(date +%Y%m%d_%H%M%S)"

# Step 1: Create backup directory for important files
echo "Step 1: Creating backup directory..."
mkdir -p "$BACKUP_DIR"
echo "✓ Created: $BACKUP_DIR"
echo ""

# Step 2: Backup evaluation results
echo "Step 2: Backing up evaluation results..."
if [ -f "outputs/evaluation_full_20251226_111202.log" ]; then
    cp outputs/evaluation_*.log "$BACKUP_DIR/" 2>/dev/null || true
    cp outputs/evaluation_*.json "$BACKUP_DIR/" 2>/dev/null || true
    echo "✓ Backed up evaluation results"
else
    echo "⚠ No evaluation results found"
fi
echo ""

# Step 3: Show what will be deleted
echo "========================================================================"
echo "📋 FILES TO BE DELETED"
echo "========================================================================"
echo ""
echo "Old 7B baseline models (9.4GB):"
echo "  - outputs/checkpoints/"
echo "  - outputs/vlsp_amr_qwen_improved_v2_NO_VARS_BACKUP/"
echo "  - outputs/vlsp_amr_qwen_7b_v2_F1_0.32/"
echo ""
echo "Empty MTUP training folders:"
echo "  - outputs/mtup_*/"
echo ""
echo "Total space to free: ~12GB"
echo ""

# Step 4: Confirmation
read -p "Type 'DELETE' (all caps) to confirm deletion: " confirmation

if [ "$confirmation" != "DELETE" ]; then
    echo ""
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "========================================================================"
echo "🗑️  DELETING OLD FILES"
echo "========================================================================"
echo ""

# Step 5: Delete old models
echo "Deleting old 7B baseline models..."
rm -rf outputs/checkpoints/
rm -rf outputs/vlsp_amr_qwen_improved_v2_NO_VARS_BACKUP/
rm -rf outputs/vlsp_amr_qwen_7b_v2_F1_0.32/
echo "✓ Deleted 7B models"

# Step 6: Delete empty MTUP folders
echo "Deleting empty MTUP training folders..."
rm -rf outputs/mtup_*/
rm -rf outputs/outputs/  # The nested outputs/ folder
echo "✓ Deleted empty folders"

# Step 7: Clean checkpoints_mtup
echo "Cleaning checkpoints_mtup..."
rm -rf outputs/checkpoints_mtup/
echo "✓ Cleaned checkpoints_mtup"

echo ""
echo "========================================================================"
echo "📁 RECREATING DIRECTORY STRUCTURE"
echo "========================================================================"
echo ""

# Step 8: Recreate clean directory structure
mkdir -p outputs/checkpoints_mtup
mkdir -p outputs/checkpoints
mkdir -p logs
echo "✓ Created directory structure:"
echo "  - outputs/checkpoints_mtup/"
echo "  - outputs/checkpoints/"
echo "  - logs/"

echo ""
echo "========================================================================"
echo "📊 DISK USAGE AFTER CLEANUP"
echo "========================================================================"
echo ""

# Show disk usage
echo "Current usage:"
du -sh outputs/
du -sh logs/
echo ""

echo "Remaining files in outputs/:"
ls -lh outputs/
echo ""

echo "========================================================================"
echo "✅ CLEANUP COMPLETE"
echo "========================================================================"
echo ""

echo "Summary:"
echo "  ✓ Backup created: $BACKUP_DIR"
echo "  ✓ Old models deleted (~12GB freed)"
echo "  ✓ Directory structure ready for training"
echo ""

echo "Next steps:"
echo "  1. Fix training code to save model correctly"
echo "  2. Train MTUP model with 3B"
echo "  3. Verify model is saved to outputs/checkpoints_mtup/mtup_full_training_final/"
echo ""

echo "Optional: Clean HuggingFace cache to free 86GB:"
echo "  rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
echo "  (Keep 3B model, delete only 7B)"
echo ""

echo "========================================================================"
