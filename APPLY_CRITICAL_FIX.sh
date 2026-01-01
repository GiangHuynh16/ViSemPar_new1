#!/bin/bash
# Apply critical fix for NaN loss issue
# Run this on server

set -e

echo "==========================================="
echo "APPLY CRITICAL FIX FOR NaN LOSS"
echo "==========================================="
echo ""
echo "Root cause found: gradient_checkpointing_enable()"
echo "was called BEFORE LoRA, causing LoRA parameters"
echo "to not track gradients properly."
echo ""
echo "Fix: Move gradient checkpointing to AFTER LoRA."
echo ""

# Navigate to project
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Stop any running training
echo "Step 1: Stopping any running training..."
pkill -f train_baseline.py 2>/dev/null || true
echo "  ✓ Training stopped"
echo ""

# Pull latest fix
echo "Step 2: Pulling critical fix from GitHub..."
git reset --hard origin/main
git pull origin main
echo "  ✓ Latest code pulled"
echo ""

# Clear Python cache
echo "Step 3: Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Cache cleared"
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Step 4: Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
    echo "  ✓ Environment activated: $CONDA_DEFAULT_ENV"
else
    echo "Step 4: Using current environment: $CONDA_DEFAULT_ENV"
fi
echo ""

# Test the fix
echo "Step 5: Testing the fix with BF16 test..."
echo ""
python test_bf16_forward.py
TEST_EXIT=$?
echo ""

if [ $TEST_EXIT -eq 0 ]; then
    echo "==========================================="
    echo "✅ FIX VERIFIED - READY TO TRAIN"
    echo "==========================================="
    echo ""
    echo "The gradient tracking issue is fixed!"
    echo "You can now start training."
    echo ""
    echo "Start training with:"
    echo "  bash VERIFY_AND_START.sh"
    echo ""
else
    echo "==========================================="
    echo "⚠️  TEST FAILED"
    echo "==========================================="
    echo ""
    echo "Please check the error above."
    echo ""
fi
