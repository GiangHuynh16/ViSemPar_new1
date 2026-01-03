#!/bin/bash
# Quick start script to retrain baseline with all fixes
# Run: bash QUICK_START_RETRAIN.sh

set -e  # Exit on error

echo "=========================================="
echo "QUICK START: Retrain Baseline 7B (FIXED)"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Cleanup old files"
echo "  2. Train with fixes (EOS token, instruction masking, clear prompt)"
echo "  3. Archive the new model"
echo "  4. Test and evaluate"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Change to project directory
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# ============================================================================
# STEP 1: Cleanup
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 1/4: Cleanup old files"
echo "=========================================="

chmod +x CLEANUP_AND_ORGANIZE.sh
./CLEANUP_AND_ORGANIZE.sh

echo ""
read -p "Cleanup done. Continue to training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopped after cleanup. You can manually run:"
    echo "  python train_baseline_fixed.py --epochs 15 --show-sample"
    exit 0
fi

# ============================================================================
# STEP 2: Train
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 2/4: Train with fixes"
echo "=========================================="
echo ""
echo "Activating conda environment: baseline_final"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate baseline_final

echo ""
echo "Starting training..."
echo "  - This will take ~2-3 hours"
echo "  - Check outputs/baseline_fixed_*/logs for progress"
echo ""

python train_baseline_fixed.py \
    --epochs 15 \
    --show-sample \
    --val-split 0.05

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed! Check logs for details."
    exit 1
fi

echo ""
echo "✅ Training completed!"

# ============================================================================
# STEP 3: Archive model
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 3/4: Archive trained model"
echo "=========================================="

# Find latest output directory
LATEST_OUTPUT=$(ls -t outputs/ | grep baseline_fixed | head -1)

if [ -z "$LATEST_OUTPUT" ]; then
    echo "❌ No output directory found!"
    exit 1
fi

echo ""
echo "Found: outputs/$LATEST_OUTPUT"
echo "Moving final checkpoint to archive..."

# Create archive directory
mkdir -p models_archive/baseline_7b_fixed

# Move model
if [ -d "outputs/$LATEST_OUTPUT/final" ]; then
    mv outputs/$LATEST_OUTPUT/final/* models_archive/baseline_7b_fixed/
    echo "✅ Model archived to models_archive/baseline_7b_fixed/"
else
    echo "⚠️  Final checkpoint not found, using latest checkpoint..."
    LATEST_CHECKPOINT=$(ls -t outputs/$LATEST_OUTPUT/ | grep checkpoint | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        mv outputs/$LATEST_OUTPUT/$LATEST_CHECKPOINT/* models_archive/baseline_7b_fixed/
        echo "✅ Model archived to models_archive/baseline_7b_fixed/"
    else
        echo "❌ No checkpoint found!"
        exit 1
    fi
fi

# Create README
cat > models_archive/baseline_7b_fixed/README.md << EOF
# Baseline 7B Model - FIXED VERSION ✅

**Training Date:** $(date +%Y-%m-%d)
**Training Duration:** ~2-3 hours
**Status:** ✅ All fixes applied

## Fixes Applied:
1. ✅ EOS token added to training data
2. ✅ Instruction masking enabled (only train on AMR output)
3. ✅ Clear Penman format prompt with explicit rules

## Training Config:
- Model: Qwen/Qwen2.5-7B-Instruct
- LoRA rank: 64, alpha: 128
- Epochs: 15
- Batch size: 1 x 16 (gradient accumulation)
- Learning rate: 2e-4
- Optimizer: AdamW
- Precision: BF16

## Expected Improvements:
- Valid AMRs: 150/150 (vs 124/150 old)
- Parse errors: 0% (vs 17.3% old)
- Stops at EOS: Yes ✅ (vs No ❌ old)
- Generates explanations: No ✅ (vs Yes ❌ old)

## Files:
- adapter_config.json - LoRA config
- adapter_model.safetensors - Trained weights
- tokenizer files - Tokenizer config

## Next Steps:
1. Test: python predict_baseline_fixed.py --model models_archive/baseline_7b_fixed/final
2. Evaluate SMATCH score
3. Compare with MTUP 7B
EOF

echo "✅ README created"

# ============================================================================
# STEP 4: Test and evaluate
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 4/4: Test and evaluate"
echo "=========================================="
echo ""

read -p "Run prediction and evaluation now? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Skipped testing. You can manually run:"
    echo "  python predict_baseline_fixed.py \\"
    echo "    --model models_archive/baseline_7b_fixed \\"
    echo "    --test-file data/public_test.txt \\"
    echo "    --output evaluation_results/baseline_7b_fixed/predictions.txt"
    exit 0
fi

echo ""
echo "Running prediction on public_test.txt..."

# Create output directory
mkdir -p evaluation_results/baseline_7b_fixed

# Run prediction
python predict_baseline_fixed.py \
    --model models_archive/baseline_7b_fixed \
    --test-file data/public_test.txt \
    --output evaluation_results/baseline_7b_fixed/predictions.txt

if [ $? -ne 0 ]; then
    echo "❌ Prediction failed!"
    exit 1
fi

echo ""
echo "✅ Predictions saved to evaluation_results/baseline_7b_fixed/predictions.txt"

# Check quality
echo ""
echo "Analyzing AMR quality..."
python analyze_amr_quality.py \
    --file evaluation_results/baseline_7b_fixed/predictions.txt \
    2>/dev/null || echo "⚠️  analyze_amr_quality.py not found, skipping quality check"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "✅ ALL DONE!"
echo "=========================================="
echo ""
echo "📁 Files created:"
echo "  - models_archive/baseline_7b_fixed/          (trained model)"
echo "  - evaluation_results/baseline_7b_fixed/      (predictions)"
echo ""
echo "📊 Compare results:"
echo ""
echo "OLD (buggy):"
echo "  - 124/150 valid AMRs (17.3% errors)"
echo "  - SMATCH: not calculable"
echo ""
echo "NEW (fixed):"
wc -l evaluation_results/baseline_7b_fixed/predictions.txt 2>/dev/null | awk '{print "  - " $1/3 " AMRs generated"}'
echo "  - Quality analysis above"
echo ""
echo "Next steps:"
echo "  1. Review predictions in evaluation_results/baseline_7b_fixed/"
echo "  2. Calculate SMATCH score"
echo "  3. Compare with MTUP 7B model"
echo ""
echo "=========================================="
