#!/bin/bash
# Validate everything before retraining
# Run this on the server BEFORE starting training

echo "=========================================="
echo "PRE-TRAINING VALIDATION"
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Step 1: Validate training data
echo "=========================================="
echo "STEP 1: Validate training data"
echo "=========================================="
echo ""

echo "Checking train_amr_1.txt..."
python validate_vietnamese_output.py --file data/train_amr_1.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ train_amr_1.txt has issues!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Checking train_amr_2.txt..."
python validate_vietnamese_output.py --file data/train_amr_2.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ train_amr_2.txt has issues!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Step 2: Test tokenization fix
echo "=========================================="
echo "STEP 2: Test tokenization fix"
echo "=========================================="
echo ""

python diagnose_tokenization.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tokenization diagnosis failed!"
    echo "Please review the output above."
    exit 1
fi

echo ""

# Step 3: Quick sanity check with 10 samples
echo "=========================================="
echo "STEP 3: Quick training test (10 samples)"
echo "=========================================="
echo ""

echo "This will train on 10 samples to verify the fix works..."
echo ""

python train_baseline_fixed.py \
    --max-samples 10 \
    --num-epochs 1 \
    --show-sample

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Quick training test failed!"
    echo "Please review errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL VALIDATION CHECKS PASSED"
echo "=========================================="
echo ""
echo "Ready to retrain full model!"
echo ""
echo "To start training:"
echo "  bash TRAIN_BASELINE_FIXED.sh"
echo ""
echo "Or train with monitoring:"
echo "  bash TRAIN_BASELINE_FIXED.sh 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt"
echo ""
