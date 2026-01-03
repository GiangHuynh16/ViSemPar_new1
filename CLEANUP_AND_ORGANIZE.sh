#!/bin/bash
# Clean and organize server files
# Keep only important models and results

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

echo "=========================================="
echo "CLEANUP AND ORGANIZE SERVER FILES"
echo "=========================================="
echo ""

# Create organized directories
echo "Step 1: Creating organized directories..."
mkdir -p models_archive/{baseline_7b_old,baseline_7b_fixed}
mkdir -p evaluation_results/{baseline_7b_old,baseline_7b_fixed}
mkdir -p temp_files
echo "  ✓ Directories created"
echo ""

# ============================================================================
# MOVE OLD BASELINE MODEL (trained with bugs)
# ============================================================================
echo "Step 2: Archiving old baseline model (with bugs)..."

OLD_CHECKPOINT="outputs/baseline_20260102_125130/checkpoint-1545"
if [ -d "$OLD_CHECKPOINT" ]; then
    echo "  Moving old baseline checkpoint..."
    mv "$OLD_CHECKPOINT" models_archive/baseline_7b_old/
    echo "  ✓ Moved to models_archive/baseline_7b_old/"
fi

# Move old evaluation results
echo "  Moving old evaluation results..."
mv predictions_formatted.txt evaluation_results/baseline_7b_old/ 2>/dev/null || true
mv public_test_result_baseline_7b.txt evaluation_results/baseline_7b_old/ 2>/dev/null || true
mv predictions_valid.txt evaluation_results/baseline_7b_old/ 2>/dev/null || true
mv gold_valid.txt evaluation_results/baseline_7b_old/ 2>/dev/null || true
echo "  ✓ Moved evaluation files"
echo ""

# ============================================================================
# CLEAN TEMPORARY FILES
# ============================================================================
echo "Step 3: Cleaning temporary files..."

# Move temp files to temp directory
echo "  Moving temporary base64 files..."
mv *.b64 temp_files/ 2>/dev/null || true
echo "  ✓ Moved base64 files"

# Clean old outputs directory (keep only archives)
echo "  Cleaning old output directories..."
if [ -d "outputs/baseline_20260102_125130" ]; then
    # Keep only final checkpoint, remove intermediate
    cd outputs/baseline_20260102_125130
    for dir in checkpoint-*; do
        if [ "$dir" != "checkpoint-1545" ]; then
            echo "    Removing intermediate checkpoint: $dir"
            rm -rf "$dir"
        fi
    done
    cd ../..
fi
echo "  ✓ Cleaned intermediate checkpoints"
echo ""

# ============================================================================
# ORGANIZE MODELS
# ============================================================================
echo "Step 4: Organizing models with clear names..."

# Create README for each model
cat > models_archive/baseline_7b_old/README.md << 'EOF'
# Baseline 7B Model (OLD - HAS BUGS)

**Training Date:** 2026-01-02
**Status:** ⚠️ Has critical bugs - DO NOT USE

## Issues:
1. ❌ Missing EOS token - model doesn't know when to stop
2. ❌ No instruction masking - trained on prompt too
3. ❌ Unclear Penman format - generates invalid AMR

## Results:
- 150 test sentences
- 26/150 invalid AMRs (17.3% error rate)
- Cannot calculate SMATCH due to syntax errors

## Files:
- `checkpoint-1545/` - Final checkpoint
- See `evaluation_results/baseline_7b_old/` for outputs

**Use the FIXED version instead!**
EOF

cat > evaluation_results/baseline_7b_old/README.md << 'EOF'
# Baseline 7B Old - Evaluation Results

**Model:** baseline_7b_old (buggy version)
**Test set:** public_test (150 sentences)

## Files:
- `predictions_formatted.txt` - Formatted predictions with #::snt
- `public_test_result_baseline_7b.txt` - Raw model outputs
- `predictions_valid.txt` - Filtered valid AMRs only (124/150)
- `gold_valid.txt` - Corresponding gold AMRs

## Quality Issues:
- 26/150 AMRs invalid (17.3% error rate)
  - Unmatched parentheses
  - Duplicate node names
  - Model generating explanations after AMR

**SMATCH score:** Unable to calculate due to parse errors
EOF

cat > models_archive/README.md << 'EOF'
# Models Archive

## Directory Structure:

### baseline_7b_old/
⚠️ **DO NOT USE** - Has critical training bugs
- Missing EOS token
- No instruction masking
- Unclear prompt

### baseline_7b_fixed/
✅ **USE THIS** - Fixed version
- Added EOS token
- Instruction masking enabled
- Clear Penman format prompt
- Should generate valid AMR only

## Training Command (Fixed):
```bash
conda activate baseline_final
python train_baseline_fixed.py --epochs 15 --show-sample
```
EOF

echo "  ✓ Created README files for documentation"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "📁 Directory structure:"
echo ""
echo "models_archive/"
echo "  ├── baseline_7b_old/          ⚠️  Buggy model (archived)"
echo "  │   └── checkpoint-1545/"
echo "  ├── baseline_7b_fixed/        ✅ Fixed model (train this)"
echo "  └── README.md"
echo ""
echo "evaluation_results/"
echo "  ├── baseline_7b_old/          Old results (17.3% errors)"
echo "  └── baseline_7b_fixed/        New results (after retraining)"
echo ""
echo "temp_files/"
echo "  └── *.b64                     Base64 encoded files"
echo ""
echo "outputs/"
echo "  └── baseline_fixed_YYYYMMDD_HHMMSS/  ← New model will be here"
echo ""

# Show disk usage
echo "💾 Disk usage:"
du -sh models_archive/* evaluation_results/* temp_files/ 2>/dev/null | column -t
echo ""

echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Train fixed model:"
echo "   conda activate baseline_final"
echo "   python train_baseline_fixed.py --epochs 15 --show-sample"
echo ""
echo "2. After training, move model:"
echo "   mv outputs/baseline_fixed_*/final models_archive/baseline_7b_fixed/"
echo ""
echo "3. Test and evaluate:"
echo "   python predict_baseline_fixed.py"
echo ""
echo "=========================================="
