#!/bin/bash
################################################################################
# Fix NumPy Compatibility on Server
# Downgrades NumPy to 1.x for compatibility with pandas/sklearn
################################################################################

echo "========================================================================"
echo "FIX NUMPY COMPATIBILITY"
echo "========================================================================"
echo ""

# Check current NumPy version
echo "📦 Current NumPy version:"
python3 -c "import numpy; print(f'   NumPy {numpy.__version__}')" 2>/dev/null || echo "   NumPy not found"
echo ""

# Downgrade NumPy
echo "🔧 Downgrading NumPy to 1.x for compatibility..."
pip install "numpy<2.0.0" --upgrade

echo ""
echo "✅ NumPy downgrade complete"
echo ""

# Verify
echo "📦 New NumPy version:"
python3 -c "import numpy; print(f'   NumPy {numpy.__version__}')"
echo ""

# Reinstall dependencies to ensure compatibility
echo "🔄 Reinstalling pandas and scikit-learn for compatibility..."
pip install --force-reinstall pandas scikit-learn

echo ""
echo "========================================================================"
echo "✅ NUMPY FIX COMPLETE"
echo "========================================================================"
echo ""
echo "Now you can run training:"
echo "  python3 train_mtup.py --use-case quick_test"
echo ""
