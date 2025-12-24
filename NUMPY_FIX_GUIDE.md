# NumPy Compatibility Fix for Server

## Problem

The server has **NumPy 2.3.5**, but dependencies like `pandas`, `scikit-learn`, `bottleneck`, and `numexpr` were compiled with **NumPy 1.x**, causing this error:

```
AttributeError: _ARRAY_API not found
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

## Quick Fix (Method 1 - Recommended)

Run the automated fix script:

```bash
cd ~/ViSemPar_new1
bash fix_numpy_server.sh
```

This will:
1. Downgrade NumPy to 1.x
2. Reinstall pandas and scikit-learn for compatibility
3. Verify the fix

## Manual Fix (Method 2)

If the script doesn't work, fix manually:

```bash
# Downgrade NumPy
pip install "numpy<2.0.0" --upgrade

# Reinstall dependencies
pip install --force-reinstall pandas scikit-learn

# Verify
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import pandas; print('pandas OK')"
python3 -c "import sklearn; print('sklearn OK')"
```

## Expected Result

After the fix:

```
NumPy version: 1.26.4 (or similar 1.x version)
pandas: OK
sklearn: OK
```

## Verify Training Works

```bash
python3 train_mtup.py --use-case quick_test --show-sample
```

You should see:
```
✓ Created: /mnt/nghiepth/giang/ViSemPar/outputs/checkpoints_mtup
✓ Training file found: train_amr_1.txt
✓ Training file found: train_amr_2.txt
✅ Environment check passed

Loading MTUP data...
```

## Why This Happens

- **NumPy 2.x** (released 2024) changed internal C API
- Packages compiled with NumPy 1.x are **not compatible** with NumPy 2.x
- Solution: Use NumPy 1.x until all dependencies update

## Long-term Solution

Wait for these packages to release NumPy 2.x compatible versions:
- pandas (working on 2.x support)
- scikit-learn (working on 2.x support)
- bottleneck, numexpr (pending updates)

For now, **NumPy 1.x is the stable choice** for ML/NLP projects.

---

## Alternative: Virtual Environment

If you want to avoid affecting other projects:

```bash
# Create new environment
conda create -n amr-mtup python=3.10 -y
conda activate amr-mtup

# Install compatible versions
pip install -r requirements.txt

# Run training
python3 train_mtup.py --use-case quick_test
```

This isolates the fix to this project only.
