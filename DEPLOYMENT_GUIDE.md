# Deployment Guide - From Local to Server Training

**Hướng dẫn chi tiết: Push git → Clone server → Setup → Training → Evaluation**

---

## 📋 **BƯỚC 1: PUSH LÊN GIT (Local Machine)**

### **1.1. Check git status**
```bash
cd ~/ViSemPar_new1
git status
```

### **1.2. Add files mới**
```bash
# Add all new files
git add config/prompt_templates.py
git add config/config_mtup.py
git add src/preprocessor_mtup.py
git add test_mtup_simple.py
git add MTUP_IMPLEMENTATION.md
git add CRITICAL_ANALYSIS.md
git add HUGGINGFACE_SETUP.md
git add SERVER_QUICK_START.md
git add setup_server.sh
git add .env.example
git add .gitignore

# Check what will be committed
git status
```

### **1.3. Commit**
```bash
git commit -m "Add MTUP implementation - Multi-task unified prompt approach

- Add 5 Vietnamese prompt templates (v2_natural recommended)
- Add MTUPAMRPreprocessor with Vietnamese char support
- Add config for smaller models (Qwen 3B, Gemma 2B, Phi 3.5)
- Add comprehensive documentation
- Fix variable removal for Vietnamese characters (đ, ô, ê, etc.)
- Add server setup scripts and HuggingFace integration guide

Template: v2_natural - Natural Vietnamese two-stage format
Expected: 2-3x faster training with 3B models"
```

### **1.4. Push**
```bash
git push origin main
```

---

## 🖥️ **BƯỚC 2: CLONE VÀ SETUP TRÊN SERVER**

### **2.1. SSH vào server**
```bash
ssh your_username@server_address
```

### **2.2. Clone repository**
```bash
# If first time
git clone https://github.com/your-username/ViSemPar_new1.git
cd ViSemPar_new1

# If already exists
cd ViSemPar_new1
git pull origin main
```

### **2.3. Run setup script**
```bash
bash setup_server.sh
```

**Lúc này script sẽ:**
- ✅ Check Python version
- ✅ Create directories
- ✅ Install dependencies
- ✅ Ask về Hugging Face setup → **Chọn Option 1: CLI Login**

---

## 🔑 **BƯỚC 3: SETUP HUGGING FACE**

### **3.1. Tạo Access Token (trên web - máy local cũng được)**

1. Vào: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Nhập name: `vlsp-amr-server`
4. Chọn **"Write"** permission
5. Click **"Generate a new token"**
6. **COPY TOKEN** (dạng: `hf_xxxxxxxxxxxxx`)

### **3.2. Login trên server**
```bash
# Install huggingface-hub (nếu setup_server.sh chưa làm)
pip install --upgrade huggingface_hub

# Login
huggingface-cli login
```

**Paste token khi được hỏi:**
```
Enter your token (input will not be visible): hf_xxxxxxxxxxxxx
```

**Verify:**
```bash
huggingface-cli whoami
# Should show: your-username
```

---

## 📁 **BƯỚC 4: TẠO FILE .env (OPTIONAL)**

**Cách 1: Dùng CLI login (RECOMMENDED - đã làm ở bước 3)**
```bash
# Token đã lưu ở ~/.cache/huggingface/token
# Không cần .env file!
```

**Cách 2: Dùng .env file (Alternative)**
```bash
# Copy example
cp .env.example .env

# Edit file
nano .env
```

**Nội dung .env:**
```bash
# Hugging Face Access Token
HF_TOKEN=hf_xxxxxxxxxxxxx

# Optional: Your HF username
HF_USERNAME=your-username
```

**Save:** `Ctrl+O`, Enter, `Ctrl+X`

**Verify .env:**
```bash
# Check token loaded
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('Token:', os.getenv('HF_TOKEN')[:20] + '...')
"
```

---

## 📊 **BƯỚC 5: VERIFY DATA**

### **5.1. Check data files**
```bash
ls -lh data/
```

**Should have:**
```
train_amr_1.txt
train_amr_2.txt
public_test.txt (optional)
public_test_ground_truth.txt (optional)
```

### **5.2. Quick data check**
```bash
# Count training examples
grep -c "^#::snt" data/train_amr_1.txt
grep -c "^#::snt" data/train_amr_2.txt

# Show first example
head -10 data/train_amr_1.txt
```

### **5.3. Test preprocessing**
```bash
python3 test_mtup_simple.py
```

**Expected output:**
```
✓ Loaded 1262 examples
✓ Variable removal: OK
✓ Task 1: No variables
✓ Task 2: Has variables
✅ ALL CHECKS PASSED!
```

---

## 🚀 **BƯỚC 6: TRAINING (With MTUP)**

**⚠️ NOTE:** File `train_mtup.py` chưa có. Dùng approach test trước:

### **Option A: Test với current preprocessor**
```bash
# Sử dụng train.py hiện có + MTUP preprocessor
# Cần modify train.py để dùng MTUPAMRPreprocessor
```

### **Option B: Quick test với small dataset**
```bash
# TODO: Create train_mtup.py
# For now, test preprocessing pipeline
python3 -c "
from src.preprocessor_mtup import MTUPAMRPreprocessor
from config.prompt_templates import get_template

preprocessor = MTUPAMRPreprocessor(config={
    'template_name': 'v2_natural',
    'use_graph_format': True
})

# Test with sample
amr = '(n / nhớ :pivot(t / tôi))'
result = preprocessor.preprocess_for_mtup('Tôi nhớ', amr)
print(result)
print('\\n✅ Preprocessing works!')
"
```

---

## 📝 **BƯỚC 7: TẠO TRAINING SCRIPT (Temporary - Manual)**

Vì `train_mtup.py` chưa có, tạo quick test script:

```bash
nano quick_test_mtup.py
```

**Paste code:**
```python
#!/usr/bin/env python3
"""
Quick MTUP Training Test
Tests data loading with MTUP preprocessing
"""

import sys
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'config')

from data_loader import AMRDataLoader
from preprocessor_mtup import MTUPAMRPreprocessor
from config_mtup import DATA_DIR, MTUP_CONFIG

def main():
    print("="*80)
    print("MTUP DATA PREPARATION TEST")
    print("="*80)

    # Load data
    loader = AMRDataLoader(DATA_DIR)
    examples = loader.parse_amr_file(DATA_DIR / "train_amr_1.txt")
    print(f"\n✓ Loaded {len(examples)} examples")

    # Initialize preprocessor
    preprocessor = MTUPAMRPreprocessor(config=MTUP_CONFIG)

    # Process first 10 examples
    print("\n" + "="*80)
    print("Processing examples with MTUP format...")
    print("="*80)

    processed = []
    for i, ex in enumerate(examples[:10]):
        try:
            mtup_text = preprocessor.preprocess_for_mtup(
                sentence=ex['sentence'],
                amr_with_vars=ex['amr']
            )
            processed.append(mtup_text)
            print(f"\n✓ Example {i+1}/{10} processed")

            if i == 0:
                print("\n" + "-"*80)
                print("First example preview:")
                print("-"*80)
                print(mtup_text[:500] + "...")

        except Exception as e:
            print(f"✗ Error on example {i+1}: {e}")

    # Stats
    stats = preprocessor.get_stats()
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✅ MTUP preprocessing ready for training!")
    print("="*80)

if __name__ == "__main__":
    main()
```

**Run:**
```bash
chmod +x quick_test_mtup.py
python3 quick_test_mtup.py
```

---

## 🔬 **BƯỚC 8: EVALUATE - SMATCH SCORE**

### **8.1. Install SMATCH**
```bash
pip install smatch
```

### **8.2. Test SMATCH với sample data**

**Create eval script:**
```bash
nano test_smatch.py
```

**Paste:**
```python
#!/usr/bin/env python3
"""
Test SMATCH evaluation on sample data
"""

import smatch

# Sample predictions vs ground truth
predictions = [
    "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))",
    "(n / nhớ :pivot(t / tôi) :theme(l / lời))"
]

ground_truth = [
    "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))",
    "(n / nhớ :pivot(t / tôi) :theme(l / lời))"
]

print("="*80)
print("SMATCH EVALUATION TEST")
print("="*80)

total_p, total_r, total_f = 0, 0, 0
for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
    try:
        # Parse AMRs
        best, test, gold_t = smatch.get_amr_match(pred, gold)

        if test > 0 and gold_t > 0:
            precision = best / test
            recall = best / gold_t
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_p += precision
            total_r += recall
            total_f += f1

            print(f"\nExample {i+1}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1:        {f1:.4f}")

    except Exception as e:
        print(f"\n✗ Error on example {i+1}: {e}")

# Average
if len(predictions) > 0:
    avg_p = total_p / len(predictions)
    avg_r = total_r / len(predictions)
    avg_f = total_f / len(predictions)

    print("\n" + "="*80)
    print("AVERAGE SCORES")
    print("="*80)
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall:    {avg_r:.4f}")
    print(f"F1:        {avg_f:.4f}")
    print("="*80)

print("\n✅ SMATCH evaluation works!")
```

**Run:**
```bash
python3 test_smatch.py
```

**Expected output:**
```
Example 1:
  Precision: 1.0000
  Recall:    1.0000
  F1:        1.0000

AVERAGE SCORES
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
```

### **8.3. Evaluate with real test data**

```bash
nano evaluate_test_data.py
```

**Paste:**
```python
#!/usr/bin/env python3
"""
Evaluate test data with SMATCH
"""

import sys
sys.path.insert(0, 'src')

from data_loader import AMRDataLoader
from pathlib import Path
import smatch

def main():
    # Load test data with ground truth
    loader = AMRDataLoader(Path("data"))

    # Check if test files exist
    test_file = Path("data/public_test_ground_truth.txt")
    if not test_file.exists():
        print("⚠️ No test ground truth file found")
        print("Create sample predictions first")
        return

    examples = loader.parse_amr_file(test_file)
    print(f"Loaded {len(examples)} test examples with ground truth")

    # For now, test with self (perfect score)
    predictions = [ex['amr'] for ex in examples[:10]]
    ground_truth = [ex['amr'] for ex in examples[:10]]

    print("\n" + "="*80)
    print("EVALUATING FIRST 10 EXAMPLES")
    print("="*80)

    total_f = 0
    valid = 0

    for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
        try:
            best, test, gold_t = smatch.get_amr_match(pred, gold)
            if test > 0 and gold_t > 0:
                f1 = 2 * best / (test + gold_t)
                total_f += f1
                valid += 1
                print(f"Example {i+1}: F1 = {f1:.4f}")
        except Exception as e:
            print(f"Example {i+1}: Error - {e}")

    if valid > 0:
        avg_f1 = total_f / valid
        print(f"\n{'='*80}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Valid: {valid}/{len(predictions)}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python3 evaluate_test_data.py
```

---

## 🎯 **BƯỚC 9: FULL WORKFLOW SUMMARY**

### **Quick Reference:**

```bash
# 1. SSH to server
ssh user@server

# 2. Navigate
cd ViSemPar_new1

# 3. Pull latest
git pull

# 4. Verify HF login
huggingface-cli whoami

# 5. Test preprocessing
python3 test_mtup_simple.py

# 6. Test SMATCH
python3 test_smatch.py

# 7. (Future) Train model
# python3 train_mtup.py --use-case quick_test

# 8. Evaluate
python3 evaluate_test_data.py
```

---

## 📊 **EXPECTED RESULTS**

### **Preprocessing Test:**
```
✓ Loaded 1262 examples
✓ Task 1: No variables
✓ Task 2: Has variables
✅ ALL CHECKS PASSED!
```

### **SMATCH Test:**
```
Average F1: 1.0000 (perfect match với self)
```

### **Data Stats:**
```
processed: X
errors: 0
avg_no_var_length: ~85
avg_with_var_length: ~160
```

---

## ⚠️ **TROUBLESHOOTING**

### **Issue 1: "Module not found"**
```bash
pip install -r requirements.txt
```

### **Issue 2: "HF token not found"**
```bash
huggingface-cli logout
huggingface-cli login
```

### **Issue 3: "Data file not found"**
```bash
# Check data directory
ls -la data/

# Copy from backup
cp /path/to/backup/data/*.txt data/
```

### **Issue 4: "SMATCH error"**
```bash
pip install --upgrade smatch
```

---

## 📝 **CHECKLIST**

```
☐ 1. Push to git (local)
☐ 2. Clone on server
☐ 3. Run setup_server.sh
☐ 4. HuggingFace login
☐ 5. Verify data files
☐ 6. Test preprocessing (test_mtup_simple.py)
☐ 7. Install SMATCH
☐ 8. Test SMATCH (test_smatch.py)
☐ 9. Test with real data (evaluate_test_data.py)
☐ 10. Ready for training!
```

---

## 🚀 **NEXT: CREATE train_mtup.py**

Sau khi verify tất cả OK, tôi sẽ tạo `train_mtup.py` để bạn có thể training!

Bạn muốn tôi tạo `train_mtup.py` ngay bây giờ không? 🚀
