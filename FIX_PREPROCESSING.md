# 🔧 FIX MTUP PREPROCESSING

## Vấn Đề Hiện Tại

`preprocessor_mtup.py` line 71:
```python
def remove_variables(self, amr_string: str) -> str:
    cleaned = re.sub(r'\([^\s/:()]+\s*/', r'(', amr_string)
    return cleaned
```

**Vấn đề**: Chỉ bỏ `(var /` thành `(` → Thiếu balance parentheses!

**Ví dụ sai**:
```
Input:  (a / ăn :agent (t / tôi))
Output: (ăn :agent (tôi))  ← ĐÚNG format nhưng...
Model sinh: ăn:agent(tôi))  ← Thiếu ngoặc mở đầu!
```

## Giải Pháp 1: Giữ Nguyên AMR Structure

**KHÔNG sửa** `remove_variables()`, thay vào đó:

1. Task 1: Sinh AMR đầy đủ KHÔNG có variables
2. Task 2: Thêm variables vào

**Cách fix**:

```python
# File: src/preprocessor_mtup.py

def remove_variables(self, amr_string: str) -> str:
    """
    Remove variables but KEEP parentheses balance
    (a / ăn :agent (t / tôi)) → (ăn :agent (tôi))
    """
    # Remove (var / ...) but keep opening parenthesis
    cleaned = re.sub(r'\([a-z0-9đôêâăươ]+\s*/\s*', r'(', amr_string)
    return cleaned.strip()
```

**Test**:
```python
input_amr = "(a / ăn :agent (t / tôi) :patient (c / cơm))"
output = preprocessor.remove_variables(input_amr)
# Expect: "(ăn :agent (tôi) :patient (cơm))"
```

## Giải Pháp 2: Thay Đổi Training Format

Thay vì bỏ variables, **thay thế bằng placeholders**:

```python
def remove_variables_with_placeholders(self, amr_string: str) -> str:
    """
    (a / ăn) → (<X> / ăn)
    Keeps structure, removes actual variable names
    """
    cleaned = re.sub(r'\([a-z0-9đôêâăươ]+\s*/\s*', r'(<VAR> / ', amr_string)
    return cleaned
```

**Training example**:
```
Task 1: (<VAR> / ăn :agent (<VAR> / tôi))
Task 2: (a / ăn :agent (t / tôi))
```

## Giải Pháp 3: Post-Processing (Quick Fix)

Thêm logic để **fix model output** sau khi sinh:

```python
def fix_incomplete_amr(amr_string: str) -> str:
    """
    Fix common issues:
    - Count opening vs closing parens
    - Add missing parens
    - Validate structure
    """
    open_count = amr_string.count('(')
    close_count = amr_string.count(')')

    if open_count > close_count:
        # Add missing closing parens
        amr_string += ')' * (open_count - close_count)
    elif close_count > open_count:
        # Add missing opening parens at start
        amr_string = '(' * (close_count - open_count) + amr_string

    return amr_string
```

## 🚀 RECOMMENDED: Giải Pháp 1 + Post-processing

1. **Fix preprocessor** (Giải pháp 1)
2. **Retrain** model với data mới
3. **Add post-processing** để handle edge cases

**Steps**:
```bash
# 1. Fix preprocessor
# Edit: src/preprocessor_mtup.py

# 2. Re-generate training data
python3 -c "
from src.preprocessor_mtup import MTUPAMRPreprocessor
from src.data_loader import AMRDataLoader
from pathlib import Path

loader = AMRDataLoader(Path('data'))
prep = MTUPAMRPreprocessor()

# Test one example
examples = loader.parse_amr_file(Path('data/train_amr_1.txt'))
ex = examples[0]

print('Sentence:', ex['sentence'])
print('\\nOriginal AMR:')
print(ex['amr'])

amr_no_vars = prep.remove_variables(ex['amr'])
print('\\nAMR without variables:')
print(amr_no_vars)

# Check parentheses
open_count = amr_no_vars.count('(')
close_count = amr_no_vars.count(')')
print(f'\\nParentheses: {open_count} open, {close_count} close')
if open_count == close_count:
    print('✓ Balanced!')
else:
    print('✗ NOT balanced!')
"

# 3. If test passes, retrain
bash RUN_FULL_TRAINING.sh
```

## ⏱️ Time Estimates

| Solution | Time | Success Rate |
|----------|------|--------------|
| **1. Fix + Retrain** | 6-8 hours | 95% |
| **2. New Format + Retrain** | 8-10 hours | 90% |
| **3. Post-processing only** | 30 min | 60% |

## 💡 Quick Test Before Retrain

```bash
# Test current preprocessing
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from preprocessor_mtup import MTUPAMRPreprocessor

prep = MTUPAMRPreprocessor()

test_cases = [
    "(a / ăn :agent (t / tôi) :patient (c / cơm))",
    "(b / bi_kịch :domain (c / chỗ :mod (đ / đó)))",
]

for amr in test_cases:
    no_vars = prep.remove_variables(amr)
    open_c = no_vars.count('(')
    close_c = no_vars.count(')')

    print(f"Input:  {amr}")
    print(f"Output: {no_vars}")
    print(f"Parens: {open_c} vs {close_c} {'✓' if open_c == close_c else '✗'}")
    print()
EOF
```

If ✗ appears → Need to fix!
