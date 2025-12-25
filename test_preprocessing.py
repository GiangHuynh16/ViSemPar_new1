#!/usr/bin/env python3
"""
Test MTUP preprocessing to identify issues
"""

import sys
sys.path.insert(0, 'src')

from preprocessor_mtup import MTUPAMRPreprocessor
from data_loader import AMRDataLoader
from pathlib import Path

print("="*80)
print("TESTING MTUP PREPROCESSING")
print("="*80)

# Initialize
prep = MTUPAMRPreprocessor()
loader = AMRDataLoader(Path('data'))

# Load first 5 examples
print("\nLoading examples from train_amr_1.txt...")
examples = loader.parse_amr_file(Path('data/train_amr_1.txt'))[:5]
print(f"✓ Loaded {len(examples)} examples")

print("\n" + "="*80)
print("TEST 1: Check parentheses balance after remove_variables()")
print("="*80)

errors = []
for i, ex in enumerate(examples, 1):
    amr_original = ex['amr']
    amr_no_vars = prep.remove_variables(amr_original)

    open_count = amr_no_vars.count('(')
    close_count = amr_no_vars.count(')')
    balanced = open_count == close_count

    status = "✓" if balanced else "✗"

    print(f"\nExample {i}: {status}")
    print(f"Sentence: {ex['sentence'][:60]}...")
    print(f"Original: {amr_original[:80]}...")
    print(f"No vars:  {amr_no_vars[:80]}...")
    print(f"Parens:   {open_count} open, {close_count} close")

    if not balanced:
        errors.append((i, open_count - close_count))

print("\n" + "="*80)
print("TEST 2: Full MTUP preprocessing")
print("="*80)

for i, ex in enumerate(examples[:3], 1):
    print(f"\nExample {i}:")
    print(f"Sentence: {ex['sentence']}")

    mtup_text = prep.preprocess_for_mtup(
        sentence=ex['sentence'],
        amr_with_vars=ex['amr']
    )

    print(f"\nMTUP Format:")
    print(mtup_text)
    print("-" * 80)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if errors:
    print(f"\n❌ Found {len(errors)} examples with unbalanced parentheses:")
    for idx, diff in errors:
        print(f"   Example {idx}: {diff:+d} difference")

    print("\n⚠️  THIS IS THE ROOT CAUSE!")
    print("Model learns from unbalanced AMRs → generates invalid output")
    print("\nFix: Update remove_variables() in src/preprocessor_mtup.py")
else:
    print("\n✅ All examples have balanced parentheses")
    print("Preprocessing is OK!")

print("\n" + "="*80)
