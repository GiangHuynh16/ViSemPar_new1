#!/usr/bin/env python3
"""
Quick test to verify instruction masking fix works correctly
This should be run BEFORE retraining to confirm the fix
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'config'))

from transformers import AutoTokenizer
from config_fixed import MODEL_NAME, PROMPT_TEMPLATE

print("=" * 70)
print("TESTING INSTRUCTION MASKING FIX")
print("=" * 70)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Tokenizer loaded: {MODEL_NAME}")
print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print()

# Test data
test_sentence = "bi kịch là ở chỗ đó !"
test_amr = """(b / bi_kịch
    :domain(c / chỗ
        :mod(đ / đó)))"""

print("Test sentence:", test_sentence)
print()

# Create prompt
prompt = PROMPT_TEMPLATE.format(sentence=test_sentence)
full_text = prompt + test_amr + tokenizer.eos_token

print("Prompt length (chars):", len(prompt))
print("AMR length (chars):", len(test_amr))
print("Full text length (chars):", len(full_text))
print()

# OLD METHOD (BUGGY)
print("=" * 70)
print("OLD METHOD (BUGGY) - Separate tokenization")
print("=" * 70)

full_encoding = tokenizer(
    full_text,
    truncation=True,
    max_length=512,
    padding='max_length',
    return_tensors='pt'
)

prompt_encoding = tokenizer(
    prompt,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

old_prompt_length = len(prompt_encoding['input_ids'][0])
print(f"Prompt tokens (separate): {old_prompt_length}")
print(f"Full text tokens: {len(full_encoding['input_ids'][0])}")
print()

# NEW METHOD (FIXED)
print("=" * 70)
print("NEW METHOD (FIXED) - Encode without special tokens")
print("=" * 70)

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = tokenizer.encode(test_amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

new_prompt_length = len(prompt_ids)
print(f"Prompt tokens (encode): {new_prompt_length}")
print(f"AMR tokens: {len(amr_ids)}")
print(f"EOS tokens: {len(eos_ids)}")
print(f"Total: {new_prompt_length + len(amr_ids) + len(eos_ids)}")
print()

# Compare
print("=" * 70)
print("COMPARISON")
print("=" * 70)

full_ids = full_encoding['input_ids'][0]
full_ids_no_pad = [id.item() for id in full_ids if id != tokenizer.pad_token_id]

print(f"Old method prompt length: {old_prompt_length}")
print(f"New method prompt length: {new_prompt_length}")
print(f"Difference: {abs(old_prompt_length - new_prompt_length)} tokens")
print()

if old_prompt_length != new_prompt_length:
    print("⚠️  TOKENIZATION MISMATCH DETECTED!")
    print()
    print("This confirms the bug exists:")
    print("  - Old method would mask WRONG positions")
    print("  - Model would learn instruction instead of AMR")
    print()
    print("✅ NEW METHOD FIXES THIS!")
    print()
else:
    print("ℹ️  No mismatch in this example")
    print("   (But mismatch can occur depending on text)")
    print()

# Verify the fix works
print("=" * 70)
print("VERIFYING FIX")
print("=" * 70)

# Build sequence the NEW way
full_ids_new = prompt_ids + amr_ids + eos_ids
max_length = 512
padding_length = max_length - len(full_ids_new)
input_ids = full_ids_new + [tokenizer.pad_token_id] * padding_length
labels = input_ids.copy()

# Mask instruction
prompt_end = len(prompt_ids)
for i in range(prompt_end):
    labels[i] = -100

# Mask padding
for i in range(len(full_ids_new), max_length):
    labels[i] = -100

# Count what's masked vs trained
masked_count = sum(1 for x in labels if x == -100)
trained_count = sum(1 for x in labels if x != -100)

print(f"Total tokens: {max_length}")
print(f"Masked (instruction + padding): {masked_count}")
print(f"Trained (AMR + EOS): {trained_count}")
print()

print("Breakdown:")
print(f"  Instruction (masked): {prompt_end} tokens")
print(f"  AMR (trained): {len(amr_ids)} tokens")
print(f"  EOS (trained): {len(eos_ids)} tokens")
print(f"  Padding (masked): {padding_length} tokens")
print()

# Verify the trained part is actually AMR
trained_ids = [input_ids[i] for i in range(len(input_ids)) if labels[i] != -100]
trained_text = tokenizer.decode(trained_ids)

print("Text that will be TRAINED on:")
print("-" * 70)
print(trained_text)
print("-" * 70)
print()

if test_amr in trained_text or trained_text.strip().startswith("("):
    print("✅ CORRECT: Training on AMR output!")
    print()
    print("=" * 70)
    print("✅ INSTRUCTION MASKING FIX VERIFIED")
    print("=" * 70)
    print()
    print("The fix is working correctly. Safe to retrain!")
    sys.exit(0)
else:
    print("❌ ERROR: Not training on AMR!")
    print()
    print("Expected to see AMR in trained text, but got something else.")
    print("Please check the implementation.")
    sys.exit(1)
