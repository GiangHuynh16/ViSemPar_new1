#!/usr/bin/env python3
"""
Diagnose tokenization issue in instruction masking
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / 'config'))

from config_fixed import MODEL_NAME, PROMPT_TEMPLATE

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test sentence and AMR
sentence = "bi kịch là ở chỗ đó !"
amr = """(b / bi_kịch
    :domain(c / chỗ
        :mod(đ / đó)))"""

prompt = PROMPT_TEMPLATE.format(sentence=sentence)
full_text = prompt + amr + tokenizer.eos_token

print("=" * 70)
print("TOKENIZATION DIAGNOSIS")
print("=" * 70)
print()

print("Sentence:", sentence)
print()
print("Prompt length (chars):", len(prompt))
print("AMR length (chars):", len(amr))
print("Full text length (chars):", len(full_text))
print()

# Method 1: Current implementation (BUGGY)
print("=" * 70)
print("METHOD 1: Separate tokenization (CURRENT - BUGGY)")
print("=" * 70)

# Tokenize full text
full_encoding = tokenizer(
    full_text,
    truncation=True,
    max_length=512,
    padding='max_length',
    return_tensors='pt'
)

# Tokenize prompt separately
prompt_encoding = tokenizer(
    prompt,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

prompt_length = len(prompt_encoding['input_ids'][0])
full_length = len(full_encoding['input_ids'][0])

print(f"Prompt tokens (separate): {prompt_length}")
print(f"Full text tokens: {full_length}")
print()

# Decode tokens to see what gets masked
full_ids = full_encoding['input_ids'][0]
print("First 20 tokens (full text):")
for i in range(min(20, len(full_ids))):
    token = tokenizer.decode([full_ids[i]])
    masked = "MASKED" if i < prompt_length else "TRAINED"
    print(f"  {i:3d}: '{token}' [{masked}]")
print()

# Check the boundary
print(f"Tokens around boundary (index {prompt_length}):")
for i in range(max(0, prompt_length-5), min(len(full_ids), prompt_length+10)):
    token = tokenizer.decode([full_ids[i]])
    masked = "MASKED" if i < prompt_length else "TRAINED"
    marker = " <-- BOUNDARY" if i == prompt_length else ""
    print(f"  {i:3d}: '{token}' [{masked}]{marker}")
print()

# Method 2: Correct implementation
print("=" * 70)
print("METHOD 2: Find prompt end in combined text (CORRECT)")
print("=" * 70)

# Encode prompt and AMR separately to find exact boundary
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = tokenizer.encode(amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

print(f"Prompt tokens (encode): {len(prompt_ids)}")
print(f"AMR tokens (encode): {len(amr_ids)}")
print(f"EOS tokens (encode): {len(eos_ids)}")
print()

# The correct mask length should account for special tokens
# when full text is tokenized
full_ids_no_padding = [id for id in full_ids if id != tokenizer.pad_token_id]

print(f"Full text tokens (no padding): {len(full_ids_no_padding)}")
print(f"Expected total: {len(prompt_ids) + len(amr_ids) + len(eos_ids)}")
print()

# Key insight: Check if tokenizing separately gives same result as together
combined_ids = tokenizer.encode(full_text, add_special_tokens=False)
separate_ids = prompt_ids + amr_ids + eos_ids

print("CRITICAL TEST: Are tokenizations equal?")
print(f"  Combined tokenization length: {len(combined_ids)}")
print(f"  Separate concatenation length: {len(separate_ids)}")
print(f"  Equal: {combined_ids == separate_ids}")
print()

if combined_ids != separate_ids:
    print("⚠️  TOKENIZATION MISMATCH DETECTED!")
    print()
    print("This means instruction masking is INCORRECT!")
    print("The prompt length calculated separately doesn't match")
    print("the actual prompt position in the combined text.")
    print()

    # Find first difference
    for i in range(min(len(combined_ids), len(separate_ids))):
        if i >= len(combined_ids) or i >= len(separate_ids) or combined_ids[i] != separate_ids[i]:
            print(f"First difference at index {i}:")
            if i < len(combined_ids):
                print(f"  Combined: {tokenizer.decode([combined_ids[i]])}")
            if i < len(separate_ids):
                print(f"  Separate: {tokenizer.decode([separate_ids[i]])}")
            break

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if combined_ids != separate_ids:
    print("❌ Current instruction masking method is BROKEN")
    print()
    print("FIX: Instead of tokenizing prompt separately, find the")
    print("prompt end position by encoding and searching in combined text:")
    print()
    print("  prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)")
    print("  full_ids = tokenizer.encode(full_text, add_special_tokens=False)")
    print("  prompt_end = len(prompt_ids)")
    print("  labels[:prompt_end] = -100")
else:
    print("✅ Tokenization is consistent - instruction masking should work")

print("=" * 70)
