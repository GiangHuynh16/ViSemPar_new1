#!/usr/bin/env python3
"""
Test what the model actually generates
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/checkpoints_mtup/mtup_full_training_final"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

print("\n" + "="*80)
print("TEST 1: Task 1 - Structure Generation")
print("="*80)

prompt1 = """Sentence: Tôi ăn cơm

Task 1: Generate AMR structure without variables.
Output:"""

print(f"\nPrompt:\n{prompt1}")

inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=300,
    temperature=0.1,  # Lower temperature
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nFull Output:\n{result}")
print("\n" + "="*80)

# Extract only new generation
if "Output:" in result:
    generated = result.split("Output:")[-1].strip()
    print(f"\nGenerated Part Only:\n{generated}")
else:
    print(f"\nGenerated:\n{result}")

print("\n" + "="*80)
print("TEST 2: Direct AMR Request")
print("="*80)

prompt2 = """Generate AMR for: Tôi ăn cơm

("""

print(f"\nPrompt:\n{prompt2}")

inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=200,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nOutput:\n{result}")

print("\n" + "="*80)
