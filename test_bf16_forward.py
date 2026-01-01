#!/usr/bin/env python3
"""
Test if BF16 is working correctly with the current setup
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import sys

print("=" * 70)
print("BF16 FORWARD PASS TEST")
print("=" * 70)
print()

# Check BF16 support
print("Step 1: Checking BF16 support...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    # Check if BF16 is supported on this GPU
    if hasattr(torch.cuda, 'is_bf16_supported'):
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print(f"  BF16 support check: Not available in this PyTorch version")
print()

# Load tokenizer
print("Step 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  ✓ Tokenizer loaded")
print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print()

# Load model with BF16
print("Step 3: Loading model with BF16...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    trust_remote_code=True
)
print(f"  ✓ Model loaded")
print(f"  Model dtype: {model.dtype}")
print()

# Apply LoRA
print("Step 4: Applying LoRA...")
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
print(f"  ✓ LoRA applied")
print(f"  Trainable params: {model.print_trainable_parameters()}")
print()

# Create test input
print("Step 5: Creating test input...")
test_text = "Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR.\n\nCâu: Chủ tịch nước gặp đại sứ.\n\nAMR:"
encoding = tokenizer(test_text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
input_ids = encoding['input_ids'].cuda()
attention_mask = encoding['attention_mask'].cuda()
labels = input_ids.clone()
# Mask padding tokens
labels[labels == tokenizer.pad_token_id] = -100
print(f"  ✓ Test input created")
print(f"  Input shape: {input_ids.shape}")
print(f"  Non-padding tokens: {(labels != -100).sum().item()}")
print()

# Test forward pass WITHOUT BF16 autocast first
print("Step 6: Testing forward pass WITHOUT autocast...")
try:
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss_no_autocast = outputs.loss
    print(f"  ✓ Forward pass successful")
    print(f"  Loss: {loss_no_autocast.item():.6f}")
    print(f"  Loss dtype: {loss_no_autocast.dtype}")
    print(f"  Is NaN: {torch.isnan(loss_no_autocast).item()}")
    print(f"  Is Zero: {loss_no_autocast.item() == 0.0}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)
print()

# Test forward pass WITH BF16 autocast
print("Step 7: Testing forward pass WITH BF16 autocast...")
try:
    model.train()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_with_autocast = outputs.loss
    print(f"  ✓ Forward pass successful")
    print(f"  Loss: {loss_with_autocast.item():.6f}")
    print(f"  Loss dtype: {loss_with_autocast.dtype}")
    print(f"  Is NaN: {torch.isnan(loss_with_autocast).item()}")
    print(f"  Is Zero: {loss_with_autocast.item() == 0.0}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)
print()

# Test backward pass
print("Step 8: Testing backward pass with gradient checkpointing...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    print(f"  Loss before backward: {loss.item():.6f}")

    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Gradient norm: {grad_norm.item():.6f}")
    print(f"  Is grad NaN: {torch.isnan(grad_norm).item()}")

    # Check if any gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            if torch.isnan(param.grad).any():
                print(f"  ✗ NaN gradient in: {name}")
            break

    if has_grad:
        print(f"  ✓ Gradients computed successfully")
    else:
        print(f"  ✗ No gradients found!")

except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
if torch.isnan(loss).item():
    print("❌ FAILED: Loss is NaN with BF16 + gradient checkpointing")
    sys.exit(1)
elif loss.item() == 0.0:
    print("⚠️  WARNING: Loss is exactly 0.0 (may indicate issue)")
    sys.exit(1)
else:
    print("✅ SUCCESS: BF16 working correctly")
    print(f"   Final loss: {loss.item():.6f}")
    print(f"   Final grad_norm: {grad_norm.item():.6f}")
print()
