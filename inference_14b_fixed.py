"""
FIXED Inference for 14B - Match training format exactly
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print("="*70)
print("14B INFERENCE - FIXED VERSION")
print("="*70)

# Load model
print("\nLoading 14B model...")
model_path = "outputs/vlsp_amr_qwen_improved_v2/merged_16bit"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Model loaded!")

# Load test data
with open('data/public_test.txt', 'r') as f:
    test_sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"Total sentences: {len(test_sentences)}")

# EXACT TRAINING PROMPT FORMAT
PROMPT_TEMPLATE = """Bạn là một chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation):

Câu: {sentence}

AMR:"""

# Generate
predictions = []
print("\nGenerating predictions...")

for i, sentence in enumerate(tqdm(test_sentences)):
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,        # Increased
            min_new_tokens=10,         # Added minimum
            do_sample=False,           # Greedy
            num_beams=1,
            temperature=None,          # No temperature for greedy
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,    # Prevent loops!
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract AMR only
    if "AMR:" in response:
        # Get everything after last "AMR:"
        amr = response.split("AMR:")[-1].strip()
        # Remove any trailing prompt leakage
        if "Human:" in amr:
            amr = amr.split("Human:")[0].strip()
        if "Câu:" in amr:
            amr = amr.split("Câu:")[0].strip()
    else:
        # Fallback
        amr = response.split(prompt)[-1].strip()
    
    # Clean up
    amr = amr.strip()
    
    # Save with sentence
    predictions.append(f"# ::snt {sentence}\n{amr}")
    
    # Checkpoint every 50
    if (i + 1) % 50 == 0:
        checkpoint_file = f"outputs/checkpoint_14b_fixed_{i+1}.txt"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(predictions))
        print(f"\n✅ Checkpoint: {checkpoint_file}")

# Save final
output_file = "outputs/public_14b_predictions_FIXED.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(predictions))

print(f"\n{'='*70}")
print("INFERENCE COMPLETE!")
print(f"{'='*70}")
print(f"✅ Saved to: {output_file}")
print(f"✅ Total: {len(predictions)} AMRs")

# Show samples
print(f"\n{'='*70}")
print("SAMPLE OUTPUTS (first 2):")
print(f"{'='*70}")
for i, pred in enumerate(predictions[:2]):
    print(f"\n{i+1}.")
    print(pred[:200] + "..." if len(pred) > 200 else pred)
