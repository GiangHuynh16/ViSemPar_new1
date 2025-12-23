"""
Optimized inference for 14B model - Full 150 samples
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
sys.path.insert(0, 'src')

print("="*70)
print("INFERENCE - 14B MODEL - FULL PUBLIC TEST")
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
print("✅ Model loaded!")

# Load test data
with open('data/public_test.txt', 'r') as f:
    test_sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"Total sentences: {len(test_sentences)}")

# Simplified prompt
PROMPT = "Chuyển câu sau sang AMR:\n\nCâu: {sentence}\n\nAMR:"

# Generate predictions
predictions = []
print("\nGenerating predictions...")

for i, sentence in enumerate(tqdm(test_sentences)):
    prompt = PROMPT.format(sentence=sentence)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Reduced from 512
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract AMR
    if "AMR:" in response:
        amr = response.split("AMR:")[-1].strip()
    else:
        amr = response.split(prompt)[-1].strip()
    
    predictions.append(f"# ::snt {sentence}\n{amr}")
    
    # Save checkpoint every 50 samples
    if (i + 1) % 50 == 0:
        checkpoint_file = f"outputs/checkpoint_14b_{i+1}.txt"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(predictions))
        print(f"\n✅ Checkpoint saved: {checkpoint_file}")

# Save final predictions
output_file = "outputs/public_14b_predictions_full.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(predictions))

print(f"\n{'='*70}")
print("INFERENCE COMPLETE!")
print(f"{'='*70}")
print(f"✅ Predictions saved to: {output_file}")
print(f"✅ Total: {len(predictions)} AMRs")

# Show samples
print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS (first 2)")
print(f"{'='*70}")
for i, pred in enumerate(predictions[:2]):
    print(f"\n{i+1}.")
    print(pred[:150] + "..." if len(pred) > 150 else pred)
