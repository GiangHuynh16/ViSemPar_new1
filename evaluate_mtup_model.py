#!/usr/bin/env python3
"""
Evaluate MTUP Model on Test Data
Generates predictions and computes SMATCH scores
"""

import sys
import torch
import re
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, 'src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import AMRDataLoader

def fix_incomplete_amr(amr_string: str) -> str:
    """Fix common issues in model-generated AMR"""
    amr = amr_string.strip()

    # If doesn't start with '(', add it
    if amr and not amr.startswith('('):
        amr = '(' + amr

    # Count parentheses
    open_count = amr.count('(')
    close_count = amr.count(')')

    # Balance parentheses
    if open_count > close_count:
        amr += ')' * (open_count - close_count)
    elif close_count > open_count:
        amr = '(' * (close_count - open_count) + amr

    return amr

def load_model(checkpoint_path: str):
    """Load MTUP model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    print("✓ Model loaded successfully")
    return model, tokenizer

def generate_mtup_prediction(model, tokenizer, sentence: str, max_length=512):
    """Generate AMR using MTUP (2-task approach)"""

    # CRITICAL: Use SAME prompt format as training (v2_natural template)
    # The model was trained with Vietnamese prompts, NOT English!

    # Full prompt for both tasks (as seen in training)
    full_prompt = f"""### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Greedy decoding - completely deterministic
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the final AMR (after "Bước 2" section)
    # Model should generate complete output including both steps
    if "## Bước 2" in result:
        # Extract everything after the Bước 2 header
        parts = result.split("## Bước 2")[1]
        # Look for "AMR hoàn chỉnh:" section
        if "AMR hoàn chỉnh:" in parts:
            final_amr = parts.split("AMR hoàn chỉnh:")[-1].strip()
        else:
            # Fallback: take everything after Bước 2
            final_amr = parts.strip()
    else:
        # Fallback: try to extract any AMR-like structure
        final_amr = result.strip()

    # Clean up: Remove any prompt text that leaked into output
    # Only keep the AMR structure (starting with '(')
    if '(' in final_amr:
        # Find first '(' and take everything from there
        first_paren = final_amr.index('(')
        final_amr = final_amr[first_paren:].strip()

    return final_amr

def evaluate_on_test_data(model, tokenizer, test_file: Path, max_samples=None):
    """Evaluate model on test data"""

    print(f"\nLoading test data from {test_file}...")
    loader = AMRDataLoader(test_file.parent.parent)

    try:
        examples = loader.parse_amr_file(test_file)
        if max_samples:
            examples = examples[:max_samples]
        print(f"✓ Loaded {len(examples)} test examples")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    ground_truth = []

    for i, example in enumerate(tqdm(examples, desc="Generating")):
        sentence = example['sentence']
        gold_amr = example['amr']

        try:
            pred_amr = generate_mtup_prediction(model, tokenizer, sentence)
            predictions.append(pred_amr)
            ground_truth.append(gold_amr)
        except Exception as e:
            print(f"\n⚠️  Error generating prediction for example {i+1}: {e}")
            predictions.append("(e / error)")
            ground_truth.append(gold_amr)

    print(f"✓ Generated {len(predictions)} predictions")

    # Compute SMATCH scores
    print("\nComputing SMATCH scores...")

    try:
        import smatch
    except ImportError:
        print("✗ SMATCH not installed. Install with: pip install smatch")
        return None

    total_p, total_r, total_f = 0, 0, 0
    valid = 0
    errors = []

    for i, (pred, gold) in enumerate(tqdm(list(zip(predictions, ground_truth)), desc="Evaluating")):
        try:
            # Linearize
            pred_linear = ' '.join(pred.split())
            gold_linear = ' '.join(gold.split())

            # Compute SMATCH
            best, test, gold_t = smatch.get_amr_match(pred_linear, gold_linear)

            if test > 0 and gold_t > 0:
                precision = best / test
                recall = best / gold_t
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                total_p += precision
                total_r += recall
                total_f += f1
                valid += 1
        except Exception as e:
            errors.append((i+1, str(e)))

    # Results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    if valid > 0:
        avg_p = total_p / valid
        avg_r = total_r / valid
        avg_f = total_f / valid

        print(f"\nProcessed: {valid}/{len(predictions)} examples")
        print(f"Errors:    {len(errors)}")

        print(f"\n{'='*80}")
        print("SMATCH SCORES")
        print(f"{'='*80}")
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f:.4f}")
        print(f"{'='*80}")

        return {
            'precision': avg_p,
            'recall': avg_r,
            'f1': avg_f,
            'valid': valid,
            'total': len(predictions),
            'errors': len(errors)
        }
    else:
        print("✗ No valid evaluations")
        return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate MTUP model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (e.g., outputs/checkpoints_mtup/checkpoint-250)')
    parser.add_argument('--test-file', type=str, default='data/public_test_ground_truth.txt',
                        help='Test data file with ground truth')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save predictions (JSON)')

    args = parser.parse_args()

    print("="*80)
    print("MTUP MODEL EVALUATION")
    print("="*80)

    # Check files
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return

    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        return

    # Load model
    model, tokenizer = load_model(str(checkpoint_path))

    # Evaluate
    results = evaluate_on_test_data(model, tokenizer, test_file, args.max_samples)

    if results:
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✓ Results saved to {output_path}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
