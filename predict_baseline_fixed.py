#!/usr/bin/env python3
"""
Predict using FIXED baseline model and evaluate

Usage:
    python predict_baseline_fixed.py --model models_archive/baseline_7b_fixed/final
"""

import sys
import torch
import argparse
from pathlib import Path
from typing import List, Dict

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))


def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned LoRA model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from pathlib import Path

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Checkpoint path: {model_path}")
    print()

    # Base model name
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer from base model (not saved in checkpoint)
    print(f"Loading tokenizer from base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Tokenizer loaded")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

    # Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter from checkpoint
    print(f"\nLoading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    print(f"✓ Model loaded (base + LoRA adapter)")
    print("=" * 70)
    print()

    return model, tokenizer


def load_test_sentences(test_file: str) -> List[str]:
    """Load test sentences from file"""
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    print(f"File: {test_file}")
    print()

    sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    print(f"✓ Loaded {len(sentences)} test sentences")
    print("=" * 70)
    print()

    return sentences


def predict_amr(model, tokenizer, sentence: str, prompt_template: str, config: dict) -> str:
    """Generate AMR for a single sentence"""
    # Create prompt
    prompt = prompt_template.format(sentence=sentence)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=256
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            top_k=config['top_k'],
            repetition_penalty=config['repetition_penalty'],
            do_sample=config['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # CRITICAL: Stop at EOS
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract AMR (remove prompt)
    amr = generated[len(prompt):].strip()

    # Remove EOS token if present
    if tokenizer.eos_token in amr:
        amr = amr.split(tokenizer.eos_token)[0].strip()

    # CRITICAL: Remove any explanation after AMR
    # AMR should end with balanced parentheses
    # If we see text after last ')', remove it
    lines = amr.split('\n')
    amr_lines = []
    found_amr_end = False

    for line in lines:
        if not found_amr_end:
            amr_lines.append(line)
            # Check if this line has closing parenthesis
            if ')' in line:
                # Check if parentheses are balanced IN ACCUMULATED TEXT (not original)
                accumulated = '\n'.join(amr_lines)
                open_count = accumulated.count('(')
                close_count = accumulated.count(')')
                if open_count == close_count and open_count > 0:
                    found_amr_end = True
        # Skip lines after AMR end (explanations)

    amr = '\n'.join(amr_lines).strip()

    return amr


def save_results(sentences: List[str], predictions: List[str], output_file: str):
    """Save predictions in #::snt format"""
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    print(f"Output: {output_file}")
    print()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (sent, amr) in enumerate(zip(sentences, predictions)):
            f.write(f"#::snt {sent}\n")
            f.write(f"{amr}\n")
            f.write("\n")

    print(f"✓ Saved {len(predictions)} predictions")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description='Predict with fixed baseline model')
    parser.add_argument('--model', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--test-file', type=str, default='data/public_test.txt', help='Test file')
    parser.add_argument('--output', type=str, default='evaluation_results/baseline_7b_fixed/predictions.txt', help='Output file')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt template and config
    from config_fixed import PROMPT_TEMPLATE, INFERENCE_CONFIG

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load test sentences
    sentences = load_test_sentences(args.test_file)

    # Generate predictions
    print("=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    print(f"Using inference config:")
    print(f"  Temperature: {INFERENCE_CONFIG['temperature']}")
    print(f"  Top-p: {INFERENCE_CONFIG['top_p']}")
    print(f"  Max tokens: {INFERENCE_CONFIG['max_new_tokens']}")
    print(f"  Repetition penalty: {INFERENCE_CONFIG['repetition_penalty']}")
    print()

    predictions = []
    errors = []

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] Processing: {sentence[:60]}...")

        try:
            amr = predict_amr(model, tokenizer, sentence, PROMPT_TEMPLATE, INFERENCE_CONFIG)
            predictions.append(amr)

            # Show first few predictions
            if i <= 3:
                print(f"  AMR: {amr[:100]}...")
                print()

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            errors.append((i, sentence, str(e)))
            # Add placeholder to maintain alignment
            predictions.append(f"(e / error\n    :sentence \"{sentence}\")")
            print()

    print("=" * 70)
    print()

    if errors:
        print("⚠️  ERRORS ENCOUNTERED:")
        print("=" * 70)
        for i, sent, err in errors:
            print(f"  #{i}: {sent[:50]}...")
            print(f"       Error: {err}")
        print("=" * 70)
        print()

    # Save results
    save_results(sentences, predictions, args.output)

    # Calculate SMATCH
    print("=" * 70)
    print("CALCULATING SMATCH")
    print("=" * 70)
    print()

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'smatch', '-f',
             args.output, 'data/public_test_ground_truth.txt',
             '--significant', '4'],
            capture_output=True,
            text=True,
            timeout=600
        )

        print(result.stdout)

        if result.stderr and 'Duplicate' not in result.stderr and 'Unmatched' not in result.stderr:
            print("Warnings:")
            print(result.stderr)

        # Extract F1 score
        for line in result.stdout.split('\n'):
            if 'F-score:' in line or 'Smatch' in line:
                print()
                print("=" * 70)
                print("FINAL SMATCH SCORE")
                print("=" * 70)
                print(line)
                print("=" * 70)

    except Exception as e:
        print(f"Could not calculate SMATCH: {e}")
        print()
        print("You can manually run:")
        print(f"  python -m smatch -f {args.output} data/public_test_ground_truth.txt --significant 4")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
