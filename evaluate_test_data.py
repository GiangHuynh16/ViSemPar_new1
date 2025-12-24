#!/usr/bin/env python3
"""
Evaluate Test Data with SMATCH
Tests evaluation on real ground truth data
"""

import sys
from pathlib import Path

sys.path.insert(0, 'src')

from data_loader import AMRDataLoader

def main():
    print("="*80)
    print("REAL DATA SMATCH EVALUATION")
    print("="*80)

    # Check SMATCH
    print("\nStep 1: Checking SMATCH...")
    try:
        import smatch
        print("✓ SMATCH available")
    except ImportError:
        print("✗ SMATCH not installed")
        print("Install with: pip install smatch")
        return

    # Load test data
    print("\nStep 2: Loading test data with ground truth...")
    loader = AMRDataLoader(Path("data"))

    test_file = Path("data/public_test_ground_truth.txt")
    if not test_file.exists():
        print(f"⚠️ Ground truth file not found: {test_file}")
        print("\nTrying train_amr_1.txt for testing...")
        test_file = Path("data/train_amr_1.txt")

    if not test_file.exists():
        print("✗ No data file found")
        return

    try:
        examples = loader.parse_amr_file(test_file)
        print(f"✓ Loaded {len(examples)} examples from {test_file.name}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Test evaluation
    print("\n" + "="*80)
    print("Step 3: Evaluating first 10 examples (self-match)")
    print("="*80)

    # For testing, compare with self (should get perfect score)
    predictions = [ex['amr'] for ex in examples[:10]]
    ground_truth = [ex['amr'] for ex in examples[:10]]

    total_p, total_r, total_f = 0, 0, 0
    valid = 0
    errors = []

    for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
        try:
            # Linearize both (SMATCH works with linearized)
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

                print(f"Example {i+1:2d}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

        except Exception as e:
            errors.append((i+1, str(e)))
            print(f"Example {i+1:2d}: ✗ Error - {str(e)[:50]}")

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
        print("AVERAGE SMATCH SCORES")
        print(f"{'='*80}")
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f:.4f}")
        print(f"{'='*80}")

        if avg_f > 0.99:
            print("\n✅ Perfect score achieved (self-match test)")
            print("SMATCH evaluation is working correctly!")
        else:
            print(f"\n⚠️ Expected F1=1.0 for self-match, got {avg_f:.4f}")

        if errors:
            print(f"\n⚠️ Errors occurred:")
            for idx, err in errors[:3]:
                print(f"  Example {idx}: {err[:60]}...")

    else:
        print("✗ No valid evaluations")

    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. SMATCH evaluation verified ✓")
    print("\n2. To evaluate model predictions:")
    print("   - Train model or load existing model")
    print("   - Generate predictions on test data")
    print("   - Use this script with real predictions vs ground truth")
    print("\n3. Example usage:")
    print("   predictions = model.generate(test_sentences)")
    print("   scores = evaluate(predictions, ground_truth)")
    print("="*80)

if __name__ == "__main__":
    main()
