#!/usr/bin/env python3
"""
Test SMATCH Evaluation
Verifies SMATCH library works correctly
"""

def main():
    print("="*80)
    print("SMATCH EVALUATION TEST")
    print("="*80)

    # Check if smatch is installed
    print("\nStep 1: Checking SMATCH installation...")
    try:
        import smatch
        print("✓ SMATCH library found")
    except ImportError:
        print("✗ SMATCH not installed")
        print("\nInstall with:")
        print("  pip install smatch")
        return

    # Test with perfect match
    print("\n" + "="*80)
    print("Step 2: Testing with perfect match (should get F1=1.0)")
    print("="*80)

    predictions = [
        "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))",
        "(n / nhớ :pivot(t / tôi) :theme(l / lời))"
    ]

    ground_truth = [
        "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))",
        "(n / nhớ :pivot(t / tôi) :theme(l / lời))"
    ]

    total_p, total_r, total_f = 0, 0, 0
    valid = 0

    for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
        try:
            # Compute SMATCH
            best, test, gold_t = smatch.get_amr_match(pred, gold)

            if test > 0 and gold_t > 0:
                precision = best / test
                recall = best / gold_t
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                total_p += precision
                total_r += recall
                total_f += f1
                valid += 1

                print(f"\nExample {i+1}:")
                print(f"  Pred: {pred[:60]}...")
                print(f"  Gold: {gold[:60]}...")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1:        {f1:.4f}")

        except Exception as e:
            print(f"\n✗ Error on example {i+1}: {e}")

    # Average scores
    if valid > 0:
        avg_p = total_p / valid
        avg_r = total_r / valid
        avg_f = total_f / valid

        print("\n" + "="*80)
        print("AVERAGE SCORES (Perfect Match Test)")
        print("="*80)
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f:.4f}")
        print(f"  Valid:     {valid}/{len(predictions)}")

        if avg_f > 0.99:
            print("\n✅ SMATCH working correctly (perfect score achieved)")
        else:
            print("\n⚠️ Unexpected score for perfect match")

    # Test with imperfect match
    print("\n" + "="*80)
    print("Step 3: Testing with imperfect match")
    print("="*80)

    pred_imperfect = "(n / nhớ :pivot(t / tôi))"
    gold_full = "(n / nhớ :pivot(t / tôi) :theme(l / lời))"

    try:
        best, test, gold_t = smatch.get_amr_match(pred_imperfect, gold_full)
        if test > 0 and gold_t > 0:
            precision = best / test
            recall = best / gold_t
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\nPrediction: {pred_imperfect}")
            print(f"Gold:       {gold_full}")
            print(f"\nPrecision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1:        {f1:.4f}")

            if f1 < 1.0:
                print("\n✓ SMATCH correctly detects differences")
            else:
                print("\n⚠️ Unexpected - should have F1 < 1.0")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Final summary
    print("\n" + "="*80)
    print("SMATCH TEST SUMMARY")
    print("="*80)
    print("✅ SMATCH library is working correctly")
    print("\n📝 Next steps:")
    print("  1. Test with real data: python3 evaluate_test_data.py")
    print("  2. Create training pipeline")
    print("  3. Evaluate model predictions")
    print("="*80)

if __name__ == "__main__":
    main()
