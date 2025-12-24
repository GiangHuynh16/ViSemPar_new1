#!/usr/bin/env python3
"""
Quick MTUP Data Preparation Test
Verifies MTUP preprocessing works on server
"""

import sys
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'config')

from data_loader import AMRDataLoader
from preprocessor_mtup import MTUPAMRPreprocessor
from config_mtup import DATA_DIR, MTUP_CONFIG

def main():
    print("="*80)
    print("MTUP DATA PREPARATION TEST")
    print("="*80)

    # Load data
    print("\nStep 1: Loading data...")
    try:
        loader = AMRDataLoader(DATA_DIR)
        examples = loader.parse_amr_file(DATA_DIR / "train_amr_1.txt")
        print(f"✓ Loaded {len(examples)} examples from train_amr_1.txt")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Initialize preprocessor
    print("\nStep 2: Initializing MTUP preprocessor...")
    try:
        preprocessor = MTUPAMRPreprocessor(config=MTUP_CONFIG)
        print(f"✓ Preprocessor initialized")
        print(f"  Template: {MTUP_CONFIG['template_name']}")
        print(f"  Graph format: {MTUP_CONFIG['use_graph_format']}")
    except Exception as e:
        print(f"✗ Error initializing preprocessor: {e}")
        return

    # Process examples
    print("\n" + "="*80)
    print("Step 3: Processing examples with MTUP format...")
    print("="*80)

    processed = []
    errors = 0

    for i, ex in enumerate(examples[:10]):  # Test first 10
        try:
            mtup_text = preprocessor.preprocess_for_mtup(
                sentence=ex['sentence'],
                amr_with_vars=ex['amr']
            )
            processed.append(mtup_text)

            if i == 0:
                print(f"\n{'='*80}")
                print("EXAMPLE 1 - FULL OUTPUT:")
                print(f"{'='*80}")
                print(mtup_text)
                print(f"{'='*80}")

            print(f"✓ Example {i+1}/10 processed ({len(mtup_text)} chars)")

        except Exception as e:
            errors += 1
            print(f"✗ Error on example {i+1}: {e}")

    # Stats
    print("\n" + "="*80)
    print("PROCESSING STATISTICS")
    print("="*80)

    stats = preprocessor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n  Processed successfully: {len(processed)}/10")
    print(f"  Errors: {errors}/10")

    # Token estimation
    if processed:
        avg_length = sum(len(p) for p in processed) / len(processed)
        avg_tokens = avg_length / 4  # Rough estimate: 1 token ≈ 4 chars
        print(f"\n  Average text length: {avg_length:.0f} chars")
        print(f"  Estimated tokens: ~{avg_tokens:.0f} tokens")

    # Validation
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    if processed:
        sample = processed[0]
        checks = {
            'Has input section': 'Câu cần phân tích:' in sample,
            'Has Task 1': 'Bước 1' in sample,
            'Has Task 2': 'Bước 2' in sample,
            'Has instructions': 'Hướng dẫn:' in sample,
        }

        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")

        all_passed = all(checks.values())
        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL CHECKS PASSED - MTUP PREPROCESSING READY!")
        else:
            print("⚠️ SOME CHECKS FAILED - Review configuration")
        print("="*80)

        if all_passed:
            print("\n📝 Next steps:")
            print("  1. Verify SMATCH evaluation: python3 test_smatch.py")
            print("  2. Create training script: train_mtup.py")
            print("  3. Start training!")

    else:
        print("✗ No examples processed successfully")

if __name__ == "__main__":
    main()
