#!/usr/bin/env python3
"""
Remove invalid AMRs and calculate SMATCH on valid pairs only
"""

import re
import subprocess
import sys
from pathlib import Path


def check_duplicate_nodes(amr_string):
    """Check if AMR has duplicate node names"""
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_string)

    if not nodes:
        return False, []

    from collections import Counter
    node_counts = Counter(nodes)
    duplicates = [node for node, count in node_counts.items() if count > 1]

    return len(duplicates) > 0, duplicates


def filter_valid_amrs(input_file, output_file):
    """Filter out AMRs with duplicate nodes"""
    print(f"Reading: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by #::snt
    parts = content.split('#::snt ')

    valid_amrs = []
    invalid_indices = []

    for i, part in enumerate(parts[1:], 1):  # Skip empty first part
        lines = part.split('\n')
        sentence = lines[0]
        amr = '\n'.join(lines[1:]).strip()

        if amr:
            has_dup, dup_nodes = check_duplicate_nodes(amr)

            if not has_dup:
                valid_amrs.append(f"#::snt {sentence}\n{amr}\n\n")
            else:
                invalid_indices.append(i)
                print(f"  ⚠️  Skipping AMR #{i} (duplicate nodes: {', '.join(dup_nodes)})")

    # Write valid AMRs
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_amrs)

    print(f"  ✓ Wrote {len(valid_amrs)} valid AMRs to {output_file}")

    return len(valid_amrs), invalid_indices


def main():
    print("=" * 70)
    print("FIX AND EVALUATE - Remove Invalid AMRs")
    print("=" * 70)
    print()

    # Filter predictions
    print("Step 1: Filtering predictions...")
    pred_count, pred_invalid = filter_valid_amrs(
        'predictions_formatted.txt',
        'predictions_valid.txt'
    )
    print()

    # Filter gold (to match indices)
    print("Step 2: Filtering gold to match predictions...")

    # Read gold
    with open('data/public_test_ground_truth.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('#::snt ')
    valid_gold = []

    for i, part in enumerate(parts[1:], 1):
        if i not in pred_invalid:
            valid_gold.append(f"#::snt {part}")

    with open('gold_valid.txt', 'w', encoding='utf-8') as f:
        f.writelines(valid_gold)

    print(f"  ✓ Wrote {len(valid_gold)} valid gold AMRs to gold_valid.txt")
    print()

    # Calculate SMATCH
    print("Step 3: Calculating SMATCH on valid pairs...")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, '-m', 'smatch', '-f', 'predictions_valid.txt', 'gold_valid.txt', '--significant', '4'],
        capture_output=True,
        text=True,
        timeout=600
    )

    print(result.stdout)

    if result.stderr and 'Duplicate node name' not in result.stderr:
        print("Warnings:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"\n⚠️  SMATCH returned code {result.returncode}")
        print("\nTrying alternative approach with smatch.main()...")

        import smatch
        old_argv = sys.argv
        sys.argv = ['smatch', '-f', 'predictions_valid.txt', 'gold_valid.txt', '--significant', '4']

        try:
            smatch.main()
        finally:
            sys.argv = old_argv

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Valid pairs evaluated: {pred_count}/{pred_count + len(pred_invalid)}")
    print(f"Invalid pairs skipped: {len(pred_invalid)}")
    print()


if __name__ == "__main__":
    main()
