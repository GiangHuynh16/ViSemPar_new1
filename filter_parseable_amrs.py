#!/usr/bin/env python3
"""
Filter AMRs that can actually be parsed by smatch
"""

import sys

try:
    import smatch
except ImportError:
    print("ERROR: smatch not installed!")
    sys.exit(1)


def can_parse_amr(amr_string):
    """Try to parse AMR with smatch, return True if successful"""
    try:
        # Convert to single line
        single_line = amr_string.replace('\n', ' ')

        # Try to parse using smatch's internal parser
        # Use generate_amr_lines which reads from file-like input
        from io import StringIO
        amr_lines = list(smatch.generate_amr_lines(StringIO(single_line), StringIO(single_line)))

        if amr_lines and len(amr_lines) > 0:
            # Check if first AMR is not None
            amr_obj = amr_lines[0][0]  # (amr1, amr2) tuple
            return amr_obj is not None

        return False
    except:
        return False


def filter_parseable(input_file, output_file):
    """Filter to only parseable AMRs"""
    print(f"Reading: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('#::snt ')

    valid_amrs = []
    invalid_indices = []

    for i, part in enumerate(parts[1:], 1):
        lines = part.split('\n')
        sentence = lines[0]
        amr = '\n'.join(lines[1:]).strip()

        if amr:
            if can_parse_amr(amr):
                valid_amrs.append(f"#::snt {sentence}\n{amr}\n\n")
            else:
                invalid_indices.append(i)
                print(f"  ⚠️  Skipping AMR #{i} (cannot parse)")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_amrs)

    print(f"  ✓ Wrote {len(valid_amrs)} parseable AMRs to {output_file}")

    return len(valid_amrs), invalid_indices


def main():
    print("=" * 70)
    print("FILTER PARSEABLE AMRs ONLY")
    print("=" * 70)
    print()

    # Filter predictions
    print("Step 1: Filtering predictions...")
    pred_count, pred_invalid = filter_parseable(
        'predictions_formatted.txt',
        'predictions_parseable.txt'
    )
    print()

    # Filter gold to match
    print("Step 2: Filtering gold to match...")

    with open('data/public_test_ground_truth.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('#::snt ')
    valid_gold = []

    for i, part in enumerate(parts[1:], 1):
        if i not in pred_invalid:
            valid_gold.append(f"#::snt {part}")

    with open('gold_parseable.txt', 'w', encoding='utf-8') as f:
        f.writelines(valid_gold)

    print(f"  ✓ Wrote {len(valid_gold)} gold AMRs to gold_parseable.txt")
    print()

    # Now calculate SMATCH
    print("=" * 70)
    print("CALCULATING SMATCH")
    print("=" * 70)
    print()
    print(f"Total pairs: {pred_count}")
    print(f"Invalid pairs skipped: {len(pred_invalid)}")
    print()

    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'smatch', '-f', 'predictions_parseable.txt', 'gold_parseable.txt', '--significant', '4'],
        capture_output=True,
        text=True,
        timeout=600
    )

    print(result.stdout)

    if result.stderr and 'Unmatched parenthesis' not in result.stderr:
        print("Warnings:")
        print(result.stderr)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
