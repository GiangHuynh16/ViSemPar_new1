#!/usr/bin/env python3
"""
Calculate SMATCH using subprocess to call smatch command
"""

import subprocess
import sys
import argparse
from pathlib import Path


def calculate_smatch_subprocess(pred_file, gold_file):
    """Calculate SMATCH by calling smatch via subprocess"""
    print("=" * 70)
    print("SMATCH CALCULATION VIA SUBPROCESS")
    print("=" * 70)
    print(f"Predictions: {pred_file}")
    print(f"Gold: {gold_file}")
    print()

    # Check files exist
    if not Path(pred_file).exists():
        print(f"ERROR: Predictions file not found: {pred_file}")
        sys.exit(1)

    if not Path(gold_file).exists():
        print(f"ERROR: Gold file not found: {gold_file}")
        sys.exit(1)

    # Try to call smatch as a module
    print("Running SMATCH calculation...")
    print("(This may take a few minutes...)")
    print()

    try:
        # Call: python -m smatch pred_file gold_file
        result = subprocess.run(
            [sys.executable, '-m', 'smatch', pred_file, gold_file, '--significant', '4'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        if result.returncode != 0:
            print(f"\nSMATCH command failed with return code {result.returncode}")
            return None

        # Parse output to extract F1 score
        lines = result.stdout.split('\n')
        for line in lines:
            if 'F-score:' in line or 'F1:' in line or 'Smatch score:' in line:
                print()
                print("=" * 70)
                print("RESULT")
                print("=" * 70)
                print(line)
                print("=" * 70)

        return result.stdout

    except subprocess.TimeoutExpired:
        print("ERROR: SMATCH calculation timed out (>10 minutes)")
        return None
    except Exception as e:
        print(f"ERROR: Failed to run smatch: {e}")
        print()
        print("Trying alternative approach...")

        # Try using Python API directly
        try:
            import smatch
            print("Using smatch.main() directly...")

            # Temporarily replace sys.argv
            old_argv = sys.argv
            sys.argv = ['smatch', pred_file, gold_file, '--significant', '4']

            try:
                smatch.main()
            finally:
                sys.argv = old_argv

        except Exception as e2:
            print(f"ERROR: smatch.main() also failed: {e2}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Calculate SMATCH via subprocess')
    parser.add_argument('--predictions', '-p', required=True, help='Predictions file')
    parser.add_argument('--gold', '-g', required=True, help='Ground truth file')
    args = parser.parse_args()

    calculate_smatch_subprocess(args.predictions, args.gold)


if __name__ == "__main__":
    main()
