#!/usr/bin/env python3
"""
Calculate SMATCH score between predictions and ground truth
"""

import argparse
import sys
from pathlib import Path

try:
    import smatch
except ImportError:
    print("ERROR: smatch library not installed!")
    print("Install with: pip install smatch")
    sys.exit(1)


def read_amr_file(filepath):
    """Read AMR graphs from file"""
    amrs = []
    current_amr = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                current_amr.append(line)
            else:
                if current_amr:
                    amrs.append('\n'.join(current_amr))
                    current_amr = []

        # Don't forget the last AMR
        if current_amr:
            amrs.append('\n'.join(current_amr))

    return amrs


def calculate_smatch_score(predictions_file, gold_file):
    """Calculate SMATCH score"""
    print("=" * 70)
    print("SMATCH SCORE CALCULATION")
    print("=" * 70)
    print(f"Predictions: {predictions_file}")
    print(f"Gold: {gold_file}")
    print()

    # Read files
    print("Reading predictions...")
    predictions = read_amr_file(predictions_file)
    print(f"✓ Found {len(predictions)} predictions")

    print("Reading ground truth...")
    gold = read_amr_file(gold_file)
    print(f"✓ Found {len(gold)} gold AMRs")
    print()

    if len(predictions) != len(gold):
        print(f"WARNING: Number of predictions ({len(predictions)}) != gold ({len(gold)})")
        min_len = min(len(predictions), len(gold))
        print(f"Using first {min_len} examples for comparison")
        predictions = predictions[:min_len]
        gold = gold[:min_len]
        print()

    # Calculate SMATCH
    print("Calculating SMATCH score...")
    print("(This may take a few minutes...)")
    print()

    total_match = 0
    total_test = 0
    total_gold = 0

    for i, (pred_amr, gold_amr) in enumerate(zip(predictions, gold)):
        # Parse AMR graphs
        try:
            pred_graph = smatch.AMR.parse_AMR_line(pred_amr)
            gold_graph = smatch.AMR.parse_AMR_line(gold_amr)

            # Calculate match
            best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                pred_graph, gold_graph
            )

            total_match += best_match_num
            total_test += test_triple_num
            total_gold += gold_triple_num

            # Print progress every 10 examples
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(predictions)} examples...")

        except Exception as e:
            print(f"  WARNING: Error parsing AMR {i+1}: {e}")
            continue

    # Calculate final scores
    precision = total_match / total_test if total_test > 0 else 0.0
    recall = total_match / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total examples: {len(predictions)}")
    print(f"Total matches: {total_match}")
    print(f"Total test triples: {total_test}")
    print(f"Total gold triples: {total_gold}")
    print()
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 (SMATCH): {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 70)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_match': total_match,
        'total_test': total_test,
        'total_gold': total_gold
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate SMATCH score')
    parser.add_argument('--predictions', '-p', required=True, help='Predictions file')
    parser.add_argument('--gold', '-g', required=True, help='Ground truth file')
    parser.add_argument('--output', '-o', help='Output file for results (optional)')
    args = parser.parse_args()

    # Check files exist
    if not Path(args.predictions).exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not Path(args.gold).exists():
        print(f"ERROR: Ground truth file not found: {args.gold}")
        sys.exit(1)

    # Calculate SMATCH
    results = calculate_smatch_score(args.predictions, args.gold)

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Predictions: {args.predictions}\n")
            f.write(f"Gold: {args.gold}\n")
            f.write(f"\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 (SMATCH): {results['f1']:.4f}\n")
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
