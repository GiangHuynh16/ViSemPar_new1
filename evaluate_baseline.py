#!/usr/bin/env python3
"""
Evaluate baseline model: format predictions and calculate SMATCH score
"""

import argparse
import sys
import re
from pathlib import Path


def read_sentences(filepath):
    """Read test sentences"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def read_amr_file(filepath):
    """Read AMR file with #::snt format"""
    amr_pairs = []  # [(sentence, amr_graph), ...]
    current_sentence = None
    current_amr = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            if line.startswith('#::snt '):
                # Save previous AMR if exists
                if current_sentence is not None and current_amr:
                    amr_pairs.append((current_sentence, '\n'.join(current_amr)))

                # Start new AMR
                current_sentence = line[7:]  # Remove '#::snt '
                current_amr = []

            elif line and not line.startswith('#'):
                # AMR graph line
                current_amr.append(line)

            elif not line and current_amr:
                # Empty line marks end of AMR
                if current_sentence is not None:
                    amr_pairs.append((current_sentence, '\n'.join(current_amr)))
                    current_sentence = None
                    current_amr = []

        # Don't forget last AMR
        if current_sentence is not None and current_amr:
            amr_pairs.append((current_sentence, '\n'.join(current_amr)))

    return amr_pairs


def read_raw_predictions(filepath):
    """Read predictions without #::snt format (just AMR graphs)"""
    amrs = []
    current_amr = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            if line.startswith('('):
                # Start of new AMR
                if current_amr:
                    amrs.append('\n'.join(current_amr))
                current_amr = [line]
            elif line and current_amr:
                # Continuation of AMR
                current_amr.append(line)

        # Last AMR
        if current_amr:
            amrs.append('\n'.join(current_amr))

    return amrs


def format_predictions_with_sentences(sentences, amrs, output_file):
    """Format predictions with #::snt like ground truth"""
    print(f"\nFormatting predictions...")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Predictions: {len(amrs)}")

    if len(sentences) != len(amrs):
        print(f"  WARNING: Mismatch! Using min({len(sentences)}, {len(amrs)})")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (sent, amr) in enumerate(zip(sentences, amrs)):
            f.write(f"#::snt {sent}\n")
            f.write(f"{amr}\n")
            f.write("\n")

    print(f"  ✓ Formatted predictions saved to: {output_file}")
    return output_file


def calculate_smatch(pred_file, gold_file):
    """Calculate SMATCH score using smatch package"""
    print(f"\n{'='*70}")
    print("CALCULATING SMATCH SCORE")
    print(f"{'='*70}")

    # Read formatted files
    print("\nReading files...")
    pred_pairs = read_amr_file(pred_file)
    gold_pairs = read_amr_file(gold_file)

    print(f"  Predictions: {len(pred_pairs)} AMRs")
    print(f"  Gold: {len(gold_pairs)} AMRs")

    if len(pred_pairs) != len(gold_pairs):
        print(f"\n  WARNING: Count mismatch!")
        min_len = min(len(pred_pairs), len(gold_pairs))
        print(f"  Using first {min_len} examples")
        pred_pairs = pred_pairs[:min_len]
        gold_pairs = gold_pairs[:min_len]

    # Try to import amrlib for SMATCH calculation
    try:
        import amrlib
        print("\n  Using amrlib for SMATCH calculation...")

        # Extract just the AMR graphs
        pred_amrs = [amr for _, amr in pred_pairs]
        gold_amrs = [amr for _, amr in gold_pairs]

        # Calculate SMATCH
        scorer = amrlib.evaluate.smatch_enhanced.SmatchEnhanced()
        precision, recall, f1 = scorer.score_amr_pairs(pred_amrs, gold_amrs)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"{'='*70}")

        return {'precision': precision, 'recall': recall, 'f1': f1}

    except ImportError:
        print("\n  amrlib not available, trying smatch package...")

        try:
            import smatch

            total_match = 0
            total_test = 0
            total_gold = 0
            errors = 0

            # Detect smatch API version - check score_amr_pairs FIRST
            if hasattr(smatch, 'score_amr_pairs'):
                print("\n  Using smatch.score_amr_pairs() API...")
                # This API takes lists of AMR strings directly
                # Convert multiline AMRs to single-line by replacing newlines with spaces
                pred_amrs = [amr.replace('\n', ' ') for _, amr in pred_pairs]
                gold_amrs = [amr.replace('\n', ' ') for _, amr in gold_pairs]

                precision, recall, f1 = smatch.score_amr_pairs(pred_amrs, gold_amrs)

                print(f"\n{'='*70}")
                print("RESULTS")
                print(f"{'='*70}")
                print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
                print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
                print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
                print(f"{'='*70}")

                return {'precision': precision, 'recall': recall, 'f1': f1}
            elif hasattr(smatch, 'amr') and hasattr(smatch.amr, 'AMR'):
                print("\n  Using smatch.amr.AMR API...")
                parse_func = lambda s: smatch.amr.AMR.parse_AMR_line(s.replace('\n', ' '))
                match_func = smatch.get_amr_match
            elif hasattr(smatch, 'AMR'):
                print("\n  Using smatch.AMR API...")
                parse_func = lambda s: smatch.AMR.parse_AMR_line(s.replace('\n', ' '))
                match_func = smatch.get_amr_match
            else:
                print("\n  ERROR: Cannot detect smatch API!")
                print("  Available attributes:", [a for a in dir(smatch) if not a.startswith('_')])
                sys.exit(1)

            print("\n  Processing AMR pairs...")
            for i, ((_, pred_amr), (_, gold_amr)) in enumerate(zip(pred_pairs, gold_pairs)):
                try:
                    # Parse AMRs using detected API
                    pred_amr_obj = parse_func(pred_amr)
                    gold_amr_obj = parse_func(gold_amr)

                    # Get best match
                    match_num, test_num, gold_num = match_func(pred_amr_obj, gold_amr_obj)

                    total_match += match_num
                    total_test += test_num
                    total_gold += gold_num

                    if (i + 1) % 10 == 0:
                        print(f"    Processed {i+1}/{len(pred_pairs)}...")

                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"    Error on pair {i+1}: {e}")

            # Calculate scores
            precision = total_match / total_test if total_test > 0 else 0.0
            recall = total_match / total_gold if total_gold > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"\n{'='*70}")
            print("RESULTS")
            print(f"{'='*70}")
            print(f"Total pairs processed: {len(pred_pairs) - errors}")
            print(f"Errors: {errors}")
            print(f"Total matches: {total_match}")
            print(f"Total test triples: {total_test}")
            print(f"Total gold triples: {total_gold}")
            print()
            print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
            print(f"{'='*70}")

            return {'precision': precision, 'recall': recall, 'f1': f1}

        except ImportError:
            print("\n  ERROR: Neither amrlib nor smatch is installed!")
            print("\n  Please install one of:")
            print("    pip install amrlib")
            print("    pip install smatch")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='Raw predictions file')
    parser.add_argument('--sentences', default='data/public_test.txt', help='Test sentences')
    parser.add_argument('--gold', default='data/public_test_ground_truth.txt', help='Ground truth')
    parser.add_argument('--formatted-output', default='predictions_formatted.txt', help='Formatted predictions output')
    parser.add_argument('--results', default='evaluation_results.txt', help='Results output file')
    args = parser.parse_args()

    print("="*70)
    print("BASELINE MODEL EVALUATION")
    print("="*70)
    print(f"Predictions: {args.predictions}")
    print(f"Sentences: {args.sentences}")
    print(f"Gold: {args.gold}")

    # Step 1: Read sentences and predictions
    print("\n[1/3] Reading files...")
    sentences = read_sentences(args.sentences)
    predictions = read_raw_predictions(args.predictions)
    print(f"  ✓ {len(sentences)} sentences")
    print(f"  ✓ {len(predictions)} predictions")

    # Step 2: Format predictions with sentences
    print("\n[2/3] Formatting predictions...")
    formatted_file = format_predictions_with_sentences(
        sentences, predictions, args.formatted_output
    )

    # Step 3: Calculate SMATCH
    print("\n[3/3] Calculating SMATCH score...")
    results = calculate_smatch(formatted_file, args.gold)

    # Save results
    with open(args.results, 'w', encoding='utf-8') as f:
        f.write(f"Baseline Model Evaluation Results\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Predictions: {args.predictions}\n")
        f.write(f"Gold: {args.gold}\n\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")

    print(f"\n✓ Results saved to: {args.results}")
    print(f"✓ Formatted predictions: {args.formatted_output}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
