"""Final evaluation of 14B model"""
import smatch
from collections import defaultdict

def read_amr_multiline(file):
    amrs, sents = [], []
    current_amr, current_sent = [], None
    with open(file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#::snt') or line.startswith('# ::snt'):
                if current_amr:
                    amrs.append(' '.join(l.strip() for l in current_amr))
                    sents.append(current_sent or "")
                current_amr = []
                current_sent = line.replace('#::snt', '').replace('# ::snt', '').strip()
            elif line.strip():
                current_amr.append(line)
    if current_amr:
        amrs.append(' '.join(l.strip() for l in current_amr))
        sents.append(current_sent or "")
    return amrs, sents

print("="*70)
print("FINAL EVALUATION - 14B MODEL")
print("="*70)

pred, pred_sents = read_amr_multiline('outputs/public_14b_predictions_full.txt')
gold, gold_sents = read_amr_multiline('data/public_test_ground_truth.txt')

print(f"\nPredictions: {len(pred)}")
print(f"Gold:        {len(gold)}")

min_len = min(len(pred), len(gold))
pred, gold = pred[:min_len], gold[:min_len]

# Evaluate
total_f1, total_p, total_r = 0, 0, 0
valid = 0
errors = defaultdict(int)

print(f"\nEvaluating {len(pred)} AMRs...")
for i, (p, g) in enumerate(zip(pred, gold)):
    try:
        best, test, gold_t = smatch.get_amr_match(p, g)
        if test > 0 and gold_t > 0:
            precision = best / test
            recall = best / gold_t
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            total_f1 += f1
            total_p += precision
            total_r += recall
            valid += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pred)}...")
    except Exception as e:
        error_str = str(e)
        if 'Unmatched parenthesis' in error_str:
            errors['Unbalanced parens'] += 1
        elif 'Duplicate node' in error_str:
            errors['Duplicate vars'] += 1
        else:
            errors['Other'] += 1

# Results
print(f"\n{'='*70}")
print("FINAL RESULTS - 14B MODEL")
print(f"{'='*70}")

if valid > 0:
    avg_f1 = total_f1 / valid
    avg_p = total_p / valid
    avg_r = total_r / valid
    
    print(f"\nTotal:       {len(pred)}")
    print(f"Valid:       {valid} ({valid/len(pred)*100:.1f}%)")
    print(f"Invalid:     {len(pred)-valid}")
    
    print(f"\n{'='*70}")
    print("SMATCH SCORES")
    print(f"{'='*70}")
    print(f"Precision:   {avg_p:.4f}")
    print(f"Recall:      {avg_r:.4f}")
    print(f"F1 Score:    {avg_f1:.4f}")
    
    print(f"\n{'='*70}")
    print("COMPARISON: 7B vs 14B")
    print(f"{'='*70}")
    print(f"7B Model:")
    print(f"  F1:        0.4208")
    print(f"  Validity:  58.0%")
    print(f"\n14B Model:")
    print(f"  F1:        {avg_f1:.4f}")
    print(f"  Validity:  {valid/len(pred)*100:.1f}%")
    print(f"\nImprovement:")
    print(f"  F1:        {(avg_f1 - 0.4208)*100:+.1f} percentage points")
    print(f"  Validity:  {(valid/len(pred)*100 - 58.0):+.1f}%")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"TOP 1 (14B):    0.58")
    print(f"TOP 2 (1.7B):   0.46")
    print(f"YOUR 14B:       {avg_f1:.4f}")
    print(f"Gap to TOP 2:   {avg_f1 - 0.46:+.4f}")
    
    if avg_f1 >= 0.46:
        print("\n🎉🎉🎉 EXCELLENT! MATCHED OR BEAT TOP 2! 🎉🎉🎉")
    elif avg_f1 >= 0.43:
        print("\n🎉 GREAT! Very close to TOP 2!")
    elif avg_f1 >= 0.40:
        print("\n✅ GOOD! Above baseline!")
    
    if len(errors) > 0:
        print(f"\n{'='*70}")
        print("ERROR BREAKDOWN")
        print(f"{'='*70}")
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"{err:20s}: {count}")

else:
    print("\n❌ No valid parses")

