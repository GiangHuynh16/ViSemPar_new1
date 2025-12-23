"""
Detailed evaluation of 7B model with error analysis
"""
import smatch
from collections import defaultdict

def read_amr(file):
    """Read AMR file"""
    amrs = []
    sentences = []
    current = []
    current_sent = None
    
    with open(file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('# ::snt'):
                current_sent = line.replace('# ::snt', '').strip()
                if current:
                    amrs.append('\n'.join(current))
                    sentences.append(current_sent or "")
                    current = []
            elif line.strip() and not line.startswith('#'):
                current.append(line.strip())
    
    if current:
        amrs.append('\n'.join(current))
        sentences.append(current_sent or "")
    
    return amrs, sentences

print("="*70)
print("DETAILED EVALUATION - 7B MODEL")
print("="*70)

pred_file = 'outputs/public_results_20251217_220109/vietnamese_amr_public_vlsp.txt'
gold_file = 'data/public_test_ground_truth.txt'

print(f"\nLoading predictions: {pred_file}")
print(f"Loading gold:        {gold_file}\n")

pred, pred_sents = read_amr(pred_file)
gold, gold_sents = read_amr(gold_file)

print(f"Predictions: {len(pred)} AMRs")
print(f"Gold:        {len(gold)} AMRs\n")

# Match by count
min_len = min(len(pred), len(gold))
if len(pred) != len(gold):
    print(f"⚠️  Length mismatch! Using first {min_len}\n")
    pred = pred[:min_len]
    gold = gold[:min_len]
    pred_sents = pred_sents[:min_len]

# Detailed metrics
total_f1 = 0
total_precision = 0
total_recall = 0
valid_count = 0
error_types = defaultdict(int)
f1_distribution = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
good_examples = []
bad_examples = []

print("Evaluating...")
for i, (p, g, sent) in enumerate(zip(pred, gold, pred_sents)):
    try:
        best, test, gold_t = smatch.get_amr_match(p, g)
        
        if test > 0 and gold_t > 0:
            precision = best / test
            recall = best / gold_t
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            valid_count += 1
            
            # F1 distribution
            if f1 < 0.2:
                f1_distribution['0.0-0.2'] += 1
            elif f1 < 0.4:
                f1_distribution['0.2-0.4'] += 1
            elif f1 < 0.6:
                f1_distribution['0.4-0.6'] += 1
            elif f1 < 0.8:
                f1_distribution['0.6-0.8'] += 1
            else:
                f1_distribution['0.8-1.0'] += 1
            
            # Collect examples
            if f1 >= 0.8 and len(good_examples) < 3:
                good_examples.append((i, sent, f1, p[:100]))
            elif f1 <= 0.2 and len(bad_examples) < 3:
                bad_examples.append((i, sent, f1, p[:100]))
        else:
            error_types['Empty triples'] += 1
            
    except Exception as e:
        error_str = str(e)
        if 'Unmatched parenthesis' in error_str:
            error_types['Unbalanced parentheses'] += 1
        elif 'Duplicate node' in error_str:
            error_types['Duplicate variables'] += 1
        elif 'NoneType' in error_str:
            error_types['Parse error'] += 1
        else:
            error_types['Other errors'] += 1

# Results
print(f"\n{'='*70}")
print("OVERALL RESULTS")
print(f"{'='*70}")

if valid_count > 0:
    avg_f1 = total_f1 / valid_count
    avg_precision = total_precision / valid_count
    avg_recall = total_recall / valid_count
    
    print(f"\nTotal samples:     {len(pred)}")
    print(f"Valid samples:     {valid_count}")
    print(f"Invalid samples:   {len(pred) - valid_count}")
    print(f"Validity rate:     {valid_count/len(pred)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("SMATCH SCORES")
    print(f"{'='*70}")
    print(f"Precision:  {avg_precision:.4f}")
    print(f"Recall:     {avg_recall:.4f}")
    print(f"F1 Score:   {avg_f1:.4f}")
    
    print(f"\n{'='*70}")
    print("F1 SCORE DISTRIBUTION")
    print(f"{'='*70}")
    for range_name, count in sorted(f1_distribution.items()):
        pct = count/valid_count*100 if valid_count > 0 else 0
        bar = '█' * int(pct/2)
        print(f"{range_name}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"{error_type:25s}: {count:3d}")
    
    print(f"\n{'='*70}")
    print("GOOD EXAMPLES (F1 >= 0.8)")
    print(f"{'='*70}")
    for idx, sent, f1, pred_sample in good_examples:
        print(f"\n#{idx+1} | F1: {f1:.3f}")
        print(f"Sentence: {sent}")
        print(f"Prediction: {pred_sample}...")
    
    print(f"\n{'='*70}")
    print("BAD EXAMPLES (F1 <= 0.2)")
    print(f"{'='*70}")
    for idx, sent, f1, pred_sample in bad_examples:
        print(f"\n#{idx+1} | F1: {f1:.3f}")
        print(f"Sentence: {sent}")
        print(f"Prediction: {pred_sample}...")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print("TOP 1 (Qwen 14B):     0.58")
    print("TOP 2 (Qwen 1.7B):    0.46")
    print("Baseline:             0.30-0.40")
    print(f"YOUR 7B MODEL:        {avg_f1:.4f}")
    
    if avg_f1 >= 0.40:
        print("\n🎉 EXCELLENT! Above baseline!")
    elif avg_f1 >= 0.30:
        print("\n✅ GOOD! Solid baseline performance!")
    else:
        print("\n⚠️  Below baseline, but model is working")

else:
    print("❌ No valid AMRs parsed")

print(f"\n{'='*70}")
