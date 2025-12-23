"""
Fixed evaluation for multi-line AMR format
"""
import smatch
from collections import defaultdict

def read_amr_multiline(file):
    """Read multi-line AMR file"""
    amrs = []
    sentences = []
    current_lines = []
    current_sent = None
    
    with open(file, encoding='utf-8') as f:
        for line in f:
            # Sentence comment
            if line.startswith('#::snt'):
                # Save previous AMR
                if current_lines:
                    amr_text = ' '.join(l.strip() for l in current_lines if l.strip())
                    if amr_text:
                        amrs.append(amr_text)
                        sentences.append(current_sent or "")
                    current_lines = []
                
                current_sent = line.replace('#::snt', '').strip()
            
            # AMR content (starts with ( or whitespace+content)
            elif line.strip():
                current_lines.append(line)
    
    # Don't forget last AMR
    if current_lines:
        amr_text = ' '.join(l.strip() for l in current_lines if l.strip())
        if amr_text:
            amrs.append(amr_text)
            sentences.append(current_sent or "")
    
    return amrs, sentences

print("="*70)
print("EVALUATION - 7B MODEL (FIXED READER)")
print("="*70)

pred_file = 'outputs/public_results_20251217_220109/vietnamese_amr_public_vlsp.txt'
gold_file = 'data/public_test_ground_truth.txt'

print(f"\nLoading predictions...")
pred, pred_sents = read_amr_multiline(pred_file)
print(f"✅ Loaded {len(pred)} predictions")

print(f"\nLoading gold...")
gold, gold_sents = read_amr_multiline(gold_file)
print(f"✅ Loaded {len(gold)} gold AMRs")

# Match
min_len = min(len(pred), len(gold))
if len(pred) != len(gold):
    print(f"\n⚠️ Length mismatch! Using first {min_len}")
    pred = pred[:min_len]
    gold = gold[:min_len]

# Evaluate
total_f1 = 0
total_precision = 0
total_recall = 0
valid = 0
error_types = defaultdict(int)
f1_bins = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}

print(f"\nEvaluating {len(pred)} AMRs...")
for i, (p, g) in enumerate(zip(pred, gold)):
    try:
        best, test, gold_t = smatch.get_amr_match(p, g)
        
        if test > 0 and gold_t > 0:
            precision = best / test
            recall = best / gold_t
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            valid += 1
            
            # Bin F1 scores
            if f1 < 0.2:
                f1_bins['0.0-0.2'] += 1
            elif f1 < 0.4:
                f1_bins['0.2-0.4'] += 1
            elif f1 < 0.6:
                f1_bins['0.4-0.6'] += 1
            elif f1 < 0.8:
                f1_bins['0.6-0.8'] += 1
            else:
                f1_bins['0.8-1.0'] += 1
                
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pred)}...")
            
    except Exception as e:
        error_str = str(e)
        if 'Unmatched parenthesis' in error_str:
            error_types['Unbalanced parens'] += 1
        elif 'Duplicate node' in error_str:
            error_types['Duplicate vars'] += 1
        elif 'NoneType' in error_str:
            error_types['Parse error'] += 1
        else:
            error_types['Other'] += 1

# Results
print(f"\n{'='*70}")
print("RESULTS - 7B MODEL WITH VARIABLES")
print(f"{'='*70}")

if valid > 0:
    avg_f1 = total_f1 / valid
    avg_p = total_precision / valid
    avg_r = total_recall / valid
    
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
    print("F1 DISTRIBUTION")
    print(f"{'='*70}")
    for bin_range, count in sorted(f1_bins.items()):
        pct = count/valid*100 if valid > 0 else 0
        bar = '█' * int(pct/2)
        print(f"{bin_range}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n{'='*70}")
    print("ERROR TYPES")
    print(f"{'='*70}")
    for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"{err_type:20s}: {count}")
    
    print(f"\n{'='*70}")
    print("BENCHMARK")
    print(f"{'='*70}")
    print(f"TOP 1 (14B):  0.58")
    print(f"TOP 2 (1.7B): 0.46")
    print(f"Baseline:     0.30-0.40")
    print(f"YOUR 7B:      {avg_f1:.4f}")
    
    if avg_f1 >= 0.40:
        print("\n🎉 EXCELLENT!")
    elif avg_f1 >= 0.30:
        print("\n✅ GOOD - Baseline level!")
    else:
        print("\n⚠️ Below baseline")
        
else:
    print("\n❌ No valid parses")

