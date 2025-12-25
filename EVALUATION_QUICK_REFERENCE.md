# 📋 Evaluation Quick Reference

## 🚀 Run Full Evaluation (1 command)

```bash
cd ~/ViSemPar_new1
git pull origin main
bash RUN_FULL_EVALUATION_TMUX.sh
```

## 📊 Check Status

```bash
bash CHECK_EVALUATION_STATUS.sh
```

## 👀 Monitor Live

```bash
# Attach to tmux
tmux attach -t mtup_eval

# Detach: Ctrl+B then D

# OR watch log
tail -f outputs/evaluation_full_*.log
```

## 📈 View Results

```bash
# Find latest results
ls -t outputs/evaluation_results_full_*.json | head -1

# View formatted
cat outputs/evaluation_results_full_*.json | python3 -m json.tool
```

## ⏱️ Time Estimates

| Samples | Time |
|---------|------|
| 10 | ~3 min |
| 50 | ~17 min |
| 100 | ~33 min |
| 200 | ~67 min |
| 500 | ~2.8 hours |

Formula: `samples × 20 sec ÷ 60 = minutes`

## 🎯 Current Results

**Quick Test (10 samples)**:
- ✅ F1: **0.4933** (~49%)
- ✅ Precision: 0.4978
- ✅ Recall: 0.5002
- ✅ Success: 7/10 (70%)

**Errors**:
- 2× Duplicate node names
- 1× Unmatched parenthesis

## 🛑 Stop Evaluation

```bash
tmux kill-session -t mtup_eval
# OR
pkill -f evaluate_mtup_model.py
```

## 📁 Output Files

```
outputs/
├── evaluation_results_full_TIMESTAMP.json  ← Scores
└── evaluation_full_TIMESTAMP.log           ← Log
```

## ✅ Success Criteria

| F1 Score | Status | Action |
|----------|--------|--------|
| > 0.60 | 🟢 Excellent | Ready for deployment |
| 0.50-0.60 | 🟡 Good | Minor improvements |
| 0.40-0.50 | 🟠 Acceptable | Consider training more |
| < 0.40 | 🔴 Poor | Need retraining |

Current: **0.49** (Acceptable, close to Good)

## 🔧 Troubleshooting

### Stuck?
```bash
nvidia-smi  # Check GPU
ps aux | grep evaluate  # Check process
tail -30 outputs/evaluation_full_*.log  # Check log
```

### Restart?
```bash
tmux kill-session -t mtup_eval
bash RUN_FULL_EVALUATION_TMUX.sh
```

## 📖 Full Documentation

See: [HOW_TO_RUN_FULL_EVALUATION.md](HOW_TO_RUN_FULL_EVALUATION.md)
