#!/bin/bash
################################################################################
# CHECK TRAINING STATUS
# Kiểm tra xem training có đang chạy không
################################################################################

echo "========================================================================"
echo "🔍 CHECKING TRAINING STATUS"
echo "========================================================================"
echo ""

# Check if Python process is running
echo "1️⃣  Checking Python processes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ps aux | grep "train_mtup.py" | grep -v grep
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training process is RUNNING"
else
    echo "❌ No training process found"
fi
echo ""

# Check GPU usage
echo "2️⃣  Checking GPU usage..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "GPU %s (%s):\n  Utilization: %s%%\n  Memory: %s/%s MB (%.1f%%)\n\n", $1, $2, $3, $4, $5, ($4/$5)*100}'

# If GPU usage > 50%, likely training
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$gpu_util" -gt 50 ]; then
    echo "✅ GPU utilization > 50% - Training likely running"
else
    echo "⚠️  GPU utilization < 50% - Training may not be running"
fi
echo ""

# Check tmux sessions
echo "3️⃣  Checking tmux sessions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tmux list-sessions 2>/dev/null
if [ $? -eq 0 ]; then
    echo ""
    echo "To attach to session:"
    echo "  tmux attach -t mtup_full"
else
    echo "No tmux sessions found"
fi
echo ""

# Check recent checkpoints
echo "4️⃣  Checking recent checkpoints..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "outputs/checkpoints_mtup" ]; then
    ls -lt outputs/checkpoints_mtup/ | head -5
    echo ""
    last_checkpoint=$(ls -t outputs/checkpoints_mtup/ | head -1)
    if [ -n "$last_checkpoint" ]; then
        checkpoint_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "outputs/checkpoints_mtup/$last_checkpoint" 2>/dev/null || stat -c "%y" "outputs/checkpoints_mtup/$last_checkpoint" 2>/dev/null)
        echo "Latest checkpoint: $last_checkpoint"
        echo "Created: $checkpoint_time"

        # Check if checkpoint is recent (within 10 minutes)
        if [ -n "$checkpoint_time" ]; then
            echo "✅ Recent checkpoint found - Training is/was active"
        fi
    fi
else
    echo "No checkpoints directory found"
fi
echo ""

# Check log files
echo "5️⃣  Checking recent log activity..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "outputs/logs" ]; then
    latest_log=$(find outputs/logs -type f -name "*.out.tfevents.*" -o -name "events.out.tfevents.*" | head -1)
    if [ -n "$latest_log" ]; then
        log_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$latest_log" 2>/dev/null || stat -c "%y" "$latest_log" 2>/dev/null)
        echo "Latest log: $latest_log"
        echo "Modified: $log_time"
    else
        echo "No log files found yet"
    fi
else
    echo "No logs directory found"
fi
echo ""

# Show last few lines of training output if in tmux
echo "6️⃣  Recent training output (if available)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tmux_session=$(tmux list-sessions 2>/dev/null | grep "mtup" | cut -d: -f1 | head -1)
if [ -n "$tmux_session" ]; then
    echo "Capturing output from tmux session: $tmux_session"
    tmux capture-pane -t "$tmux_session" -p | tail -20
else
    echo "No mtup tmux session found"
fi
echo ""

echo "========================================================================"
echo "💡 COMMANDS TO MONITOR TRAINING"
echo "========================================================================"
echo ""
echo "Attach to tmux session:"
echo "  tmux attach -t mtup_full"
echo ""
echo "Watch GPU usage (real-time):"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "View training logs (real-time):"
echo "  tail -f outputs/logs/mtup_*/events.out.tfevents.*"
echo ""
echo "List all Python processes:"
echo "  ps aux | grep python"
echo ""
echo "Kill training if needed:"
echo "  pkill -f train_mtup.py"
echo "  # Or in tmux: Ctrl+C"
echo ""
echo "========================================================================"
