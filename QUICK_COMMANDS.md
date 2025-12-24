# ⚡ QUICK COMMANDS - Copy Paste Nhanh

## 🔍 Kiểm Tra Training Đang Chạy Không

### Cách 1: Script Tự Động (Khuyến nghị)
```bash
cd ~/ViSemPar_new1
bash CHECK_TRAINING_STATUS.sh
```

### Cách 2: Manual Commands

**Xem process Python:**
```bash
ps aux | grep train_mtup
```
- Nếu có output → Training đang chạy ✅
- Nếu không có gì → Training không chạy ❌

**Xem GPU usage:**
```bash
nvidia-smi
```
- GPU-Util > 50% → Training đang chạy ✅
- Memory-Usage > 20GB → Model đã load ✅

**Xem tmux sessions:**
```bash
tmux list-sessions
```
- Có session `mtup_full` → Training trong tmux ✅

---

## 👁️ Xem Training Progress

### Attach vào tmux
```bash
tmux attach -t mtup_full
```
Bấm `Ctrl+B` rồi `D` để detach lại

### Xem GPU real-time
```bash
watch -n 1 nvidia-smi
```
Bấm `Ctrl+C` để thoát

### Xem checkpoints
```bash
ls -lh outputs/checkpoints_mtup/
```
Mỗi checkpoint mới = training đã chạy thêm 250 steps

---

## 🛑 Dừng Training

### Dừng tạm (có thể resume)
```bash
# Trong tmux session
tmux attach -t mtup_full
# Nhấn Ctrl+C

# Hoặc từ ngoài
pkill -f train_mtup.py
```

### Kill tmux session hoàn toàn
```bash
tmux kill-session -t mtup_full
```

---

## 🎯 Most Common Commands

```bash
# Kiểm tra status
bash CHECK_TRAINING_STATUS.sh

# Attach vào training
tmux attach -t mtup_full

# Xem GPU
nvidia-smi

# Xem checkpoints
ls -lh outputs/checkpoints_mtup/

# Dừng training
pkill -f train_mtup.py
```
