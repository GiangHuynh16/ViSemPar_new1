# Hugging Face Setup Guide

## 📋 **QUY TRÌNH SETUP TRÊN SERVER**

---

## **BƯỚC 1: Tạo Hugging Face Token**

### 1.1. Tạo Account
- Truy cập: https://huggingface.co/join
- Đăng ký account (free)

### 1.2. Tạo Access Token
1. Vào: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Nhập tên token (ví dụ: `vlsp-amr-server`)
4. Chọn **"Write"** permission
5. Click **"Generate token"**
6. **COPY TOKEN** (dạng: `hf_xxxxxxxxxxxxx`)

⚠️ **LƯU Ý:** Token chỉ hiện 1 lần, hãy lưu lại ngay!

---

## **BƯỚC 2: Setup Token trên Server**

### **CÁCH 1: Environment Variable (RECOMMENDED ⭐)**

Thêm vào file `~/.bashrc` hoặc `~/.zshrc`:

```bash
# Mở file
nano ~/.bashrc

# Thêm dòng này (thay YOUR_TOKEN bằng token thực)
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Lưu và reload
source ~/.bashrc
```

**Test:**
```bash
echo $HF_TOKEN
# Phải hiện token của bạn
```

---

### **CÁCH 2: Hugging Face CLI Login (DễAsy & Secure ⭐⭐)**

```bash
# Install huggingface-cli (nếu chưa có)
pip install --upgrade huggingface_hub

# Login
huggingface-cli login

# Paste token khi được hỏi
# Token will be saved to ~/.cache/huggingface/token
```

**Advantages:**
- ✅ An toàn hơn (token được encrypt)
- ✅ Không cần hardcode vào code
- ✅ Works với tất cả HF libraries

---

### **CÁCH 3: .env File (For Development)**

Tạo file `.env` trong project root:

```bash
# File: /path/to/ViSemPar_new1/.env
HF_TOKEN=hf_xxxxxxxxxxxxx
```

⚠️ **BẮT BUỘC:** Thêm `.env` vào `.gitignore`!

```bash
echo ".env" >> .gitignore
```

Load trong Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('HF_TOKEN')
```

---

## **BƯỚC 3: Verify Setup**

### Test 1: Check Token
```bash
# Nếu dùng CLI login
huggingface-cli whoami

# Nếu dùng environment variable
echo $HF_TOKEN
```

### Test 2: Test với Python
```python
from huggingface_hub import HfApi

api = HfApi()
user_info = api.whoami()
print(f"Logged in as: {user_info['name']}")
```

---

## **BƯỚC 4: Update Code để Sử Dụng Token**

### File cần update: `config/config_mtup.py`

Đã có sẵn config, chỉ cần update:

```python
HF_CONFIG = {
    "repo_name": "vietnamese-amr-mtup-qwen",  # Đổi tên repo của bạn
    "private": False,                          # True nếu muốn repo private
    "push_to_hub": False,                      # True khi muốn push
    "hub_strategy": "every_save",
}
```

### Code sẽ tự động lấy token:

```python
# Trong training script
from huggingface_hub import HfApi

# Tự động lấy token từ:
# 1. ~/.cache/huggingface/token (nếu dùng CLI login)
# 2. $HF_TOKEN environment variable
# 3. .env file (nếu dùng python-dotenv)

model.push_to_hub(
    repo_name,
    token=True,  # Auto-detect token
    private=False
)
```

---

## **CÁCH SETUP RECOMMENDED CHO SERVER:**

### **Option A: University Server (Shared Server)**

```bash
# 1. SSH vào server
ssh your_username@server_address

# 2. Login với huggingface-cli (RECOMMENDED)
pip install --upgrade huggingface_hub
huggingface-cli login
# Paste token: hf_xxxxxxxxxxxxx

# 3. Verify
huggingface-cli whoami

# 4. Clone project
git clone <your-repo-url>
cd ViSemPar_new1

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run training
python train_mtup.py --use-case fast_iteration
```

---

### **Option B: Personal Server / Cloud (tmux session)**

```bash
# 1. SSH vào server
ssh your_server

# 2. Setup token
export HF_TOKEN="hf_xxxxxxxxxxxxx"
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxx"' >> ~/.bashrc

# 3. Start tmux
tmux new -s amr-training

# 4. Navigate & train
cd ViSemPar_new1
python train_mtup.py --use-case fast_iteration

# 5. Detach (Ctrl+B, then D)
# Training continues in background

# 6. Reattach later
tmux attach -t amr-training
```

---

## **PUSH MODEL LÊN HUGGING FACE**

### **Trong code (train_mtup.py):**

```python
# Option 1: Push during training (automatic)
HF_CONFIG = {
    "push_to_hub": True,
    "repo_name": "your-username/vietnamese-amr-mtup",
}

# Option 2: Push sau khi training xong (manual)
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/vlsp_amr_mtup_v1",
    repo_id="your-username/vietnamese-amr-mtup",
    repo_type="model",
    token=True  # Auto-detect
)
```

---

## **SECURITY BEST PRACTICES**

### ✅ **DO:**
- ✅ Dùng `huggingface-cli login` (most secure)
- ✅ Dùng environment variables
- ✅ Add `.env` to `.gitignore`
- ✅ Use `token=True` (auto-detect) thay vì hardcode

### ❌ **DON'T:**
- ❌ Hardcode token trong code
- ❌ Commit token vào git
- ❌ Share token publicly
- ❌ Dùng chung token cho nhiều người

---

## **TROUBLESHOOTING**

### Issue 1: "Token not found"
```bash
# Check token location
ls -la ~/.cache/huggingface/

# Re-login
huggingface-cli logout
huggingface-cli login
```

### Issue 2: "Permission denied"
```bash
# Token cần Write permission
# Tạo lại token với Write permission tại:
# https://huggingface.co/settings/tokens
```

### Issue 3: "Repository not found"
```bash
# Create repo first
huggingface-cli repo create vietnamese-amr-mtup --type model

# Or create on web:
# https://huggingface.co/new
```

---

## **QUICK REFERENCE**

### Check Token Status:
```bash
huggingface-cli whoami
```

### Create Model Repository:
```bash
huggingface-cli repo create MODEL_NAME --type model
```

### Upload Model:
```bash
huggingface-cli upload MODEL_NAME ./path/to/model
```

### List Your Models:
```bash
huggingface-cli repo list
```

---

## **FILE LOCATIONS**

### Token Locations (in order of priority):
1. `~/.cache/huggingface/token` (CLI login)
2. `$HF_TOKEN` environment variable
3. `.env` file in project root

### Config Files:
- Project config: `config/config_mtup.py`
- HF config section: `HF_CONFIG` dict
- gitignore: `.gitignore` (make sure `.env` is there)

---

## **RECOMMENDED WORKFLOW FOR YOUR CASE:**

```bash
# 1. Trên server (lần đầu tiên)
ssh your_server
pip install --upgrade huggingface_hub
huggingface-cli login
# Paste token: hf_xxxxxxxxxxxxx

# 2. Verify
huggingface-cli whoami

# 3. Clone project (nếu chưa)
git clone <repo_url>
cd ViSemPar_new1

# 4. Training với tmux
tmux new -s amr-mtup
python train_mtup.py --use-case fast_iteration

# 5. Detach và đợi
# Ctrl+B, D

# 6. Check lại sau
tmux attach -t amr-mtup
```

---

## **BONUS: Auto-Push to HF During Training**

Trong `config/config_mtup.py`, update:

```python
HF_CONFIG = {
    "repo_name": "your-username/vietnamese-amr-mtup-qwen3b",
    "private": False,  # True if you want private repo
    "push_to_hub": True,  # ⭐ Set to True
    "hub_strategy": "every_save",  # Push every checkpoint
}
```

Model sẽ tự động push lên HF sau mỗi checkpoint! 🚀

---

**Done! Token được setup và code sẽ tự động detect token khi push model.**
