# HuggingFace Login Guide

**Hướng dẫn đăng nhập HuggingFace để training model**

---

## 🎯 **TẠI SAO CẦN LOGIN?**

HuggingFace login cần thiết để:
- ✅ Download pretrained models (Qwen, Gemma, Phi)
- ✅ Save model checkpoints
- ✅ Push model to Hub (optional)
- ✅ Access gated models (nếu có)

---

## 🔑 **LẤY TOKEN TỪ HUGGINGFACE**

### Bước 1: Đăng ký/Đăng nhập HuggingFace
1. Truy cập: https://huggingface.co
2. Đăng ký tài khoản (nếu chưa có)
3. Đăng nhập

### Bước 2: Tạo Access Token
1. Vào: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Đặt tên: `vlsp-amr-server` (hoặc tên khác)
4. Chọn permission: **"Write"** (quan trọng!)
5. Click **"Generate a new token"**
6. **COPY TOKEN** (dạng: `hf_xxxxxxxxxxxxx`)

⚠️ **LƯU Ý**: Token chỉ hiện 1 lần! Copy và lưu lại.

---

## 💻 **CÁCH LOGIN - 4 PHƯƠNG PHÁP**

### **Phương pháp 1: CLI Login (RECOMMENDED) ⭐**

**Ưu điểm:**
- ✅ Đơn giản nhất
- ✅ Token lưu an toàn trong cache
- ✅ Tự động dùng cho tất cả scripts
- ✅ Không cần code gì thêm

**Cách làm:**
```bash
# Trên server
huggingface-cli login

# Paste token khi được hỏi
# Token (input will not be visible): hf_xxxxxxxxxxxxx

# Enter để confirm
```

**Verify:**
```bash
huggingface-cli whoami
# Should show: your-username
```

**Vị trí lưu token:**
```
~/.cache/huggingface/token
```

---

### **Phương pháp 2: Environment Variable**

**Ưu điểm:**
- ✅ Linh hoạt cho mỗi session
- ✅ Không lưu file

**Cách làm:**
```bash
# Set environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Verify
echo $HF_TOKEN

# Chạy training (tự động dùng token)
python3 train_mtup.py --use-case quick_test
```

**Lưu vĩnh viễn (optional):**
```bash
# Thêm vào ~/.bashrc hoặc ~/.zshrc
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

---

### **Phương pháp 3: .env File**

**Ưu điểm:**
- ✅ Dễ quản lý
- ✅ Git ignore tự động (an toàn)

**Cách làm:**
```bash
# Tạo file .env
cd ~/ViSemPar_new1
nano .env
```

**Nội dung .env:**
```bash
# HuggingFace Access Token
HF_TOKEN=hf_xxxxxxxxxxxxx

# Optional: Username
HF_USERNAME=your-username
```

**Save:** `Ctrl+O`, Enter, `Ctrl+X`

**Verify:**
```bash
# Check .env exists
cat .env

# Training sẽ tự động đọc từ .env
python3 train_mtup.py --use-case quick_test
```

**⚠️ QUAN TRỌNG:** File `.env` đã được add vào `.gitignore` → không push lên git!

---

### **Phương pháp 4: Python Script**

**Ưu điểm:**
- ✅ Interactive
- ✅ Kiểm tra token ngay

**Cách làm:**
```bash
# Sử dụng script login
python3 hf_login.py

# Hoặc với token
python3 hf_login.py --token hf_xxxxxxxxxxxxx

# Check status
python3 hf_login.py --check

# Logout
python3 hf_login.py --logout
```

**Hoặc trong Python code:**
```python
from hf_auth import ensure_hf_login

# Automatically login
ensure_hf_login()
```

---

## 🚀 **TRAINING VỚI AUTO LOGIN**

Training script đã tích hợp auto login!

### **Workflow tự động:**
```python
# train_mtup.py tự động:
1. Check HuggingFace login
2. Nếu chưa login → tìm token từ:
   - Environment variable (HF_TOKEN)
   - .env file
   - CLI cache (~/.cache/huggingface/token)
3. Tự động login nếu tìm thấy token
4. Nếu không tìm thấy → warning nhưng vẫn tiếp tục
```

### **Chạy training:**
```bash
# Nếu đã login bằng CLI hoặc có .env
python3 train_mtup.py --use-case quick_test
# → Tự động detect và login!

# Hoặc với environment variable
HF_TOKEN=hf_xxxxx python3 train_mtup.py --use-case quick_test
```

---

## 🔍 **VERIFY LOGIN**

### **Kiểm tra đã login chưa:**

**Method 1: CLI**
```bash
huggingface-cli whoami
```

**Method 2: Python**
```bash
python3 hf_login.py --check
```

**Method 3: In script**
```python
from hf_auth import get_hf_username

username = get_hf_username()
if username:
    print(f"Logged in as: {username}")
else:
    print("Not logged in")
```

---

## 📋 **SO SÁNH CÁC PHƯƠNG PHÁP**

| Method | Ease | Security | Persistence | Recommended |
|--------|------|----------|-------------|-------------|
| **CLI Login** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | **YES** ⭐ |
| **Env Variable** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ No (session) | For testing |
| **.env File** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes | Alternative |
| **Python Script** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes | Advanced |

---

## 🎓 **BEST PRACTICES**

### **1. Luôn dùng CLI Login trên server:**
```bash
# One-time setup
huggingface-cli login
# Paste token
# Done! Không cần làm gì thêm
```

### **2. Sử dụng .env cho development (local):**
```bash
# Local machine
echo 'HF_TOKEN=hf_xxxxx' > .env
python3 train_mtup.py --use-case quick_test
```

### **3. Không commit token vào git:**
```bash
# .gitignore đã có:
.env
.env.local
.env.*.local
```

### **4. Verify trước khi training:**
```bash
# Quick check
huggingface-cli whoami

# Nếu OK → bắt đầu training
python3 train_mtup.py --use-case full_training
```

---

## 🔧 **TROUBLESHOOTING**

### **Lỗi: "401 Unauthorized"**

**Nguyên nhân:** Token invalid hoặc hết hạn

**Fix:**
```bash
# Logout
huggingface-cli logout

# Login lại với token mới
huggingface-cli login
# Paste token mới
```

---

### **Lỗi: "Token not found"**

**Nguyên nhân:** Chưa login hoặc token không đọc được

**Fix:**
```bash
# Check token location
ls -la ~/.cache/huggingface/

# Re-login
huggingface-cli login
```

---

### **Lỗi: "Permission denied"**

**Nguyên nhân:** Token không có "Write" permission

**Fix:**
1. Vào: https://huggingface.co/settings/tokens
2. Tạo token MỚI
3. Chọn **"Write"** permission
4. Login lại

---

### **Training vẫn chạy nhưng warning login failed**

**Nguyên nhân:** Token không tìm thấy nhưng model có thể download public

**Fix (optional):**
```bash
# Login để tránh rate limits
huggingface-cli login
```

---

## 🚨 **BẢO MẬT TOKEN**

### **DO:**
✅ Lưu token trong CLI cache (`huggingface-cli login`)
✅ Dùng .env file (đã git ignore)
✅ Set environment variable
✅ Chỉ share với người tin cậy

### **DON'T:**
❌ Commit token vào git
❌ Share token public
❌ Hard-code token trong code
❌ Screenshot token

### **Nếu token bị lộ:**
1. Vào: https://huggingface.co/settings/tokens
2. Revoke token cũ
3. Tạo token mới
4. Login lại

---

## 📝 **QUICK REFERENCE**

### **Lệnh hay dùng:**

```bash
# Login
huggingface-cli login

# Check
huggingface-cli whoami

# Logout
huggingface-cli logout

# Training with auto-login
python3 train_mtup.py --use-case quick_test

# Manual login script
python3 hf_login.py

# Check in Python
python3 -c "from hf_auth import ensure_hf_login; ensure_hf_login()"
```

---

## 🎯 **RECOMMENDED WORKFLOW**

**Setup lần đầu trên server:**
```bash
# Step 1: SSH
ssh user@server

# Step 2: Clone code
cd ~/ViSemPar_new1
git pull

# Step 3: Login HuggingFace (ONE TIME)
huggingface-cli login
# Paste token: hf_xxxxx

# Step 4: Verify
huggingface-cli whoami

# Step 5: Training
python3 train_mtup.py --use-case quick_test
# → Auto-detect login, no config needed!

# Step 6: Full training
tmux new -s amr-training
python3 train_mtup.py --use-case full_training
```

**Lần sau chỉ cần:**
```bash
cd ~/ViSemPar_new1
git pull
python3 train_mtup.py --use-case full_training
# → Auto-login, không cần làm gì!
```

---

## 🎉 **DONE!**

Với CLI login, bạn chỉ cần login **1 LẦN DUY NHẤT** trên server.
Sau đó mọi script tự động sử dụng token đã lưu!

**Simple & Secure!** 🔐

---

**Get token:** https://huggingface.co/settings/tokens
**Need help?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
