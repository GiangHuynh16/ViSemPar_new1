#!/usr/bin/env python3
"""
Quick script to verify HuggingFace authentication method
Run this on the server to confirm CLI login is working
"""

from huggingface_hub import HfApi
import os
from pathlib import Path

print("=" * 80)
print("HUGGINGFACE AUTHENTICATION CHECK")
print("=" * 80)
print()

# Check 1: CLI token file
print("1. Checking CLI token file...")
cli_token_path = Path.home() / ".cache" / "huggingface" / "token"
if cli_token_path.exists():
    print(f"   ✅ CLI token found: {cli_token_path}")
    with open(cli_token_path) as f:
        token_preview = f.read().strip()[:20]
    print(f"   Token starts with: {token_preview}...")
else:
    print(f"   ❌ CLI token NOT found at: {cli_token_path}")
    print("   Run: huggingface-cli login")

print()

# Check 2: Environment variables
print("2. Checking environment variables...")
if "HF_TOKEN" in os.environ:
    print("   ⚠️  HF_TOKEN found in environment")
    print(f"   Value starts with: {os.environ['HF_TOKEN'][:20]}...")
else:
    print("   ✅ HF_TOKEN NOT in environment (good - using CLI)")

if "HUGGING_FACE_HUB_TOKEN" in os.environ:
    print("   ⚠️  HUGGING_FACE_HUB_TOKEN found in environment")
else:
    print("   ✅ HUGGING_FACE_HUB_TOKEN NOT in environment")

print()

# Check 3: .env file in project
print("3. Checking .env file...")
env_file = Path(".env")
if env_file.exists():
    print(f"   ⚠️  .env file exists: {env_file.absolute()}")
    with open(env_file) as f:
        content = f.read()
    if "HF_TOKEN" in content or "HUGGING_FACE" in content:
        print("   ⚠️  .env contains HuggingFace token")
        print("   This might interfere with CLI login!")
    else:
        print("   ✅ .env doesn't contain HF token")
else:
    print("   ✅ No .env file (good)")

print()

# Check 4: API authentication
print("4. Testing HuggingFace API authentication...")
try:
    api = HfApi()  # Should use CLI token automatically
    user = api.whoami()
    print(f"   ✅ Successfully authenticated!")
    print(f"   Username: {user['name']}")
    print(f"   Type: {user.get('type', 'unknown')}")

    # Check which token was used
    token_info = user.get('auth', {})
    print(f"   Auth method: {token_info.get('type', 'CLI token')}")

except Exception as e:
    print(f"   ❌ Authentication failed: {e}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if cli_token_path.exists() and "HF_TOKEN" not in os.environ:
    print("✅ Configuration looks good!")
    print("   Using CLI login (huggingface-cli login)")
    print("   push_to_hf_cli.py should work correctly")
elif "HF_TOKEN" in os.environ:
    print("⚠️  WARNING: Environment variable HF_TOKEN detected")
    print("   This might override CLI login")
    print("   Unset it with: unset HF_TOKEN")
elif env_file.exists():
    print("⚠️  WARNING: .env file detected")
    print("   Make sure it doesn't contain HF_TOKEN")
else:
    print("❌ No authentication found")
    print("   Run: huggingface-cli login")

print()
