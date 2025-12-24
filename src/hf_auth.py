"""
HuggingFace Authentication Helper for Training
Handles automatic login before training starts
"""

import os
import sys
from pathlib import Path


def ensure_hf_login(require_write=True):
    """
    Ensure user is logged in to HuggingFace before training

    Args:
        require_write: Whether to require write permission (for push to hub)

    Returns:
        bool: True if login successful or already logged in

    Raises:
        SystemExit: If login fails and is required
    """
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    # Check if already logged in
    try:
        api = HfApi()
        user_info = api.whoami()

        print("✅ Already logged in to HuggingFace")
        print(f"   User: {user_info['name']}")

        return True

    except Exception:
        # Not logged in, need to login
        print("⚠️  Not logged in to HuggingFace")
        print("Attempting automatic login...")

        # Try to get token from environment or .env
        token = get_hf_token()

        if token:
            try:
                login(token=token, add_to_git_credential=False)

                # Verify login
                user_info = api.whoami()
                print("✅ Login successful")
                print(f"   User: {user_info['name']}")

                return True

            except Exception as e:
                print(f"❌ Login failed: {e}")
                print("\nPlease login manually:")
                print("  Method 1 (CLI): huggingface-cli login")
                print("  Method 2 (Script): python3 hf_login.py")
                sys.exit(1)
        else:
            print("\n❌ No HuggingFace token found")
            print("\nPlease login using one of these methods:")
            print("\n1. CLI Login (RECOMMENDED):")
            print("   huggingface-cli login")
            print("   # Then paste your token")
            print("\n2. Environment Variable:")
            print("   export HF_TOKEN=your_token_here")
            print("\n3. .env File:")
            print("   echo 'HF_TOKEN=your_token_here' > .env")
            print("\n4. Script:")
            print("   python3 hf_login.py")
            print("\nGet token from: https://huggingface.co/settings/tokens")
            print("(Needs 'Write' permission)\n")

            sys.exit(1)


def get_hf_token():
    """
    Get HuggingFace token from various sources

    Priority:
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable
    3. .env file
    4. Return None (will need manual login)

    Returns:
        str or None: Token if found
    """
    # Try environment variables
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    if token:
        return token

    # Try .env file
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            token = os.getenv('HF_TOKEN')
            if token:
                return token
        except ImportError:
            # python-dotenv not installed, skip
            pass

    return None


def check_hf_token_valid(token):
    """
    Check if HuggingFace token is valid

    Args:
        token: HF token string

    Returns:
        bool: True if valid
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami(token=token)
        return True

    except Exception:
        return False


def get_hf_username():
    """
    Get logged-in HuggingFace username

    Returns:
        str or None: Username if logged in
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user_info = api.whoami()
        return user_info['name']

    except Exception:
        return None


def setup_hf_cache_dir(cache_dir=None):
    """
    Setup HuggingFace cache directory

    Args:
        cache_dir: Custom cache directory (optional)

    Returns:
        Path: Cache directory path
    """
    if cache_dir is None:
        # Use default: ~/.cache/huggingface
        cache_dir = Path.home() / '.cache' / 'huggingface'
    else:
        cache_dir = Path(cache_dir)

    # Create if not exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable
    os.environ['HF_HOME'] = str(cache_dir)

    return cache_dir


if __name__ == "__main__":
    # Test authentication
    print("Testing HuggingFace Authentication...")
    ensure_hf_login()
