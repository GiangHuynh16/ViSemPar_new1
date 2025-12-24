#!/usr/bin/env python3
"""
HuggingFace Login Helper
Programmatic login to HuggingFace Hub
"""

import os
import sys
from pathlib import Path


def login_huggingface(token=None, method='auto'):
    """
    Login to HuggingFace Hub

    Args:
        token: HF token string (optional, can read from .env or prompt)
        method: 'auto', 'token', 'cli', or 'env'

    Returns:
        bool: True if login successful
    """
    print("=" * 80)
    print("HUGGINGFACE LOGIN")
    print("=" * 80)

    # Method 1: Try to import huggingface_hub
    try:
        from huggingface_hub import login, HfApi
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("\nInstall with:")
        print("  pip install huggingface_hub")
        return False

    # Get token from different sources
    if token is None:
        if method == 'env' or method == 'auto':
            # Try from environment variable
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
            if token:
                print("✓ Found token from environment variable")

        if token is None and method in ['auto', 'token']:
            # Try from .env file
            env_file = Path(__file__).parent / '.env'
            if env_file.exists():
                from dotenv import load_dotenv
                load_dotenv(env_file)
                token = os.getenv('HF_TOKEN')
                if token:
                    print("✓ Found token from .env file")

        if token is None:
            # Prompt for token
            print("\n⚠️  No token found in environment or .env file")
            print("\nYou can:")
            print("  1. Enter token now (will be saved to cache)")
            print("  2. Set HF_TOKEN environment variable")
            print("  3. Create .env file with HF_TOKEN=your_token")
            print("\nGet token from: https://huggingface.co/settings/tokens")
            print("(Need 'Write' permission)")

            token = input("\nEnter HF token (or press Enter to skip): ").strip()

            if not token:
                print("❌ No token provided")
                return False

    # Perform login
    try:
        print("\n🔐 Logging in to HuggingFace Hub...")

        # Login with token
        login(token=token, add_to_git_credential=False)

        # Verify login
        api = HfApi()
        user_info = api.whoami()

        print("\n" + "=" * 80)
        print("✅ LOGIN SUCCESSFUL")
        print("=" * 80)
        print(f"Username: {user_info['name']}")
        print(f"Email:    {user_info.get('email', 'N/A')}")
        print(f"Type:     {user_info.get('type', 'user')}")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ LOGIN FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\nPossible issues:")
        print("  1. Invalid token")
        print("  2. Token doesn't have 'Write' permission")
        print("  3. Network connection issue")
        print("\nGet new token from: https://huggingface.co/settings/tokens")
        print("=" * 80)

        return False


def check_login_status():
    """Check if already logged in to HuggingFace"""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user_info = api.whoami()

        print("=" * 80)
        print("✅ ALREADY LOGGED IN")
        print("=" * 80)
        print(f"Username: {user_info['name']}")
        print(f"Email:    {user_info.get('email', 'N/A')}")
        print("=" * 80)

        return True

    except Exception:
        print("⚠️  Not logged in to HuggingFace")
        return False


def logout_huggingface():
    """Logout from HuggingFace"""
    try:
        from huggingface_hub import logout

        logout()
        print("✓ Logged out from HuggingFace")
        return True

    except Exception as e:
        print(f"❌ Logout failed: {e}")
        return False


def main():
    """Main function with CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace Login Helper")
    parser.add_argument('--token', type=str, help='HF token')
    parser.add_argument('--method', choices=['auto', 'token', 'cli', 'env'],
                       default='auto', help='Login method')
    parser.add_argument('--check', action='store_true', help='Check login status')
    parser.add_argument('--logout', action='store_true', help='Logout')

    args = parser.parse_args()

    if args.logout:
        logout_huggingface()
    elif args.check:
        check_login_status()
    else:
        login_huggingface(token=args.token, method=args.method)


if __name__ == "__main__":
    main()
