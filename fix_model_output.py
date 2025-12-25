#!/usr/bin/env python3
"""
Fix model output to valid AMR format
"""

import re

def fix_incomplete_amr(amr_string: str) -> str:
    """
    Fix common issues in model-generated AMR:
    1. Missing opening parenthesis at start
    2. Unbalanced parentheses
    3. Extra closing parentheses
    """
    amr = amr_string.strip()

    # If doesn't start with '(', add it
    if amr and not amr.startswith('('):
        amr = '(' + amr

    # Count parentheses
    open_count = amr.count('(')
    close_count = amr.count(')')

    # Balance parentheses
    if open_count > close_count:
        # Add missing closing parens
        amr += ')' * (open_count - close_count)
    elif close_count > open_count:
        # Add missing opening parens at start
        amr = '(' * (close_count - open_count) + amr

    return amr

def validate_amr_basic(amr_string: str) -> bool:
    """
    Basic validation:
    - Has balanced parentheses
    - Has at least one concept
    """
    if not amr_string or not amr_string.strip():
        return False

    open_count = amr_string.count('(')
    close_count = amr_string.count(')')

    return open_count == close_count and open_count > 0

# Test cases
if __name__ == "__main__":
    test_cases = [
        "ăn:agent(tôi):patient(cơm))",  # Missing opening paren
        "(n1:topic(n2:agent(",  # Missing closing parens
        "w1:ARG1(c:topic(",  # Multiple issues
        "(x:domain(sx)",  # Missing closing paren
    ]

    print("="*80)
    print("TESTING AMR FIX")
    print("="*80)

    for i, test_amr in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input:  {test_amr}")

        fixed = fix_incomplete_amr(test_amr)
        print(f"Fixed:  {fixed}")

        is_valid = validate_amr_basic(fixed)
        print(f"Valid:  {'✓' if is_valid else '✗'}")

        open_c = fixed.count('(')
        close_c = fixed.count(')')
        print(f"Parens: {open_c} open, {close_c} close")
