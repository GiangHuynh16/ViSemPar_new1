#!/usr/bin/env python3
"""
Decode base64 files exported from server and calculate SMATCH
"""

import base64
import sys
from pathlib import Path


def decode_base64_file(b64_string, output_path):
    """Decode base64 string and save to file with UTF-8 encoding"""
    try:
        # Remove any whitespace/newlines
        b64_string = b64_string.strip()

        # Decode base64
        decoded_bytes = base64.b64decode(b64_string)

        # Write to file with UTF-8 encoding
        with open(output_path, 'wb') as f:
            f.write(decoded_bytes)

        print(f"✓ Decoded: {output_path} ({len(decoded_bytes)} bytes)")

        # Verify Vietnamese text is readable
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(f"  First line: {first_line[:100]}...")

        return True
    except Exception as e:
        print(f"✗ Error decoding {output_path}: {e}")
        return False


def main():
    print("=" * 70)
    print("BASE64 FILE DECODER - Vietnamese AMR Evaluation")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Decode 3 base64-encoded files")
    print("  2. Save them with UTF-8 encoding")
    print("  3. Optionally calculate SMATCH score")
    print()
    print("Please paste the base64 strings when prompted.")
    print("Press Ctrl+D (Unix/Mac) or Ctrl+Z (Windows) when done with each input.")
    print()

    files_to_decode = [
        {
            'name': 'predictions_formatted.txt',
            'prompt': 'Paste base64 for predictions_formatted.txt (between BEGIN and END markers):',
            'marker_start': '--- BEGIN PREDICTIONS_FORMATTED ---',
            'marker_end': '--- END PREDICTIONS_FORMATTED ---'
        },
        {
            'name': 'ground_truth.txt',
            'prompt': 'Paste base64 for ground_truth.txt (between BEGIN and END markers):',
            'marker_start': '--- BEGIN GROUND_TRUTH ---',
            'marker_end': '--- END GROUND_TRUTH ---'
        },
        {
            'name': 'result_baseline.txt',
            'prompt': 'Paste base64 for result_baseline.txt (between BEGIN and END markers):',
            'marker_start': '--- BEGIN RESULT_BASELINE ---',
            'marker_end': '--- END RESULT_BASELINE ---'
        }
    ]

    decoded_files = []

    for file_info in files_to_decode:
        print("-" * 70)
        print(file_info['prompt'])
        print()

        # Read multiline input
        lines = []
        try:
            while True:
                line = input()
                # Skip marker lines
                if file_info['marker_start'] in line or file_info['marker_end'] in line:
                    continue
                # Skip empty lines at start
                if not line and not lines:
                    continue
                lines.append(line)
        except EOFError:
            pass

        if not lines:
            print(f"⚠️  No input provided for {file_info['name']}, skipping...")
            continue

        # Combine all lines
        b64_string = ''.join(lines)

        # Decode and save
        output_path = file_info['name']
        if decode_base64_file(b64_string, output_path):
            decoded_files.append(output_path)

        print()

    print()
    print("=" * 70)
    print("DECODING COMPLETE")
    print("=" * 70)
    print()
    print(f"Successfully decoded {len(decoded_files)}/{len(files_to_decode)} files:")
    for f in decoded_files:
        size = Path(f).stat().st_size
        print(f"  ✓ {f} ({size:,} bytes)")
    print()

    # Ask if user wants to calculate SMATCH
    if len(decoded_files) >= 2:
        print("=" * 70)
        print("CALCULATE SMATCH?")
        print("=" * 70)
        print()

        try:
            import smatch
            print("✓ smatch package found")

            response = input("Calculate SMATCH score now? (y/n): ").strip().lower()

            if response == 'y':
                print()
                print("Calculating SMATCH...")
                print("(This may take a few minutes...)")
                print()

                # Run smatch
                import subprocess
                result = subprocess.run(
                    [sys.executable, '-m', 'smatch', '-f',
                     'predictions_formatted.txt', 'ground_truth.txt',
                     '--significant', '4'],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                print(result.stdout)

                if result.stderr:
                    # Only show relevant errors
                    stderr_lines = result.stderr.split('\n')
                    important_errors = [line for line in stderr_lines
                                       if 'ERROR' in line or 'Failed' in line]
                    if important_errors:
                        print("Errors:")
                        for line in important_errors:
                            print(f"  {line}")

                # Extract F1 score
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'F-score:' in line or 'F1:' in line or 'Smatch' in line:
                        print()
                        print("=" * 70)
                        print("FINAL RESULT")
                        print("=" * 70)
                        print(line)
                        print("=" * 70)

        except ImportError:
            print("⚠️  smatch package not installed")
            print()
            print("To calculate SMATCH, install smatch:")
            print("  pip install smatch")
            print()
            print("Then run:")
            print("  python -m smatch -f predictions_formatted.txt ground_truth.txt --significant 4")

        except Exception as e:
            print(f"⚠️  Error calculating SMATCH: {e}")
            print()
            print("You can manually run:")
            print("  python -m smatch -f predictions_formatted.txt ground_truth.txt --significant 4")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
