#!/bin/bash
# Check Vietnamese encoding issues

echo "=========================================="
echo "VIETNAMESE ENCODING DIAGNOSTICS"
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

echo "1. Current locale settings:"
echo "=========================================="
locale
echo ""

echo "2. File encoding check:"
echo "=========================================="
file predictions_formatted.txt
file data/public_test_ground_truth.txt
echo ""

echo "3. First Vietnamese sentence from predictions:"
echo "=========================================="
head -n 5 predictions_formatted.txt
echo ""

echo "4. Environment variables:"
echo "=========================================="
echo "LANG=$LANG"
echo "LC_ALL=$LC_ALL"
echo "LC_CTYPE=$LC_CTYPE"
echo ""

echo "5. Test Vietnamese display:"
echo "=========================================="
echo "Test: Tiếng Việt có dấu - àáảãạ êôơư"
echo ""

echo "=========================================="
echo "RECOMMENDED FIX"
echo "=========================================="
echo ""
echo "If you see garbled text above, run these commands:"
echo ""
echo "  export LANG=en_US.UTF-8"
echo "  export LC_ALL=en_US.UTF-8"
echo ""
echo "Then try again: cat predictions_formatted.txt | head -n 20"
echo ""
