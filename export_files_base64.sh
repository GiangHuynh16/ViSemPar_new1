#!/bin/bash
# Export evaluation files as base64 for manual copy

echo "=========================================="
echo "BASELINE EVALUATION FILES - BASE64 EXPORT"
echo "=========================================="
echo ""
echo "Copy the base64 strings below and use the decode script on your local machine"
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

echo "=========================================="
echo "FILE 1: predictions_formatted.txt"
echo "=========================================="
base64 predictions_formatted.txt > predictions_formatted.txt.b64
echo "Encoded file size: $(wc -c < predictions_formatted.txt.b64) bytes"
echo ""
echo "--- BEGIN PREDICTIONS_FORMATTED ---"
cat predictions_formatted.txt.b64
echo "--- END PREDICTIONS_FORMATTED ---"
echo ""

echo "=========================================="
echo "FILE 2: data/public_test_ground_truth.txt"
echo "=========================================="
base64 data/public_test_ground_truth.txt > ground_truth.txt.b64
echo "Encoded file size: $(wc -c < ground_truth.txt.b64) bytes"
echo ""
echo "--- BEGIN GROUND_TRUTH ---"
cat ground_truth.txt.b64
echo "--- END GROUND_TRUTH ---"
echo ""

echo "=========================================="
echo "FILE 3: public_test_result_baseline_7b.txt"
echo "=========================================="
base64 public_test_result_baseline_7b.txt > result_baseline.txt.b64
echo "Encoded file size: $(wc -c < result_baseline.txt.b64) bytes"
echo ""
echo "--- BEGIN RESULT_BASELINE ---"
cat result_baseline.txt.b64
echo "--- END RESULT_BASELINE ---"
echo ""

echo "=========================================="
echo "EXPORT COMPLETE"
echo "=========================================="
echo ""
echo "To decode on your local machine, use:"
echo "  python decode_base64_files.py"
echo ""
