#!/usr/bin/env bash
# Fetch a pre-converted PaddleOCR text-detection ONNX model from HuggingFace.
#
# Default model: PP-OCRv3 mobile det (monkt/paddleocr-onnx mirror)
# - Architecture: DB (Differentiable Binarization)
# - Training data: PaddleOCR text detection
# - Classes (1): text
# - Input: (1, 3, H, W) float32 RGB, normalised (ImageNet mean/std),
#           max side 960px, padded to multiple of 32
# - Output: (1, 1, H, W) float32 — sigmoid probability map
#
# Outputs:
#   output/ocr_detect.onnx
#   output/labels.txt   (already committed; single class: text)
#
# The model is downloaded pre-converted from HuggingFace. No paddle2onnx
# or PaddlePaddle installation required.
#
# If you need to convert from PaddleOCR's native format instead, use
# download_and_convert.sh (requires paddle2onnx + paddlepaddle).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"
mkdir -p "$OUT"

URL="https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v3/det.onnx"
DEST="$OUT/ocr_detect.onnx"

if [ -f "$DEST" ] && [ "$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")" -gt 1000000 ]; then
  echo "✓ $DEST already present ($(du -h "$DEST" | cut -f1))"
else
  echo "▶ downloading $URL"
  curl -fSL --retry 2 -o "$DEST" "$URL"
fi

echo "✓ done"
echo "  - $DEST"
echo "  - $OUT/labels.txt (committed, 1 class: text)"
echo
echo "Next: copy these two files into rs-pdf-core/runtime/ocr_detect/"
