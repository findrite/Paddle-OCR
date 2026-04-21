#!/usr/bin/env bash
# Download PaddleOCR PP-OCRv3 det model and convert to ONNX.
#
# This is the full conversion path that requires paddle2onnx + paddlepaddle.
# For a simpler path, see fetch_onnx.sh which is the default entry point.
#
# Prerequisites:
#   pip install paddlepaddle paddle2onnx onnx onnxruntime
#
# Usage:
#   cd Paddle-OCR/ocr_onnx
#   bash download_and_convert.sh
#   cp output/ocr_detect.onnx output/labels.txt ../../rs-pdf-core/runtime/ocr_detect/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"
mkdir -p "$OUT"

DEST="$OUT/ocr_detect.onnx"

# --- PP-OCRv3 English detection model ---
PADDLE_URL="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
TAR_FILE="$OUT/en_PP-OCRv3_det_infer.tar"
MODEL_DIR="$OUT/en_PP-OCRv3_det_infer"

if [ -f "$DEST" ]; then
  echo "✓ $DEST already exists ($(du -h "$DEST" | cut -f1))"
  echo "  Delete it and re-run to force re-conversion."
  exit 0
fi

echo "▶ step 1/3: downloading PaddleOCR PP-OCRv3 det inference model…"
curl -fSL --retry 2 -o "$TAR_FILE" "$PADDLE_URL"
tar -xf "$TAR_FILE" -C "$OUT"

echo "▶ step 2/3: converting .pdmodel → .onnx via paddle2onnx…"
paddle2onnx \
  --model_dir "$MODEL_DIR" \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file "$DEST" \
  --opset_version 11 \
  --enable_onnx_checker True

echo "▶ step 3/3: verifying ONNX model…"
python3 -c "
import onnx
model = onnx.load('$DEST')
onnx.checker.check_model(model)
print(f'  inputs:  {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]}')
print(f'  outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]}')
"

# Clean up
rm -rf "$TAR_FILE" "$MODEL_DIR"

echo
echo "✓ conversion complete"
echo "  $DEST ($(du -h "$DEST" | cut -f1))"
echo
echo "Copy to runtime:"
echo "  cp $DEST $OUT/labels.txt ../../rs-pdf-core/runtime/ocr_detect/"
