#!/usr/bin/env bash
# Download PP-DocLayout_plus-L Paddle inference model and convert to ONNX.
#
# Outputs:
#   output/layout.onnx
#   output/labels.txt
#
# Requires: paddlepaddle, paddle2onnx, PyYAML (see requirements.txt).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"
RAW="$OUT/raw"
mkdir -p "$RAW"

MODEL_NAME="PP-DocLayout_plus-L_infer"
# Official PaddleOCR / PaddleX BOS bucket. If this URL ever 404s, the
# PaddleOCR PP-StructureV3 release notes always link to the current
# inference-model tarball; update MODEL_URL below.
MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddlex3.0.0/${MODEL_NAME}.tar"
TAR="$RAW/${MODEL_NAME}.tar"
EXTRACT_DIR="$RAW/${MODEL_NAME}"

echo "▶ downloading $MODEL_URL"
if [ ! -f "$TAR" ]; then
  curl -fSL "$MODEL_URL" -o "$TAR"
fi

echo "▶ extracting"
if [ ! -d "$EXTRACT_DIR" ]; then
  tar -xf "$TAR" -C "$RAW"
fi

# Locate inference model files (.pdmodel/.pdiparams) and the inference.yml.
INFER_MODEL="$(find "$EXTRACT_DIR" -name 'inference.pdmodel' -print -quit)"
INFER_PARAMS="$(find "$EXTRACT_DIR" -name 'inference.pdiparams' -print -quit)"
INFER_YML="$(find "$EXTRACT_DIR" -name 'inference.yml' -print -quit || true)"

if [ -z "$INFER_MODEL" ] || [ -z "$INFER_PARAMS" ]; then
  echo "✗ inference.pdmodel / inference.pdiparams not found inside $EXTRACT_DIR"
  exit 1
fi

MODEL_DIR="$(dirname "$INFER_MODEL")"
echo "▶ inference model dir: $MODEL_DIR"

echo "▶ converting to ONNX (paddle2onnx, opset 11)"
paddle2onnx \
  --model_dir "$MODEL_DIR" \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file "$OUT/layout.onnx" \
  --opset_version 11 \
  --enable_dev_version True \
  --enable_onnx_checker True

echo "▶ writing labels.txt from inference.yml"
python "$HERE/extract_labels.py" "$INFER_YML" "$OUT/labels.txt"

echo "✓ done"
echo "  - $OUT/layout.onnx"
echo "  - $OUT/labels.txt"
echo
echo "Next: copy these two files into rs-pdf-core/runtime/layout/"
