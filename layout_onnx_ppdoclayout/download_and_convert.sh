#!/usr/bin/env bash
# Download PP-DocLayout_plus-L (PaddleX) layout model and convert to ONNX for
# the rs_pdf_core "paddle-latest-mobile" model profile.
#
# Versioned copy — does NOT replace ../layout_onnx (which stays YOLOv10b for the
# legacy-stable profile).
#
# ⚠ PP-DocLayout is a different architecture from YOLOv10b: its ONNX output
#   tensor shape and class set differ, so the rs_pdf_core layout pipeline needs
#   the Phase-6 adapter (layout_preprocess.json + output adapter) before this
#   model can be consumed. This script just produces the artifacts.
#
# Requires: paddlepaddle, paddle2onnx, PyYAML.
#
# Outputs:
#   output/layout.onnx
#   output/labels.txt   (PP-DocLayout_plus-L class set, ~20+ classes)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; RAW="$OUT/raw"; mkdir -p "$RAW"

MODEL_NAME="PP-DocLayout_plus-L_infer"
# If this 404s, swap paddle3.0.0 <-> paddlex3.0.0.
MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/${MODEL_NAME}.tar"
TAR="$RAW/${MODEL_NAME}.tar"; EXTRACT_DIR="$RAW/${MODEL_NAME}"

command -v paddle2onnx >/dev/null 2>&1 || { echo "✗ paddle2onnx not found (pip install -r requirements.txt)"; exit 1; }

echo "▶ downloading $MODEL_URL"
[ -f "$TAR" ] || curl -fSL "$MODEL_URL" -o "$TAR"
[ -d "$EXTRACT_DIR" ] || tar -xf "$TAR" -C "$RAW"

INFER_MODEL="$(find "$EXTRACT_DIR" -name 'inference.pdmodel' -print -quit)"
INFER_YML="$(find "$EXTRACT_DIR" -name 'inference.yml' -print -quit || true)"
[ -n "$INFER_MODEL" ] || { echo "✗ inference.pdmodel not found"; exit 1; }
MODEL_DIR="$(dirname "$INFER_MODEL")"

echo "▶ converting to ONNX (paddle2onnx, opset 11)"
paddle2onnx --model_dir "$MODEL_DIR" \
  --model_filename inference.pdmodel --params_filename inference.pdiparams \
  --save_file "$OUT/layout.onnx" --opset_version 11 --enable_onnx_checker True

echo "▶ writing labels.txt from inference.yml"
python "$HERE/extract_labels.py" "$INFER_YML" "$OUT/labels.txt"

echo "✓ done"
echo "  - $OUT/layout.onnx"
echo "  - $OUT/labels.txt  ($(wc -l < "$OUT/labels.txt" 2>/dev/null || echo '?') classes)"
echo
echo "Phase 6: once the rs_pdf_core PP-DocLayout adapter lands, copy into:"
echo "  cp output/layout.onnx output/labels.txt ../../rs-pdf-core/runtime-paddle-latest-mobile/layout/"
