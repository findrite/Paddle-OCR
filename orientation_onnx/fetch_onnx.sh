#!/usr/bin/env bash
# Fetch PRE-CONVERTED PP-LCNet orientation ONNX models (textline + doc).
# Replaces the paddle2onnx conversion path in download_and_convert.sh —
# the PaddleX bucket now publishes *_onnx_infer tarballs, so this needs
# only curl + tar.
#
# Models:
#   PP-LCNet_x1_0_textline_ori — 2-class (0/180) text-line angle classifier
#                                -> drop-in for the legacy PP-OCR v2.0 cls
#   PP-LCNet_x1_0_doc_ori      — 4-class (0/90/180/270) page orientation
#                                -> Phase 6 (needs 4-class AngleClassifier)
#
# Outputs:
#   output/ocr_cls.onnx   (textline_ori — the wired one)
#   output/doc_ori.onnx   (doc_ori — for Phase 6)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; RAW="$OUT/raw"; mkdir -p "$RAW"

# If a URL 404s, try swapping paddle3.0.0 <-> paddlex3.0.0 in BASE.
BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"

fetch_onnx () {  # $1=model name (tar stem)  $2=dest .onnx
  local name="$1" dest="$2" tar="$RAW/$1.tar" dir="$RAW/$1"
  if [ -f "$dest" ] && [ "$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")" -gt 100000 ]; then
    echo "✓ $dest already present"; return
  fi
  echo "▶ downloading $name"
  [ -f "$tar" ] || curl -fSL --retry 2 -o "$tar" "$BASE/$name.tar"
  [ -d "$dir" ] || tar -xf "$tar" -C "$RAW"
  local onnx; onnx="$(find "$dir" -name '*.onnx' -print -quit)"
  [ -n "$onnx" ] || { echo "✗ no .onnx inside $name.tar"; exit 1; }
  cp "$onnx" "$dest"
  echo "  -> $dest"
}

fetch_onnx "PP-LCNet_x1_0_textline_ori_onnx_infer" "$OUT/ocr_cls.onnx"
fetch_onnx "PP-LCNet_x1_0_doc_ori_onnx_infer"      "$OUT/doc_ori.onnx"

echo "✓ done."
echo
echo "Check input geometry before staging (rs_pdf_core cls preprocessing):"
echo "  cat output/raw/PP-LCNet_x1_0_textline_ori_onnx_infer/inference.yml"
echo
echo "Copy into the runtime profiles:"
echo "  cp output/ocr_cls.onnx ../../rs-pdf-core/runtime-ppocrv6-small/ocr_cls/ocr_cls.onnx"
echo "  cp output/ocr_cls.onnx ../../rs-pdf-core/runtime-paddle-latest-mobile/ocr_cls/ocr_cls.onnx"
