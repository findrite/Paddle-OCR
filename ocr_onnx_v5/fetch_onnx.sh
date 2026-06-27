#!/usr/bin/env bash
# Fetch PRE-CONVERTED PP-OCRv5 mobile ONNX models (detection + recognition)
# for the rs_pdf_core "paddle-latest-mobile" model profile.
#
# These are PaddleX official *_onnx_infer tarballs — they already contain a
# converted .onnx, so NO paddle2onnx / paddlepaddle is required.
#
# Models:
#   PP-OCRv5_mobile_det  — DB text detector (language-agnostic)
#   PP-OCRv5_mobile_rec  — CTC text recognizer (English dict, 436 chars)
#
# Outputs:
#   output/ocr_detect.onnx
#   output/ocr_rec.onnx
#   output/ppocr_keys_v5_en.txt   (copied from ../ppocr/utils/dict/ppocrv5_en_dict.txt)
#   output/rec_preprocess.json    (committed; rec input geometry)
#
# Next: copy into rs-pdf-core/runtime-paddle-latest-mobile/ :
#   output/ocr_detect.onnx           -> runtime-paddle-latest-mobile/ocr_detect/ocr_detect.onnx
#   output/ocr_rec.onnx              -> runtime-paddle-latest-mobile/ocr_rec/en/ocr_rec.onnx
#   output/ppocr_keys_v5_en.txt      -> runtime-paddle-latest-mobile/ocr_rec/en/ppocr_keys_v5_en.txt
#   output/rec_preprocess.json       -> runtime-paddle-latest-mobile/ocr_rec/en/preprocess.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; RAW="$OUT/raw"; mkdir -p "$RAW"

# PaddleX official inference-model bucket. If a URL 404s, try swapping
# paddle3.0.0 <-> paddlex3.0.0 in BASE (the bucket has been published under both).
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

fetch_onnx "PP-OCRv5_mobile_det_onnx_infer" "$OUT/ocr_detect.onnx"
fetch_onnx "PP-OCRv5_mobile_rec_onnx_infer" "$OUT/ocr_rec.onnx"

# Recognition dictionary: PP-OCRv5 English keys (ships in this repo).
DICT_SRC="$HERE/../ppocr/utils/dict/ppocrv5_en_dict.txt"
if [ -f "$DICT_SRC" ]; then
  cp "$DICT_SRC" "$OUT/ppocr_keys_v5_en.txt"
  echo "  -> $OUT/ppocr_keys_v5_en.txt ($(wc -l < "$OUT/ppocr_keys_v5_en.txt") chars)"
else
  echo "⚠ dict not found at $DICT_SRC — copy ppocrv5_en_dict.txt manually"
fi

echo "✓ done. See header for where to copy each file."
