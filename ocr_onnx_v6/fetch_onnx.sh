#!/usr/bin/env bash
# Fetch PRE-CONVERTED PP-OCRv6_small ONNX models (detection + recognition)
# for the rs_pdf_core "ppocrv6-small" model profile.
#
# These are PaddleX official *_onnx_infer tarballs — they already contain a
# converted .onnx, so NO paddle2onnx / paddlepaddle is required.
#
# Models:
#   PP-OCRv6_small_det  — DB text detector (PPLCNetV4 + RepLKFPN, language-agnostic)
#   PP-OCRv6_small_rec  — CTC text recognizer (unified 50-language model)
#
# Outputs:
#   output/ocr_detect.onnx
#   output/ocr_rec.onnx
#   output/ppocr_keys_v6.txt      (copied from ../ppocr/utils/dict/ppocrv6_dict.txt)
#   rec_preprocess.json           (committed; rec input geometry)
#
# Next: copy into rs-pdf-core/runtime-ppocrv6-small/ :
#   output/ocr_detect.onnx           -> runtime-ppocrv6-small/ocr_detect/ocr_detect.onnx
#   output/ocr_detect_labels.txt     -> runtime-ppocrv6-small/ocr_detect/labels.txt
#   output/ocr_rec.onnx              -> runtime-ppocrv6-small/ocr_rec/en/ocr_rec.onnx
#   output/ppocr_keys_v6.txt         -> runtime-ppocrv6-small/ocr_rec/en/ppocr_keys_v6.txt
#   rec_preprocess.json              -> runtime-ppocrv6-small/ocr_rec/en/preprocess.json
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

fetch_onnx "PP-OCRv6_small_det_onnx_infer" "$OUT/ocr_detect.onnx"
fetch_onnx "PP-OCRv6_small_rec_onnx_infer" "$OUT/ocr_rec.onnx"

# Recognition dictionary: PP-OCRv6_small_rec is the UNIFIED 50-LANGUAGE model,
# so it needs the full ppocrv6_dict.txt (18708 chars). rs_pdf_core validates
# dict.len()+1 against the model's output dim at first inference.
DICT_SRC="$HERE/../ppocr/utils/dict/ppocrv6_dict.txt"
if [ -f "$DICT_SRC" ]; then
  cp "$DICT_SRC" "$OUT/ppocr_keys_v6.txt"
  echo "  -> $OUT/ppocr_keys_v6.txt ($(wc -l < "$OUT/ppocr_keys_v6.txt") chars, 50-language unified)"
else
  echo "⚠ dict not found at $DICT_SRC — copy ppocrv6_dict.txt manually"
fi

# The OcrDetector also requires a one-class labels.txt ("text") next to the
# detector.
printf 'text\n' > "$OUT/ocr_detect_labels.txt"

echo "✓ done."
echo
echo "Verify det preprocessing against rs_pdf_core's hardcoded values"
echo "(max_side 960, pad multiple 32, ImageNet mean/std) before staging:"
echo "  cat output/raw/PP-OCRv6_small_det_onnx_infer/inference.yml"
echo
echo "Copy into rs-pdf-core/runtime-ppocrv6-small/ :"
echo "  cp output/ocr_detect.onnx        \$P/ocr_detect/ocr_detect.onnx"
echo "  cp output/ocr_detect_labels.txt  \$P/ocr_detect/labels.txt"
echo "  cp output/ocr_rec.onnx           \$P/ocr_rec/en/ocr_rec.onnx"
echo "  cp output/ppocr_keys_v6.txt      \$P/ocr_rec/en/ppocr_keys_v6.txt"
echo "  cp rec_preprocess.json           \$P/ocr_rec/en/preprocess.json"
