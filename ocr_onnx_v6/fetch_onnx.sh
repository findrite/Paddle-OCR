#!/usr/bin/env bash
# Fetch PRE-CONVERTED PP-OCRv6 ONNX models (detection + recognition) for the
# rs_pdf_core "ppocrv6-<tier>" model profiles.
#
# Usage:
#   bash fetch_onnx.sh            # tier "small" (default)
#   bash fetch_onnx.sh medium     # tier "medium"
#   bash fetch_onnx.sh tiny       # tier "tiny"
#
# These are PaddleX official *_onnx_infer tarballs — they already contain a
# converted .onnx, so NO paddle2onnx / paddlepaddle is required.
#
# All tiers share the same DB detector + CTC recogniser architecture, the
# same 48x320 rec input geometry (rec_preprocess.json), and the same unified
# 50-language dict (18708 chars) — so one script serves the whole family.
#
# Outputs (per tier):
#   output/<tier>/ocr_detect.onnx
#   output/<tier>/ocr_rec.onnx
#   output/<tier>/ppocr_keys_v6.txt
#   output/<tier>/ocr_detect_labels.txt
#   output/<tier>/det_model_tag.txt   (one line, e.g. PP-OCRv6_medium_det)
set -euo pipefail

TIER="${1:-small}"
case "$TIER" in
  small|medium|tiny) ;;
  *) echo "usage: $0 [small|medium|tiny]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output/$TIER"; RAW="$HERE/output/raw"; mkdir -p "$OUT" "$RAW"

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

fetch_onnx "PP-OCRv6_${TIER}_det_onnx_infer" "$OUT/ocr_detect.onnx"
fetch_onnx "PP-OCRv6_${TIER}_rec_onnx_infer" "$OUT/ocr_rec.onnx"

# Recognition dictionary: every PP-OCRv6 tier is a UNIFIED 50-LANGUAGE model
# using ppocrv6_dict.txt (18708 chars). rs_pdf_core validates dict.len()+1
# against the model's output dim at first inference.
DICT_SRC="$HERE/../ppocr/utils/dict/ppocrv6_dict.txt"
if [ -f "$DICT_SRC" ]; then
  cp "$DICT_SRC" "$OUT/ppocr_keys_v6.txt"
  echo "  -> $OUT/ppocr_keys_v6.txt ($(wc -l < "$OUT/ppocr_keys_v6.txt") chars, 50-language unified)"
else
  echo "⚠ dict not found at $DICT_SRC — copy ppocrv6_dict.txt manually"
fi

# The OcrDetector requires a one-class labels.txt next to the detector, and
# reads an optional model_tag.txt to report the true model in API responses.
printf 'text\n' > "$OUT/ocr_detect_labels.txt"
printf 'PP-OCRv6_%s_det\n' "$TIER" > "$OUT/det_model_tag.txt"

echo "✓ done ($TIER)."
echo
echo "Copy into rs-pdf-core/runtime-ppocrv6-$TIER/ :"
echo "  P=../../rs-pdf-core/runtime-ppocrv6-$TIER"
echo "  cp output/$TIER/ocr_detect.onnx        \$P/ocr_detect/ocr_detect.onnx"
echo "  cp output/$TIER/ocr_detect_labels.txt  \$P/ocr_detect/labels.txt"
echo "  cp output/$TIER/det_model_tag.txt      \$P/ocr_detect/model_tag.txt"
echo "  cp output/$TIER/ocr_rec.onnx           \$P/ocr_rec/en/ocr_rec.onnx"
echo "  cp output/$TIER/ppocr_keys_v6.txt      \$P/ocr_rec/en/ppocr_keys_v6.txt"
echo "  sed 's/_small_/_${TIER}_/' rec_preprocess.json > \$P/ocr_rec/en/preprocess.json"
