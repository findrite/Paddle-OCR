#!/usr/bin/env bash
# Fetch + convert PP-OCRv3 per-script RECOGNITION models for the rs_pdf_core
# language folders. Unlike the v5 families (ocr_onnx_v5_langs/), the v3
# multilingual models have NO official pre-converted `_onnx_infer.tar` in the
# PaddleX bucket, and the PaddleX re-release ships the PIR format
# (inference.json) that only arm64-broken paddle2onnx 2.x can read on this
# machine. So this script uses the LEGACY PaddleOCR release (inference.pdmodel)
# and converts locally with paddle2onnx 1.0.9 (true macOS x86_64 wheel).
#
# Usage:
#   bash fetch_onnx.sh ka            # Kannada
#   bash fetch_onnx.sh all
#
# One-time venv (python3.9 x86_64):
#   python3 -m venv .venv && ./.venv/bin/pip install "paddle2onnx==1.0.9" six
#
# Outputs (per family):
#   output/<family>/ocr_rec.onnx
#   output/<family>/dict.txt          repo dict + ONE trailing space row —
#                                     Paddle's CTCLabelDecode(use_space_char)
#                                     appends a space AFTER the dict, so the
#                                     model vocab is dict + space + blank and
#                                     rs_pdf_core requires dict.len()+1 == V.
#   output/<family>/preprocess.json   target_height 48, mean/std 0.5 (v3 rec)
#   output/<family>/checksums.sha256
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$HERE/output/raw"; mkdir -p "$RAW"

# Legacy (pre-PIR) release bucket — tarballs contain inference.pdmodel.
BASE="https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual"

dict_for () {
  case "$1" in
    ka) echo "ka_dict.txt" ;;
    *) echo "unknown family '$1' (use ka|all)" >&2; exit 2 ;;
  esac
}

fetch_family () {
  local fam="$1"
  local name="${fam}_PP-OCRv3_rec_infer"
  local out="$HERE/output/$fam"; mkdir -p "$out"
  local tar="$RAW/$name.tar" dir="$RAW/$name"

  echo "▶ [$fam] downloading $name.tar"
  [ -f "$tar" ] || curl -fSL --retry 2 -o "$tar" "$BASE/$name.tar"
  tar -tf "$tar" > /dev/null
  [ -d "$dir" ] || tar -xf "$tar" -C "$RAW"

  echo "▶ [$fam] converting to ONNX (paddle2onnx 1.0.9, opset 14)"
  "$HERE/.venv/bin/paddle2onnx" \
    --model_dir "$dir" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file "$out/ocr_rec.onnx" \
    --opset_version 14

  local dict_name; dict_name="$(dict_for "$fam")"
  local dict_src="$HERE/../ppocr/utils/dict/$dict_name"
  [ -f "$dict_src" ] || { echo "✗ dict not found: $dict_src" >&2; exit 1; }
  cp "$dict_src" "$out/dict.txt"
  printf ' \n' >> "$out/dict.txt"   # use_space_char row (see header comment)

  ( cd "$RAW" && shasum -a 256 "$name.tar" ) >  "$out/checksums.sha256"
  ( cd "$out" && shasum -a 256 ocr_rec.onnx dict.txt ) >> "$out/checksums.sha256"

  echo "  -> $out/ocr_rec.onnx ($(du -h "$out/ocr_rec.onnx" | cut -f1))"
  echo "  -> $out/dict.txt ($(wc -l < "$out/dict.txt") rows incl. space)"
}

case "${1:-all}" in
  all) fetch_family ka ;;
  *)   fetch_family "$1" ;;
esac

echo "✓ done. Stage output/<family>/ into rs-pdf-core/runtime-*/ocr_rec/<family>/"
