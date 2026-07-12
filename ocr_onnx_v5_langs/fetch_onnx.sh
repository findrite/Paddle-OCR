#!/usr/bin/env bash
# Fetch PRE-CONVERTED PP-OCRv5 per-script RECOGNITION models (official PaddleX
# *_onnx_infer tarballs) for the rs_pdf_core language folders.
#
# Usage:
#   bash fetch_onnx.sh devanagari    # Hindi/Marathi/Nepali/Sanskrit/... (Devanagari script)
#   bash fetch_onnx.sh ta            # Tamil
#   bash fetch_onnx.sh te            # Telugu
#   bash fetch_onnx.sh all           # all of the above
#
# Conversion note: the PaddleX bucket publishes an official paddle2onnx
# conversion of every family (<name>_onnx_infer.tar) alongside the Paddle
# inference tarball (<name>_infer.tar). This repo's standard workflow
# (ocr_onnx_v5/, ocr_onnx_v6/) uses the official pre-converted ONNX — no
# local paddle2onnx, which the ocr_onnx/ and layout READMEs document as
# fragile on this machine. Same source, same bytes PaddleX would produce.
#
# Outputs (per family):
#   output/<family>/ocr_rec.onnx
#   output/<family>/dict.txt              (repo dict, see DICTS below)
#   output/<family>/preprocess.json       (from the tarball's inference.yml)
#   output/<family>/checksums.sha256      (archive + onnx + dict)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$HERE/output/raw"; mkdir -p "$RAW"

# If a URL 404s, try swapping paddle3.0.0 <-> paddlex3.0.0 in BASE.
BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"

# family -> repo dictionary (the dict named by each model's training config;
# rs_pdf_core validates dict+space+blank against the ONNX vocab at first use).
dict_for () {
  case "$1" in
    devanagari) echo "ppocrv5_devanagari_dict.txt" ;;
    ta)         echo "ppocrv5_ta_dict.txt" ;;
    te)         echo "ppocrv5_te_dict.txt" ;;
    *) echo "unknown family '$1' (use devanagari|ta|te|all)" >&2; exit 2 ;;
  esac
}

fetch_family () {
  local fam="$1"
  local model="${fam}_PP-OCRv5_mobile_rec"
  local name="${model}_onnx_infer"
  local out="$HERE/output/$fam"; mkdir -p "$out"
  local tar="$RAW/$name.tar" dir="$RAW/$name"

  echo "▶ [$fam] downloading $name.tar"
  [ -f "$tar" ] || curl -fSL --retry 2 -o "$tar" "$BASE/$name.tar"
  tar -tf "$tar" > /dev/null   # fail loudly on a corrupt/non-tar download
  [ -d "$dir" ] || tar -xf "$tar" -C "$RAW"

  local onnx; onnx="$(find "$dir" -name '*.onnx' -print -quit)"
  [ -n "$onnx" ] || { echo "✗ no .onnx inside $name.tar" >&2; exit 1; }
  cp "$onnx" "$out/ocr_rec.onnx"

  local dict_name; dict_name="$(dict_for "$fam")"
  local dict_src="$HERE/../ppocr/utils/dict/$dict_name"
  [ -f "$dict_src" ] || { echo "✗ dict not found: $dict_src" >&2; exit 1; }
  cp "$dict_src" "$out/dict.txt"

  # Checksums: archive, onnx, dict (repo convention: models themselves are
  # not committed; checksums pin provenance).
  ( cd "$RAW" && shasum -a 256 "$name.tar" ) >  "$out/checksums.sha256"
  ( cd "$out" && shasum -a 256 ocr_rec.onnx dict.txt ) >> "$out/checksums.sha256"

  echo "  -> $out/ocr_rec.onnx ($(du -h "$out/ocr_rec.onnx" | cut -f1))"
  echo "  -> $out/dict.txt ($(wc -l < "$out/dict.txt") entries)"
  echo "  -> inference.yml at $dir/inference.yml (source of preprocess values)"
}

case "${1:-all}" in
  all) for f in devanagari ta te; do fetch_family "$f"; done ;;
  *)   fetch_family "$1" ;;
esac

echo "✓ done."
echo "Next: write output/<family>/preprocess.json from each inference.yml"
echo "(do NOT assume identical preprocessing across families), then stage:"
echo "  P=../../rs-pdf-core/runtime-ppocrv6-small   # and runtime-ppocrv6-medium"
echo "  for f in devanagari ta te; do"
echo "    mkdir -p \$P/ocr_rec/\$f"
echo "    cp output/\$f/ocr_rec.onnx      \$P/ocr_rec/\$f/ocr_rec.onnx"
echo "    cp output/\$f/dict.txt          \$P/ocr_rec/\$f/\$(basename ppocrv5_\${f}_dict.txt)"
echo "    cp output/\$f/preprocess.json   \$P/ocr_rec/\$f/preprocess.json"
echo "  done"
