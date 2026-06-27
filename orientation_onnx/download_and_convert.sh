#!/usr/bin/env bash
# Download PP-LCNet orientation classifiers and convert to ONNX for the
# rs_pdf_core "paddle-latest-mobile" model profile.
#
#   PP-LCNet_x1_0_textline_ori  — TEXT-LINE orientation, 2 classes (0° / 180°).
#                                 Drop-in replacement for the legacy PP-OCR v2.0
#                                 angle classifier (ocr_cls).
#   PP-LCNet_x1_0_doc_ori       — PAGE orientation, 4 classes (0/90/180/270).
#                                 NOTE: rs_pdf_core's AngleClassifier currently
#                                 validates exactly 2 classes, so doc_ori needs
#                                 the Phase-6 4-class handling before it is wired.
#                                 Exported here so it is ready.
#
# These are PaddleX *_infer tarballs (native Paddle inference format), so this
# script REQUIRES paddlepaddle + paddle2onnx (see requirements.txt).
#
# Outputs:
#   output/ocr_cls.onnx          (from textline_ori — the wired one)
#   output/doc_ori.onnx          (from doc_ori — for Phase 6)
#   output/cls_preprocess.json   (committed; textline input geometry)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; RAW="$OUT/raw"; mkdir -p "$RAW"
BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"

if ! command -v paddle2onnx >/dev/null 2>&1; then
  echo "✗ paddle2onnx not found. pip install -r requirements.txt first." >&2
  exit 1
fi

convert () {  # $1=model name (tar stem)   $2=dest .onnx
  local name="$1" dest="$2" tar="$RAW/$1.tar" dir="$RAW/$1"
  echo "▶ $name"
  [ -f "$tar" ] || curl -fSL --retry 2 -o "$tar" "$BASE/$name.tar"
  [ -d "$dir" ] || tar -xf "$tar" -C "$RAW"
  local m; m="$(dirname "$(find "$dir" -name 'inference.pdmodel' -print -quit)")"
  [ -n "$m" ] || { echo "✗ inference.pdmodel not found in $name"; exit 1; }
  paddle2onnx --model_dir "$m" \
    --model_filename inference.pdmodel --params_filename inference.pdiparams \
    --save_file "$dest" --opset_version 11 --enable_onnx_checker True
  echo "  -> $dest"
}

convert "PP-LCNet_x1_0_textline_ori_infer" "$OUT/ocr_cls.onnx"
convert "PP-LCNet_x1_0_doc_ori_infer"      "$OUT/doc_ori.onnx"

echo "✓ done"
echo
echo "Copy the WIRED one into the runtime profile:"
echo "  cp output/ocr_cls.onnx        ../../rs-pdf-core/runtime-paddle-latest-mobile/ocr_cls/ocr_cls.onnx"
echo "  (input shape is static -> auto-detected by rs_pdf_core; a preprocess.json"
echo "   with {\"input_height\":H,\"input_width\":W} is only needed to override it.)"
echo "  (doc_ori.onnx is for Phase 6 — page-orientation, 4-class)"
