#!/usr/bin/env bash
# Fetch the PRE-CONVERTED PP-DocLayoutV2 ONNX layout model.
# This gives the runtime-paddle-latest-mobile / runtime-ppocrv6-small
# layout/layout.onnx a fully scripted, reproducible build (curl + tar only —
# no paddle2onnx). Previously this model existed only as a local one-off
# conversion.
#
# Model: PP-DocLayoutV2 — document layout detection (~20 classes:
#        doc_title, paragraph_title, text, table, image, formula, ...)
#
# Outputs:
#   output/layout.onnx
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; RAW="$OUT/raw"; mkdir -p "$RAW"

# If a URL 404s, try swapping paddle3.0.0 <-> paddlex3.0.0 in BASE.
BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
NAME="PP-DocLayoutV2_onnx_infer"

tar="$RAW/$NAME.tar"; dir="$RAW/$NAME"
if [ -f "$OUT/layout.onnx" ] && [ "$(stat -f%z "$OUT/layout.onnx" 2>/dev/null || stat -c%s "$OUT/layout.onnx")" -gt 1000000 ]; then
  echo "✓ $OUT/layout.onnx already present"
else
  echo "▶ downloading $NAME"
  [ -f "$tar" ] || curl -fSL --retry 2 -o "$tar" "$BASE/$NAME.tar"
  [ -d "$dir" ] || tar -xf "$tar" -C "$RAW"
  onnx="$(find "$dir" -name '*.onnx' -print -quit)"
  [ -n "$onnx" ] || { echo "✗ no .onnx inside $NAME.tar"; exit 1; }
  cp "$onnx" "$OUT/layout.onnx"
  echo "  -> $OUT/layout.onnx"
fi

echo "✓ done."
echo
echo "Compare against the staged model before replacing (should be the same"
echo "architecture; labels/preprocess config already committed in the runtime dirs):"
echo "  shasum output/layout.onnx ../../rs-pdf-core/runtime-ppocrv6-small/layout/layout.onnx"
