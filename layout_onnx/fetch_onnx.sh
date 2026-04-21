#!/usr/bin/env bash
# Fetch a pre-converted document-layout ONNX model from HuggingFace.
#
# Default model: yolov10b-doclaynet (Oblix mirror of hantian/yolo-doclaynet)
# - Architecture: YOLOv10-b
# - Training data: DocLayNet
# - Classes (11): Caption, Footnote, Formula, List-item, Page-footer,
#                  Page-header, Picture, Section-header, Table, Text, Title
# - Input: (1, 3, 640, 640) RGB, normalised by 1/255 (no mean/std), letterboxed
# - Output: (1, 300, 6) post-NMS — [xmin, ymin, xmax, ymax, score, class_id]
#
# Outputs:
#   output/layout.onnx
#   output/labels.txt   (already committed; do not regenerate)
#
# Why this model: when this project was set up the official PaddleOCR
# PP-DocLayout / PP-StructureV3 models still required `paddle2onnx` + a working
# PaddlePaddle install to convert from `.pdmodel` to ONNX, which is heavy and
# fragile across architectures. The DocLayNet-trained YOLOv10 model is the
# strongest open document-layout detector that ships **directly as ONNX** on
# HuggingFace and covers exactly the targets the user asked for: headings,
# paragraphs, lists, tables, figures, footnotes, headers/footers.
#
# The (commented-out) `download_and_convert.sh` next to this script preserves
# the full PaddleOCR conversion path for the day a richer Paddle layout model
# is needed; it just requires `paddle2onnx` + `paddlepaddle` to be installed
# locally.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"
mkdir -p "$OUT"

URL="https://huggingface.co/Oblix/yolov10b-doclaynet_ONNX_document-layout-analysis/resolve/main/onnx/model.onnx?download=true"
DEST="$OUT/layout.onnx"

if [ -f "$DEST" ] && [ "$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")" -gt 70000000 ]; then
  echo "✓ $DEST already present ($(du -h "$DEST" | cut -f1))"
else
  echo "▶ downloading $URL"
  curl -fSL --retry 2 -o "$DEST" "$URL"
fi

echo "✓ done"
echo "  - $DEST"
echo "  - $OUT/labels.txt (committed, 11 DocLayNet classes)"
echo
echo "Next: copy these two files into rs-pdf-core/runtime/layout/"
