#!/usr/bin/env bash
# Fetch a PRE-CONVERTED PP-DocLayout ONNX (official PaddlePaddle release on
# HuggingFace) for the rs_pdf_core "paddle-latest-mobile" layout profile.
#
# This avoids paddle2onnx entirely (the download_and_convert.sh route needs it,
# and paddle2onnx has no x86_64-macOS build). The model is a PaddleX RT-DETR
# detector: inputs [image(1,3,800,800), im_shape, scale_factor], output
# [N,8] = [class, score, x1, y1, x2, y2, _, _] in ORIGINAL image coords.
# rs_pdf_core's detect_paddlex handles this when layout_preprocess.json sets
# kind=paddlex_det, extra_inputs=true, coords=original (written below).
#
# Default model: PP-DocLayoutV2 (25 classes). For V3, set REPO=PP-DocLayoutV3_onnx.
#
# Outputs: output/layout.onnx, output/labels.txt, output/layout_preprocess.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/output"; mkdir -p "$OUT"
REPO="${REPO:-PP-DocLayoutV2_onnx}"
BASE="https://huggingface.co/PaddlePaddle/${REPO}/resolve/main"

echo "▶ downloading $REPO/inference.onnx (with size check)"
URL="$BASE/inference.onnx?download=true"
EXP=$(curl -sIL --max-time 30 "$URL" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v); print v}')
curl -sL -C - --max-time 900 -o "$OUT/layout.onnx" "$URL"
GOT=$(wc -c < "$OUT/layout.onnx" | tr -d ' ')
[ -n "$EXP" ] && [ "$GOT" != "$EXP" ] && { echo "✗ incomplete: $GOT/$EXP bytes — rerun to resume"; exit 1; }
echo "  -> $OUT/layout.onnx ($(du -h "$OUT/layout.onnx" | cut -f1))"

echo "▶ labels + layout_preprocess.json from inference.yml"
curl -sL --max-time 60 -o "$OUT/inference.yml" "$BASE/inference.yml"
python3 - "$OUT/inference.yml" "$OUT/labels.txt" "$OUT/layout_preprocess.json" <<'PY'
import re, json, sys
yml, labels_out, cfg_out = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(yml).read()
labs = re.search(r'label_list:\s*\n((?:\s*-\s*.+\n)+)', t)
labels = [l.strip()[2:].strip() for l in (labs.group(1).splitlines() if labs else [])]
open(labels_out, 'w').write("\n".join(labels) + "\n")
ts = re.search(r'target_size:\s*\n\s*-\s*(\d+)', t)
size = int(ts.group(1)) if ts else 800
def nums(key):
    m = re.search(key + r':\s*\n((?:\s*-\s*[\d.]+\n)+)', t)
    return [float(x) for x in re.findall(r'-\s*([\d.]+)', m.group(1))] if m else None
mean = nums('mean') or [0.0, 0.0, 0.0]
std = nums('std') or [1.0, 1.0, 1.0]
cfg = {"kind": "paddlex_det", "input_size": size, "mean": mean, "std": std,
       "extra_inputs": True, "coords": "original", "model_tag": "PP-DocLayout"}
json.dump(cfg, open(cfg_out, 'w'), indent=2)
print(f"  labels: {len(labels)} | {cfg}")
PY

echo "✓ done. Copy into the runtime profile:"
echo "  cp output/layout.onnx output/labels.txt output/layout_preprocess.json \\"
echo "     ../../rs-pdf-core/runtime-paddle-latest-mobile/layout/"
