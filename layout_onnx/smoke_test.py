#!/usr/bin/env python3
"""Smoke-test the bundled DocLayout-YOLOv10 ONNX file.

Usage:
    python smoke_test.py path/to/page.png [--score 0.4]

Loads `output/layout.onnx` and `output/labels.txt`, runs inference,
applies a score threshold, scales boxes back to the original image,
and prints:

    [{"label": "...", "score": 0.91, "box": [left, top, width, height]}]

The output shape matches the JSON the Rust pipeline emits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
DEFAULT_ONNX = HERE / "output" / "layout.onnx"
DEFAULT_LABELS = HERE / "output" / "labels.txt"

INPUT_SIZE = 640    # square model input
PAD_COLOR = (114, 114, 114)


def load_labels(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """Letterbox resize to (size, size), keeping aspect ratio. Returns
    (canvas, scale, pad_x, pad_y)."""
    w0, h0 = img.size
    scale = min(size / w0, size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess(img: Image.Image):
    canvas, scale, pad_x, pad_y = letterbox(img)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0       # rescale only
    arr = arr.transpose(2, 0, 1)[None, ...]                  # NCHW, batch=1
    return arr, scale, pad_x, pad_y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--score", type=float, default=0.4)
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"✗ {args.onnx} not found — run fetch_onnx.sh first", file=sys.stderr)
        return 2
    if not args.image.exists():
        print(f"✗ {args.image} not found", file=sys.stderr)
        return 2

    labels = load_labels(args.labels)
    img = Image.open(args.image).convert("RGB")
    w0, h0 = img.size
    tensor, scale, pad_x, pad_y = preprocess(img)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: tensor})[0]   # (1, 300, 6)
    raw = raw[0]

    inv_scale = 1.0 / scale
    boxes_out: list[dict] = []
    for row in raw:
        x1, y1, x2, y2, score, cls = row
        if score < args.score:
            continue
        # Undo letterbox padding then scale back to original image
        x1 = (x1 - pad_x) * inv_scale
        y1 = (y1 - pad_y) * inv_scale
        x2 = (x2 - pad_x) * inv_scale
        y2 = (y2 - pad_y) * inv_scale
        l = max(0.0, x1)
        t = max(0.0, y1)
        r = min(float(w0), x2)
        b = min(float(h0), y2)
        if r <= l or b <= t:
            continue
        cls_i = int(cls)
        label = labels[cls_i] if 0 <= cls_i < len(labels) else f"class_{cls_i}"
        boxes_out.append({
            "label": label,
            "score": float(score),
            "box": [int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t))],
            "class_id": cls_i,
        })

    boxes_out.sort(key=lambda b: -b["score"])
    print(json.dumps(boxes_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
