#!/usr/bin/env python3
"""Write rs_pdf_core's layout_preprocess.json from a PaddleX inference.yml.

Usage:
    python extract_preprocess.py <path/to/inference.yml> <out/layout_preprocess.json>

Reads the model's own preprocessing config (Resize target size + NormalizeImage
mean/std) so the values are authoritative rather than guessed. Falls back to
PP-DocLayout-ish defaults (800x800, ImageNet mean/std) if the yml can't be parsed.

The emitted file selects the PaddleX adapter in rs_pdf_core:
    { "kind": "paddlex_det", "input_size": N, "mean": [...], "std": [...] }
"""
import json
import sys

DEFAULTS = {
    "kind": "paddlex_det",
    "input_size": 800,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "model_tag": "PP-DocLayout_plus-L",
}


def find_in(obj, key):
    """Depth-first search for the first dict containing `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_in(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_in(v, key)
            if r is not None:
                return r
    return None


def main():
    yml_path = sys.argv[1] if len(sys.argv) > 1 else ""
    out_path = sys.argv[2] if len(sys.argv) > 2 else "layout_preprocess.json"
    cfg = dict(DEFAULTS)
    try:
        import yaml  # PyYAML
        data = yaml.safe_load(open(yml_path).read())
        # Resize target size (PaddleX uses [h, w] or a scalar).
        size = find_in(data, "target_size") or find_in(data, "image_shape")
        if isinstance(size, (list, tuple)) and size:
            cfg["input_size"] = int(size[-1])
        elif isinstance(size, int):
            cfg["input_size"] = int(size)
        mean = find_in(data, "mean")
        std = find_in(data, "std")
        if isinstance(mean, (list, tuple)) and len(mean) == 3:
            cfg["mean"] = [float(x) for x in mean]
        if isinstance(std, (list, tuple)) and len(std) == 3:
            cfg["std"] = [float(x) for x in std]
    except Exception as e:  # noqa: BLE001
        print(f"  (using defaults — could not parse {yml_path}: {e})", file=sys.stderr)

    json.dump(cfg, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}: {cfg}")


if __name__ == "__main__":
    main()
