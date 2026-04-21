#!/usr/bin/env python3
"""Extract `label_list` from a PaddleOCR/PaddleX inference.yml and write
one label per line to the given output path.

Usage:
    python extract_labels.py <path/to/inference.yml> <path/to/labels.txt>

If `label_list` is missing, falls back to the canonical PP-DocLayout_plus-L
class list.
"""

from __future__ import annotations

import sys
from pathlib import Path

FALLBACK = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    inference_yml = Path(sys.argv[1]) if sys.argv[1] else None
    out = Path(sys.argv[2])
    labels: list[str] = []

    if inference_yml and inference_yml.exists():
        try:
            import yaml  # type: ignore
        except ImportError:
            print("PyYAML not installed; using fallback label list", file=sys.stderr)
            yaml = None  # type: ignore
        if yaml is not None:
            data = yaml.safe_load(inference_yml.read_text())
            if isinstance(data, dict):
                # PaddleX inference.yml typically has either:
                #   PostProcess.label_list
                #   or arch.label_list
                # We do a wide search.
                for key in ("label_list", "labels"):
                    if key in data and isinstance(data[key], list):
                        labels = [str(x) for x in data[key]]
                        break
                if not labels:
                    pp = data.get("PostProcess", {})
                    if isinstance(pp, dict) and isinstance(pp.get("label_list"), list):
                        labels = [str(x) for x in pp["label_list"]]
                if not labels:
                    arch = data.get("arch", {})
                    if isinstance(arch, dict) and isinstance(arch.get("label_list"), list):
                        labels = [str(x) for x in arch["label_list"]]

    if not labels:
        print("⚠ no label_list found in inference.yml; using built-in fallback", file=sys.stderr)
        labels = FALLBACK

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"wrote {len(labels)} labels → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
