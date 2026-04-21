# Document Layout ONNX (PaddleOCR-aligned project folder)

Self-contained workspace that produces a single ONNX file used by
`rs_pdf_core`'s `layout_detection` module.

## Model shipped by default

| Field | Value |
|---|---|
| Architecture | **YOLOv10-b** (NMS-free, single-stage) |
| Training set | **DocLayNet** (IBM, 80k+ richly labelled doc pages) |
| Source repo | [hantian/yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet) |
| ONNX mirror | [Oblix/yolov10b-doclaynet_ONNX_document-layout-analysis](https://huggingface.co/Oblix/yolov10b-doclaynet_ONNX_document-layout-analysis) |
| File size | ~76 MB (fp32) |
| Input | `(1, 3, 640, 640)` float32, RGB, **rescaled by 1/255 only** (no mean/std), letterbox to 640×640 |
| Output | `(1, 300, 6)` float32 — `[xmin, ymin, xmax, ymax, score, class_id]`, **already post-NMS** |
| Classes (11) | `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title` |
| Licence | Apache-2.0 (model + ONNX export) |

### Why DocLayout-YOLO over PaddleOCR's PP-DocLayout

The user's intended targets were *headings, paragraphs, lists, tables, TOC,
images, references, footnotes*. PP-DocLayout-L would have covered TOC and
reference too, but its weights ship only as a Paddle inference model
(`.pdmodel` + `.pdiparams`) that requires `paddle2onnx` **and** a working
`paddlepaddle` installation to convert. On non-trivial macOS / Linux setups
this is fragile (architecture mismatches, multi-GB downloads, OpenSSL
issues).

The DocLayNet-trained YOLOv10-b model is the strongest open document-layout
detector that ships **directly as ONNX** and matches almost every requested
class one-to-one:

| User target | DocLayNet class |
|---|---|
| Headings | `Title`, `Section-header` |
| Paragraphs | `Text` |
| Lists | `List-item` |
| Tables | `Table` (+ `Caption`) |
| Images | `Picture` |
| Footnotes | `Footnote` |
| Headers / footers | `Page-header`, `Page-footer` |
| Formulas | `Formula` |
| Figure captions | `Caption` |
| TOC | *(no explicit class — usually detected as `Text` / `List-item`)* |
| References | *(no explicit class — usually detected as `Text`)* |
| Links | **Not a layout class.** Links are PDF annotations and are extracted separately by `rs_pdf_core::annotations`. |

If a future ONNX export of PP-DocLayout-L lands on HuggingFace (or
`paddle2onnx` becomes installable on this machine), drop it in
`output/layout.onnx`, replace `output/labels.txt`, and the Rust runtime will
pick up the new class set automatically (with code adjustments only if
input/output shapes change).

## Folder layout

```
layout_onnx/
├── fetch_onnx.sh            # one-shot HF download (default path)
├── download_and_convert.sh  # legacy: PaddleOCR -> ONNX via paddle2onnx
├── extract_labels.py        # helper for download_and_convert.sh
├── smoke_test.py            # standalone Python smoke test (onnxruntime)
├── requirements.txt         # python deps for Paddle conversion path
├── output/
│   ├── layout.onnx          # the model (committed via Git LFS or as binary blob)
│   └── labels.txt           # one class per line, index = class id
└── README.md
```

## Quick start

### A) Use the bundled model (default)

The repo ships with `output/layout.onnx` already populated. Nothing to do —
the Rust runtime in `rs-pdf-core/runtime/layout/` is a copy of these files.

### B) Re-fetch from HuggingFace

```bash
cd Paddle-OCR/layout_onnx
bash fetch_onnx.sh
cp output/layout.onnx output/labels.txt ../../rs-pdf-core/runtime/layout/
```

### C) Convert from PaddleOCR PP-DocLayout (heavy path)

```bash
cd Paddle-OCR/layout_onnx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_and_convert.sh
cp output/layout.onnx output/labels.txt ../../rs-pdf-core/runtime/layout/
```

## Verifying the model

```bash
python smoke_test.py path/to/page.png --score 0.4 --iou 0.5
```

The smoke test runs the same preprocessing the Rust code uses (letterbox →
1/255 → NCHW), runs the ONNX session, post-processes the `(1, 300, 6)` output
with a score threshold, scales boxes back to the source image, and prints:

```json
[
  {"label": "Title", "score": 0.97, "box": [42, 91, 530, 62]},
  {"label": "Text",  "score": 0.96, "box": [42, 180, 530, 220]}
]
```

If the smoke test produces sensible boxes, the Rust port in
`rs-pdf-core/src/layout_detection/` will produce the same shapes.

## License

- Model weights (`output/layout.onnx`): Apache-2.0 (DocLayNet weights, YOLOv10
  upstream).
- Conversion / fetch scripts in this folder: same dual MIT / Apache-2.0 as
  the parent `rs_pdf_core` repo.
