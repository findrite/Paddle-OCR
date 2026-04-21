# OCR Detection ONNX (PaddleOCR-aligned project folder)

Self-contained workspace that produces a single ONNX file used by
`rs_pdf_core`'s `ocr_detection` module.

## Model shipped by default

| Field | Value |
|---|---|
| Architecture | **DB (Differentiable Binarization)** |
| Training set | **PaddleOCR English text detection** |
| Source | [PaddleOCR PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR) |
| File size | ~2-4 MB (fp32) |
| Input | `(1, 3, H, W)` float32, RGB, normalised (ImageNet mean/std), max side 960px, padded to multiple of 32 |
| Output | `(1, 1, H, W)` float32 — sigmoid probability map |
| Classes (1) | `text` |
| Licence | Apache-2.0 |

### How it differs from Layout Detection

| | Layout Detection | OCR Detection |
|---|---|---|
| Model | YOLOv10-b (DocLayNet) | DB (PP-OCRv3) |
| Purpose | Detect document **structure** (headings, paragraphs, tables, figures, ...) | Detect **text regions** (individual text lines/words) |
| Output | Direct bounding boxes `(1, 300, 6)` | Probability map `(1, 1, H, W)` requiring post-processing |
| Classes | 11 (Caption, Footnote, Formula, ...) | 1 (text) |
| Post-processing | Score threshold + NMS (built into model) | Threshold → binarize → connected components → bounding boxes |

## Folder layout

```
ocr_onnx/
├── fetch_onnx.sh            # download + convert from PaddleOCR
├── download_and_convert.sh  # full conversion path (paddle2onnx)
├── smoke_test.py            # standalone Python smoke test (onnxruntime)
├── requirements.txt         # python deps
├── output/
│   ├── ocr_detect.onnx      # the model (produced by fetch_onnx.sh)
│   └── labels.txt           # one class: text
└── README.md
```

## Quick start

### A) Convert from PaddleOCR (requires paddle2onnx)

```bash
cd Paddle-OCR/ocr_onnx
pip install paddle2onnx
bash fetch_onnx.sh
cp output/ocr_detect.onnx output/labels.txt ../../rs-pdf-core/runtime/ocr_detect/
```

### B) Full conversion path (requires PaddlePaddle + paddle2onnx)

```bash
cd Paddle-OCR/ocr_onnx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_and_convert.sh
cp output/ocr_detect.onnx output/labels.txt ../../rs-pdf-core/runtime/ocr_detect/
```

## Verifying the model

```bash
python smoke_test.py path/to/page.png --score 0.3
```

The smoke test runs preprocessing (resize, normalize), ONNX inference,
DB post-processing (threshold → contour → bbox), and prints:

```json
[
  {"label": "text", "score": 0.87, "box": [42, 91, 530, 22]},
  {"label": "text", "score": 0.93, "box": [42, 180, 530, 18]}
]
```

## License

- Model weights: Apache-2.0 (PaddleOCR).
- Scripts in this folder: same dual MIT / Apache-2.0 as the parent `rs_pdf_core` repo.
