# Layout ONNX — PP-DocLayout_plus-L (versioned, Phase 6)

Versioned export folder for the rs_pdf_core **`paddle-latest-mobile`** profile.
**Does not replace** `../layout_onnx` — that stays **YOLOv10b-DocLayNet** for the
`legacy-stable` profile.

| | Value |
|---|---|
| Model | **PP-DocLayout_plus-L** (PaddleX) |
| Source | `PP-DocLayout_plus-L_infer` (Paddle inference → paddle2onnx) |
| Classes | ~20+ (paragraph_title, text, table, figure, formula, number, …) |

## ⚠ Needs the Phase-6 Rust adapter first

PP-DocLayout is a **different architecture** from YOLOv10b. The current
`rs_pdf_core` layout pipeline hardcodes a **640×640 input** and validates a
**`(1, 300, 6)` post-NMS** output — PP-DocLayout's input size, output tensor
shape, and class set differ. So this model can only be wired once the **Phase-6
adapter** (input-shape/normalization config + output-format adapter +
class-map) is implemented in `src/layout_detection/`.

This folder produces the artifacts so they're ready when the adapter lands.

## Build (run this) — needs `paddle2onnx`

```bash
cd Paddle-OCR/layout_onnx_ppdoclayout
pip install -r requirements.txt
bash download_and_convert.sh
```

Produces `output/layout.onnx` + `output/labels.txt`.

> If a download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in `MODEL_URL`.
