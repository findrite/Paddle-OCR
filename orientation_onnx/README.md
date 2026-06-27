# Orientation ONNX — PP-LCNet (textline + doc)

Versioned export folder for the rs_pdf_core **`paddle-latest-mobile`** profile.
There is no existing orientation export folder, so nothing is replaced.

| Model | Classes | Role | Wired now? |
|---|---|---|---|
| **PP-LCNet_x1_0_textline_ori** | 2 (0° / 180°) | text-line angle classifier → `ocr_cls` | ✅ yes (drop-in for legacy PP-OCR v2.0 cls) |
| **PP-LCNet_x1_0_doc_ori** | 4 (0/90/180/270) | page orientation | ⏳ Phase 6 (needs 4-class handling in `AngleClassifier`) |

## Build (run this) — needs `paddle2onnx`

```bash
cd Paddle-OCR/orientation_onnx
pip install -r requirements.txt
bash download_and_convert.sh
```

Produces:

```
output/ocr_cls.onnx     # textline_ori (2-class) — the wired one
output/doc_ori.onnx     # doc_ori (4-class) — for Phase 6
```

## Then copy into the runtime profile

```bash
cp output/ocr_cls.onnx ../../rs-pdf-core/runtime-paddle-latest-mobile/ocr_cls/ocr_cls.onnx
```

> The PP-LCNet input shape is static, so `rs_pdf_core` auto-detects it — no
> `preprocess.json` is required. Only add `ocr_cls/preprocess.json` with
> `{"input_height": H, "input_width": W}` if you need to override it.

> If a download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in the `BASE` URL.
