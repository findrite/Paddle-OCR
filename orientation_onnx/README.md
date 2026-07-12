# Orientation ONNX — PP-LCNet (textline + doc)

Versioned export folder for the rs_pdf_core **`ppocrv6-small`** (default) and
**`paddle-latest-mobile`** profiles.

| Model | Classes | Role | Wired now? |
|---|---|---|---|
| **PP-LCNet_x1_0_textline_ori** | 2 (0° / 180°) | text-line angle classifier → `ocr_cls` | ✅ yes (staged in both profiles) |
| **PP-LCNet_x1_0_doc_ori** | 4 (0/90/180/270) | page orientation | ⏳ Phase 6 (needs 4-class handling in `AngleClassifier`) |

## Build (run this) — curl + tar only

```bash
cd Paddle-OCR/orientation_onnx
bash fetch_onnx.sh
```

The PaddleX bucket publishes pre-converted `*_onnx_infer` tarballs, so **no
paddle2onnx / paddlepaddle is needed**. (`download_and_convert.sh` remains as
the from-source conversion fallback.) Produces:

```
output/ocr_cls.onnx     # textline_ori (2-class) — the wired one
output/doc_ori.onnx     # doc_ori (4-class) — for Phase 6
```

## Then copy into the runtime profiles

```bash
for P in runtime-ppocrv6-small runtime-paddle-latest-mobile; do
  cp output/ocr_cls.onnx "../../rs-pdf-core/$P/ocr_cls/ocr_cls.onnx"
done
```

## ⚠ Normalization — preprocess.json is REQUIRED

Per the tarball's `inference.yml`, PP-LCNet_x1_0_textline_ori expects
**ImageNet normalization** (`mean [0.485,0.456,0.406]`, `std
[0.229,0.224,0.225]`), NOT the legacy PP-OCR cls `0.5/0.5` convention the
core defaults to. Each staged `ocr_cls/` therefore carries a committed
`preprocess.json`:

```json
{
  "input_height": 80,
  "input_width": 160,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "model_tag": "PP-LCNet_x1_0_textline_ori"
}
```

(The input shape is static in the ONNX so the core auto-detects 80×160
anyway; mean/std are the load-bearing fields.)

> If a download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in the `BASE` URL.
