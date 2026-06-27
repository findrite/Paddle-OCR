# OCR ONNX v5 — PP-OCRv5 mobile (det + rec)

Versioned export folder for the rs_pdf_core **`paddle-latest-mobile`** model
profile. **Does not replace** the existing `ocr_onnx/` (PP-OCRv3 det) — that
stays the `legacy-stable` source.

| | Detection | Recognition |
|---|---|---|
| Model | **PP-OCRv5_mobile_det** | **PP-OCRv5_mobile_rec** |
| Arch | DB (Differentiable Binarization) | CTC |
| Source | PaddleX `PP-OCRv5_mobile_det_onnx_infer` | PaddleX `PP-OCRv5_mobile_rec_onnx_infer` |
| Format | **pre-converted ONNX** (no paddle2onnx) | **pre-converted ONNX** |
| Dict | n/a | `ppocrv5_en_dict.txt` (436 chars, EN) |

## Build (run this)

```bash
cd Paddle-OCR/ocr_onnx_v5
bash fetch_onnx.sh
```

Needs only `curl` + `tar` + network — the `*_onnx_infer` tarballs already
contain converted `.onnx`. Produces:

```
output/ocr_detect.onnx
output/ocr_rec.onnx
output/ppocr_keys_v5_en.txt
rec_preprocess.json          (committed — rec input geometry, height 48)
```

## Then copy into the runtime profile

```bash
P=../../rs-pdf-core/runtime-paddle-latest-mobile
cp output/ocr_detect.onnx        "$P/ocr_detect/ocr_detect.onnx"
cp output/ocr_rec.onnx           "$P/ocr_rec/en/ocr_rec.onnx"
cp output/ppocr_keys_v5_en.txt   "$P/ocr_rec/en/ppocr_keys_v5_en.txt"
cp rec_preprocess.json           "$P/ocr_rec/en/preprocess.json"
```

> If a download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in `fetch_onnx.sh`'s
> `BASE` URL (the PaddleX bucket has been published under both prefixes).
