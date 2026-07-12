# OCR ONNX v6 — PP-OCRv6_small (det + rec)

Versioned export folder for the rs_pdf_core **`ppocrv6-small`** model
profile. **Does not replace** `ocr_onnx_v5/` (PP-OCRv5 mobile,
`paddle-latest-mobile` profile) or `ocr_onnx/` (PP-OCRv3 det,
`legacy-stable`).

| | Detection | Recognition |
|---|---|---|
| Model | **PP-OCRv6_small_det** | **PP-OCRv6_small_rec** |
| Arch | DB (PPLCNetV4 + RepLKFPN) | CTC (EncoderWithLightSVTR) |
| Source | PaddleX `PP-OCRv6_small_det_onnx_infer` | PaddleX `PP-OCRv6_small_rec_onnx_infer` |
| Format | **pre-converted ONNX** (no paddle2onnx) | **pre-converted ONNX** |
| Dict | n/a | `ppocrv6_dict.txt` (18,708 chars, unified 50-language) |

## Build (run this)

```bash
cd Paddle-OCR/ocr_onnx_v6
bash fetch_onnx.sh
```

Needs only `curl` + `tar` + network — the `*_onnx_infer` tarballs already
contain converted `.onnx`. Produces:

```
output/ocr_detect.onnx
output/ocr_rec.onnx
output/ppocr_keys_v6.txt
rec_preprocess.json          (committed — rec input geometry, height 48)
```

## Then copy into the runtime profile

```bash
P=../../rs-pdf-core/runtime-ppocrv6-small
cp output/ocr_detect.onnx        "$P/ocr_detect/ocr_detect.onnx"
cp output/ocr_detect_labels.txt  "$P/ocr_detect/labels.txt"
cp output/ocr_rec.onnx           "$P/ocr_rec/en/ocr_rec.onnx"
cp output/ppocr_keys_v6.txt      "$P/ocr_rec/en/ppocr_keys_v6.txt"
cp rec_preprocess.json           "$P/ocr_rec/en/preprocess.json"
```

## Det preprocessing compatibility (verified 2026-07-12)

The det tarball's `inference.yml` matches rs_pdf_core's hardcoded DB
preprocessing: ImageNet mean/std, 1/255 scale, /32 padding. Only the
nominal resize policy differs (v6 default short-side ≥ 736 vs core's
long-side ≤ 960) — verified equivalent on rendered PDF pages.
Recommended v6 DB postprocess thresholds if ever exposed per-profile:
`thresh 0.2, box_thresh 0.45, unclip_ratio 1.4`.

> If a download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in `fetch_onnx.sh`'s
> `BASE` URL (the PaddleX bucket has been published under both prefixes).
