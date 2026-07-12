# Layout ONNX v2 — PP-DocLayoutV2 (pre-converted)

Scripted, reproducible build for the `layout/layout.onnx` used by the
rs_pdf_core `paddle-latest-mobile` and `ppocrv6-small` profiles. The PaddleX
bucket publishes a pre-converted `PP-DocLayoutV2_onnx_infer` tarball, so this
needs only `curl` + `tar` — no paddle2onnx.

Verified 2026-07-12: the tarball's ONNX is **byte-identical**
(`shasum f2cfffde…`) to the model previously staged from a local one-off
conversion, so this script is now the canonical provenance.

## Build (run this)

```bash
cd Paddle-OCR/layout_onnx_v2
bash fetch_onnx.sh
```

Produces `output/layout.onnx` (~204 MB). Labels + preprocess config are
already committed in the rs-pdf-core runtime dirs.

## Then copy into the runtime profiles

```bash
cp output/layout.onnx ../../rs-pdf-core/runtime-ppocrv6-small/layout/layout.onnx
cp output/layout.onnx ../../rs-pdf-core/runtime-paddle-latest-mobile/layout/layout.onnx
```

> If the download 404s, swap `paddle3.0.0` ↔ `paddlex3.0.0` in `fetch_onnx.sh`'s
> `BASE` URL.
