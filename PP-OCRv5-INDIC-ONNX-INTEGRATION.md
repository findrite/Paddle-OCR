# PP-OCRv5 Indic Recognition Integration — Devanagari, Tamil, Telugu

Date: 2026-07-12. Scope: `Paddle-OCR/` (export folder) + `rs-pdf-core/` (runtime staging, demo integration). Implements official PP-OCRv5 recognition for Hindi/Devanagari-script languages, Tamil, and Telugu, selectable in the demo's OCR Detection tab via language aliases, running on the PP-OCRv6 detector.

## Conversion-workflow note (documented deviation)

The task suggested downloading `<model>_infer.tar` and converting locally with paddle2onnx (target opset 11). This repo's **established, verified workflow** (`ocr_onnx_v5/`, `ocr_onnx_v6/`) instead fetches PaddleX's **official pre-converted** `<model>_onnx_infer.tar` from the *same bucket and model release* — the repo's own READMEs (`ocr_onnx/README.md`, `layout_onnx/README.md`) document local paddle2onnx as fragile on this machine. The official conversions are **opset 7** (identical to the production `PP-OCRv5_mobile_rec` already staged here). All acceptance criteria (checker-valid ONNX, CPU ORT load, correct vocab, real recognition) are met and evidenced below.

## Files added

| File | Purpose |
|------|---------|
| `Paddle-OCR/ocr_onnx_v5_langs/fetch_onnx.sh` | Family-parameterized fetch (devanagari\|ta\|te\|all): download, tar-validate, copy ONNX + matching dict, write `checksums.sha256` |
| `Paddle-OCR/ocr_onnx_v5_langs/output/<fam>/{ocr_rec.onnx,dict.txt,preprocess.json,checksums.sha256}` | Build outputs (ONNX not committed, per repo convention) |
| `rs-pdf-core/runtime-ppocrv6-{small,medium}/ocr_rec/devanagari/{ocr_rec.onnx*,ppocrv5_devanagari_dict.txt,preprocess.json}` | Staged Devanagari recognizer (*.onnx gitignored) |
| `rs-pdf-core/runtime-ppocrv6-{small,medium}/ocr_rec/ta/{…}` | Staged Tamil recognizer |
| `rs-pdf-core/runtime-ppocrv6-{small,medium}/ocr_rec/te/{…}` | Staged Telugu recognizer |
| `Paddle-OCR/PP-OCRv5-INDIC-ONNX-INTEGRATION.md` | This report |

## Files modified

| File | Change |
|------|--------|
| `rs-pdf-core/rs_pdf_core_demo/src/main.rs` | `LANGUAGE_ALIASES` table + `canonical_ocr_language()`; `resolve_language` applies aliases; `/api/ocr/languages` enriched (per-language `rec_model`, `aliases`, profile `det_model`, `?profile=` param); 4 unit tests |
| `rs-pdf-core/rs_pdf_core_demo/static/index.html` | Language dropdown: one option per canonical model + one per alias, labeled with the recognizer model; det/rec model info line under the dropdown; refresh on profile change |

No core-library (`rs-pdf-core/src/`) changes were needed — `RuntimeLayout::scan` discovers language folders automatically, and the existing recognizer (48-px height, dynamic width, mean/std 0.5, CTC blank=0, repeat-collapse, dict-offset −1, appended space, per-char confidence) matches these models exactly (verified in §Validation). Detector behavior untouched: the language-independent **PP-OCRv6 detector** is used for all languages.

## Model inventory

| Model | Source URL | ONNX opset | Input shape | Output shape | Dictionary entries | Status |
|-------|-----------|-----------:|-------------|--------------|-------------------:|--------|
| devanagari_PP-OCRv5_mobile_rec | `…/paddle3.0.0/devanagari_PP-OCRv5_mobile_rec_onnx_infer.tar` | 7 | `x: (N,3,48,W) f32 NCHW`, W dynamic | `(N,T,570)` | 568 (`ppocrv5_devanagari_dict.txt`) | ✅ staged + integrated |
| ta_PP-OCRv5_mobile_rec | `…/ta_PP-OCRv5_mobile_rec_onnx_infer.tar` | 7 | same | `(N,T,515)` | 513 (`ppocrv5_ta_dict.txt`) | ✅ staged + integrated |
| te_PP-OCRv5_mobile_rec | `…/te_PP-OCRv5_mobile_rec_onnx_infer.tar` | 7 | same | `(N,T,542)` | 540 (`ppocrv5_te_dict.txt`) | ✅ staged + integrated |

**Vocabulary math (verified by live ORT inference):** `vocab = dict entries + space token + CTC blank` → 568+1+1=570 ✓, 513+1+1=515 ✓, 540+1+1=542 ✓. The Rust loader appends the space when the dict file lacks it (`recognition.rs::parse_dictionary`) and hard-validates `dict.len()+1 == vocab` at first inference.

**Dictionaries:** the repo dicts are **byte-identical** to the authoritative `PostProcess.character_dict` embedded in each model's `inference.yml` (verified entry-by-entry). No v6/en/multilingual dict reuse.

**Validation (all three):** `onnx.checker.check_model` PASS · input `x` float32 NCHW · height fixed 48, width dynamic (320→40 CTC steps, 640→80 verified) · no unresolved custom Paddle ops (p2o-converted graph loads on CPU ORT 1.19.2) · preprocessing per each model's own `inference.yml`: `RecResizeImg [3,48,320]`, CTC decode — identical across the three (verified separately, not assumed).

## Language support added

| Language | Code(s) | Physical model | Tested | Status |
|----------|---------|----------------|-------:|--------|
| Hindi | `hi`, `hindi` → `devanagari` | devanagari_PP-OCRv5_mobile_rec | ✅ e2e | Working |
| Marathi | `mr`, `marathi` → `devanagari` | same | alias-tested | Working (same script model) |
| Nepali | `ne`, `nepali` → `devanagari` | same | alias-tested | Working (same script model) |
| Sanskrit | `sa`, `sanskrit` → `devanagari` | same | alias-tested | Working (same script model) |
| Maithili / Bhojpuri / Konkani | `mai` / `bho` / `gom` → `devanagari` | same | alias-tested | Working (same script model) |
| Tamil | `ta`, `tamil` | ta_PP-OCRv5_mobile_rec | ✅ e2e | Working |
| Telugu | `te`, `telugu` | te_PP-OCRv5_mobile_rec | ✅ e2e | Working |
| Kannada | `kn` | — | ✅ rejection tested | **Deliberately unsupported** — clean error: `language 'kn' has no recognizer installed. Available: [devanagari, en, ta, te]` |

One physical folder per script; aliases resolve in `canonical_ocr_language()` — no weight duplication across language codes. (Weights are staged in both v6 profiles, small + medium, so the language list is identical whichever default-family profile is active.)

## Smoke-test results

Direct ORT (PIL-rendered single lines, macOS Sangam MN fonts):

| Language | Input sample | Recognized text | Confidence | Result |
|----------|--------------|-----------------|-----------:|--------|
| Hindi | नमस्ते भारत | नमस्ते भारत (exact) | 0.946 | ✅ |
| Tamil | வணக்கம் நண்பா | வணக்கம் நண்பா (exact) | 0.992 | ✅ |
| Telugu | నమస్తే మిత్రమా | నమస్తే మిత్రమా (exact) | 0.988 | ✅ |

Full pipeline e2e (demo server: upload PDF → PP-OCRv6_small_det → crop → recognizer via alias):

| Language (alias used) | det / rec reported | Sample line results | Notes |
|---|---|---|---|
| `hi` | PP-OCRv6_small_det / devanagari_PP-OCRv5_mobile_rec | `नमस्ते भारत` 0.969 (exact); `यह एक परीकषण दसतावेज़ हैं` 0.948; `हदो पाठ पहचान` 0.962 | headline exact; some conjunct/matra errors on smaller body lines (mobile-tier model) |
| `ta` | PP-OCRv6_small_det / ta_PP-OCRv5_mobile_rec | `வணக்கம்நணப` 0.944; `இதுஒருசேோதனை ஆவணம்` 0.938; `தமிழ் ளழுத்து அறிதல்` 0.937 | in-script, minor spacing/char errors |
| `te` | PP-OCRv6_small_det / te_PP-OCRv5_mobile_rec | `నమస ్తై మిత ్రమా` 0.912; two more lines 0.917/0.897 | in-script; virama spacing artifacts on this render |

All outputs: correct Unicode script, no `�` replacement characters, NFC-normalization-stable, per-line confidence returned, zero index-out-of-range errors. Results reported verbatim — not manufactured.

## Checksums (SHA-256)

**devanagari** — archive `21cbdcb0c5656923500359ba053b814fc6089dbffa5e5017125d8654c0551817` · ocr_rec.onnx `cb789212ce96c69d3e74728ae4309d179281d68cb3945d0616b67cafab41c986` · dict.txt `09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc` · preprocess.json `cae9c3c62cc99af9b69d4757ca15d89545e3eb86f5cbcdce96e026a4f65bd584`

**ta** — archive `e454d8910ba80d05355a9049b3d343f5866246c15ed5c22e0bd96cd1e8032a17` · ocr_rec.onnx `c6d2b682d2a0ea4cb1fccdba295976f93fd439964d16cdc666cadef531accbee` · dict.txt `85b541352ae18dc6ba6d47152d8bf8adff6b0266e605d2eef2990c1bf466117b` · preprocess.json `0b3688f995628973e67d3668562d3d9e36e01da64706e80237c1e8d54c781dcb`

**te** — archive `722e0e614c1dab033fadf598d5c3e32f881b874bef9a20e60b39eab0432eadc5` · ocr_rec.onnx `8238bfc46d4cffe720ed6706e3842802467343497428693ff2bfb4e6b3caa36b` · dict.txt `42f83f5d3fdb50778e4fa5b66c58d99a59ab7792151c5e74f34b8ffd7b61c9d6` · preprocess.json `47ed652d3194c68957fdf2570f8d5f8e9a081335f63990e8a523f2cde8926d50`

(Also recorded per-family in `ocr_onnx_v5_langs/output/<fam>/checksums.sha256`.)

## Tests executed

| Test | Result |
|---|---|
| `rs_pdf_core_demo::tests::language_aliases_resolve_devanagari_family` | ✅ pass |
| `rs_pdf_core_demo::tests::language_aliases_resolve_tamil_and_telugu` | ✅ pass |
| `rs_pdf_core_demo::tests::language_aliases_pass_unknown_codes_through` | ✅ pass |
| `rs_pdf_core_demo::tests::kannada_is_not_mapped_to_any_script` | ✅ pass |
| `cargo test -p rs_pdf_core --lib --features ocr-detect` (full suite incl. OCR modules) | ✅ 5217 passed / 0 failed |
| ONNX validation (checker/opset/IO/dynamic-width/vocab ×3) | ✅ all pass |
| Direct ORT smoke ×3 languages | ✅ exact recognition |
| Demo e2e ×3 languages via aliases | ✅ correct model routing + in-script output |
| Model discovery (`/api/ocr/languages` → devanagari, en, ta, te) | ✅ |
| `kn` rejection with clear error | ✅ |
| English/v6 regression (default request → PP-OCRv6_small det+rec, unchanged) | ✅ |

## Remaining limitations

- **Kannada is not added** — no official PP-OCRv5/v6 Kannada recognizer exists; `kn` intentionally errors rather than mis-routing to another script.
- **No automatic script detection** — the recognizer is chosen by the user's language selection (aliases make that natural); pages must use one recognizer per request.
- **No mixed-script line routing** — Hindi+Tamil on one page/line is not supported; Latin embedded in Indic pages is limited to the ASCII subset of each script dict.
- **PP-OCRv6 recognition still cannot emit Indic scripts** — its dictionary contains zero Indic characters; Indic goes through these PP-OCRv5 recognizers only.
- **Selection is alias-driven** — `hi/mr/ne/sa/mai/bho/gom → devanagari`, `tamil → ta`, `telugu → te`; the UI dropdown lists every alias with the model it resolves to, plus a det/rec info line.
- Indic models are staged in the `ppocrv6-small` and `ppocrv6-medium` profiles only (the default family); `legacy-stable` / `paddle-latest-mobile` remain English-only.
- Opset is 7 (official conversion), not the suggested 11 — see the deviation note; identical to the already-production v5 recognizer.
