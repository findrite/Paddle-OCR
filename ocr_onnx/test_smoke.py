"""Python integration smoke test — mirror of the Rust
`ocr_pipeline_integration::smoke_end_to_end_extracts_hello_world_from_sample_en`.

Runs the full detect → classify → recognize pipeline on
`rs-pdf-core/tests/fixtures/sample_en.png` and asserts that at least one
region carries the substring `"hello"` (case-insensitive) with blended
confidence ≥ 0.7. Gated by `RS_PDF_CORE_RUN_OCR_TESTS=1` so CI without
models installed stays green but doesn't silently pass.

Usage (from this directory):

    RS_PDF_CORE_RUN_OCR_TESTS=1 python3 -m pytest test_smoke.py -v
    # or, without pytest:
    RS_PDF_CORE_RUN_OCR_TESTS=1 python3 test_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
RUNTIME = HERE.parent.parent / "rs-pdf-core" / "runtime"
SAMPLE = HERE.parent.parent / "rs-pdf-core" / "tests" / "fixtures" / "sample_en.png"


def _require_env() -> None:
    v = os.environ.get("RS_PDF_CORE_RUN_OCR_TESTS", "")
    if v not in ("1", "true", "True"):
        raise RuntimeError(
            "RS_PDF_CORE_RUN_OCR_TESTS must be set to 1 to run the Python "
            "smoke test. This gate mirrors the Rust integration-test gate "
            "so the two pipelines are verified against the same ground truth."
        )


def run_smoke(language: str = "en", level: str = "line") -> dict:
    _require_env()
    assert SAMPLE.exists(), f"sample image missing: {SAMPLE}"
    cmd = [
        sys.executable,
        str(HERE / "smoke_test.py"),
        str(SAMPLE),
        "--language", language,
        "--level", level,
        "--runtime-dir", str(RUNTIME),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_smoke_extracts_hello_from_sample_en() -> None:
    payload = run_smoke()
    assert payload["language"] == "en"
    assert payload["detection_model"] == "pp-ocrv4-det"
    assert payload["recognition_model"] == "pp-ocrv4-rec"
    boxes = payload.get("boxes", [])
    assert boxes, "no regions detected"
    hits = [
        b for b in boxes
        if "hello" in b.get("ocr_text", "").lower()
        and float(b.get("confidence", 0.0)) >= 0.7
    ]
    assert hits, (
        "no box produced the 'Hello' substring with confidence ≥ 0.7. "
        f"Got: {[(b['ocr_text'], b['confidence']) for b in boxes]}"
    )


if __name__ == "__main__":
    test_smoke_extracts_hello_from_sample_en()
    print("PASS: Python smoke test found 'Hello' in sample_en.png output.")
