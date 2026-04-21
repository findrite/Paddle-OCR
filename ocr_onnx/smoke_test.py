#!/usr/bin/env python3
"""Smoke-test the bundled PP-OCRv4 detection/classification/recognition ONNX files.

Usage:
    python smoke_test.py path/to/page.png [options]

Options:
    --score 0.3                 DB score threshold.
    --level word|line|block     Detection granularity (default: line).
    --language en|ch|...        Recognizer dir under ../ocr_rec/ (default: en).
    --skip-recognition          Skip recognition (ocr_text stays empty).
    --skip-angle-cls            Skip angle classification (no rotation fix).

The output matches the JSON the Rust pipeline emits:

    [{"region_id": "r1",
      "label": "text",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2],
      "img_coord": [left, top, width, height],
      "level": "line",
      "ocr_text": "Hello World"}]

This script loads the same three ONNX files that the Rust side does and
reproduces every preprocessing and post-processing step so both
pipelines emit byte-identical JSON for the same input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
DEFAULT_ONNX = HERE / "output" / "ocr_detect.onnx"
DEFAULT_LABELS = HERE / "output" / "labels.txt"
# The Rust runtime layout mirrors Paddle's: detector + classifier + per-language
# recognizers. We look for them next to the detector by default, but the CLI
# can override with --runtime-dir if you keep them elsewhere.
DEFAULT_RUNTIME_DIR = HERE.parent.parent / "rs-pdf-core" / "runtime"

MAX_SIDE = 960
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Classifier / recognizer use ([-1, 1]) normalisation — (x/255 - 0.5) / 0.5.
REC_INPUT_HEIGHT = 48
REC_MAX_WIDTH = 320
# Classifier shape defaults to PP-OCR mobile v2.0 (48, 192) but is auto-detected
# from the ONNX signature at load time — PP-OCRv5 textline-orientation ships
# with (80, 160) and we honour whichever the bundled model declares.
CLS_INPUT_HEIGHT = 48
CLS_INPUT_WIDTH = 192
# The angle classifier is conservative: only flip when P(180°) > 0.9. The
# threshold matches PaddleOCR's `cls_thresh` default; going lower causes
# false positives that corrupt recognition for that line.
CLS_FLIP_THRESHOLD = 0.9
AVG_WORD_ASPECT = 4.0  # word width : line height ratio for Latin text
# Default padding (pixels) added around a detected bbox before cropping
# for the recognizer. Mirrors `DEFAULT_CROP_PADDING_PX` on the Rust side.
DEFAULT_CROP_PADDING_PX = 1
# Confidence below which the whitespace-aware splitter emits a single
# fallback word — mirrors Rust's `WORD_SPLIT_LOW_CONFIDENCE`.
WORD_SPLIT_LOW_CONFIDENCE = 0.3
# Multiple of the median per-character stride above which a pixel gap
# is treated as an implicit word boundary (recogniser swallowed a
# space). Mirrors Rust's `WHITESPACE_GAP_STRIDE_MULTIPLIER`.
WHITESPACE_GAP_STRIDE_MULTIPLIER = 1.8
# CJK ratio threshold for the "this is a CJK line" branch.
CJK_RATIO_THRESHOLD = 0.30


def is_cjk_char(c: str) -> bool:
    u = ord(c)
    return (
        0x4E00 <= u <= 0x9FFF
        or 0x3400 <= u <= 0x4DBF
        or 0x3040 <= u <= 0x309F
        or 0x30A0 <= u <= 0x30FF
        or 0xAC00 <= u <= 0xD7AF
    )


def cjk_ratio(s: str) -> float:
    non_ws = cjk = 0
    for c in s:
        if c.isspace():
            continue
        non_ws += 1
        if is_cjk_char(c):
            cjk += 1
    return cjk / non_ws if non_ws else 0.0


def sniff_cls_shape(sess) -> tuple[int, int]:
    """Read the classifier input shape from its ONNX signature.

    PP-OCR mobile v2.0 declares (N, 3, 48, 192); v5 textline-orientation
    declares (N, 3, 80, 160); fully-dynamic graphs fall back to (48, 192).
    """
    try:
        dims = sess.get_inputs()[0].shape
        if len(dims) == 4:
            h, w = dims[2], dims[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return h, w
    except Exception:
        pass
    return CLS_INPUT_HEIGHT, CLS_INPUT_WIDTH


def load_labels(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def resize_normalize(img: Image.Image):
    """Resize to max side MAX_SIDE (padded to multiple of 32), normalize."""
    w0, h0 = img.size
    scale = min(MAX_SIDE / max(w0, h0), 1.0)
    nw = max(int(round(w0 * scale / 32)) * 32, 32)
    nh = max(int(round(h0 * scale / 32)) * 32, 32)
    resized = img.resize((nw, nh), Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW, batch=1
    return arr, nw, nh, w0, h0


def db_postprocess(prob_map: np.ndarray, score_thresh: float,
                   model_w: int, model_h: int,
                   orig_w: int, orig_h: int,
                   min_area: int = 16) -> list[dict]:
    """Apply DB post-processing: threshold -> contour -> bounding box."""
    if prob_map.ndim == 4:
        prob_map = prob_map[0, 0]
    elif prob_map.ndim == 3:
        prob_map = prob_map[0]

    binary = (prob_map > score_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    boxes_out = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = prob_map[y:y+h, x:x+w]
        score = float(np.mean(roi)) if roi.size > 0 else 0.0
        if score < score_thresh:
            continue
        ox = int(round(x * scale_x))
        oy = int(round(y * scale_y))
        ow = int(round(w * scale_x))
        oh = int(round(h * scale_y))
        ox = max(0, min(ox, orig_w))
        oy = max(0, min(oy, orig_h))
        ow = min(ow, orig_w - ox)
        oh = min(oh, orig_h - oy)
        if ow <= 0 or oh <= 0:
            continue
        boxes_out.append({
            "label": "text",
            "score": round(score, 4),
            "box": [ox, oy, ow, oh],
            "class_id": 0,
        })

    boxes_out.sort(key=lambda b: -b["score"])
    return boxes_out


# ── Angle classifier ─────────────────────────────────────


def preprocess_for_classifier(crop: np.ndarray,
                               target_h: int = CLS_INPUT_HEIGHT,
                               target_w: int = CLS_INPUT_WIDTH) -> np.ndarray:
    """Resize + edge-replicate-pad or squash to (3, target_h, target_w),
    normalize to [-1, 1], CHW.

    Matches the Rust `preprocess_for_classifier_with` — item #5 in the v4
    review: narrower → BORDER_REPLICATE, wider → squash (not centre-crop).
    """
    h, w = crop.shape[:2]
    h = max(h, 1)
    ratio = w / h
    scaled_w = max(int(round(target_h * ratio)), 1)
    resized = cv2.resize(crop, (scaled_w, target_h), interpolation=cv2.INTER_LINEAR)

    if scaled_w <= target_w:
        # Right-pad by replicating the last column — NOT zeros.
        pad = target_w - scaled_w
        canvas = cv2.copyMakeBorder(
            resized, 0, 0, 0, pad, cv2.BORDER_REPLICATE
        )
    else:
        # Too wide → squash to target_w. Centre-cropping discards text.
        canvas = cv2.resize(resized, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    arr = canvas.astype(np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)  # CHW


def classify_angles(sess: ort.InferenceSession, crops: list[np.ndarray],
                    batch_size: int = 16) -> list[str]:
    """Return an "angle" string ("0" or "180") per crop, in order."""
    input_name = sess.get_inputs()[0].name
    target_h, target_w = sniff_cls_shape(sess)
    angles: list[str] = []
    for i in range(0, len(crops), batch_size):
        chunk = crops[i:i + batch_size]
        batch = np.stack([preprocess_for_classifier(c, target_h, target_w) for c in chunk], axis=0)
        out = sess.run(None, {input_name: batch.astype(np.float32)})[0]
        # Some exports return (N, 2, 1, 1); flatten to (N, 2).
        flat = out.reshape(out.shape[0], -1)
        for row in flat:
            p0, p180 = float(row[0]), float(row[-1])
            # Softmax if the export dropped the final op.
            s = p0 + p180
            if s <= 0 or not np.isfinite(s):
                e0 = np.exp(p0 - max(p0, p180))
                e180 = np.exp(p180 - max(p0, p180))
                p180 = e180 / (e0 + e180)
            elif abs(s - 1.0) >= 0.05:
                p180 = p180 / s
            angles.append("180" if p180 > CLS_FLIP_THRESHOLD else "0")
    return angles


# ── Recognizer ─────────────────────────────────────


def parse_dictionary(path: Path) -> list[str]:
    chars = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    if not chars:
        raise ValueError(f"{path}: dictionary is empty")
    if chars[-1] != " ":
        # PaddleOCR convention — the final vocabulary entry is the space
        # character. Several public dict files drop the trailing line.
        chars.append(" ")
    return chars


def preprocess_for_recognition(crop: np.ndarray) -> tuple[np.ndarray, int]:
    """Resize to height 48, keep aspect, clamp width, normalise to [-1, 1], CHW."""
    h, w = crop.shape[:2]
    h = max(h, 1)
    ratio = w / h
    scaled_w = max(int(round(REC_INPUT_HEIGHT * ratio)), 1)
    if scaled_w > REC_MAX_WIDTH:
        scaled_w = REC_MAX_WIDTH
    resized = cv2.resize(crop, (scaled_w, REC_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1), scaled_w


def ctc_greedy_decode(logits: np.ndarray, dictionary: list[str]) -> tuple[str, float]:
    """Drop blanks (index 0) and collapse consecutive repeats.

    Thin wrapper around `ctc_greedy_decode_with_positions` that discards
    per-character positions.
    """
    text, conf, _ = ctc_greedy_decode_with_positions(
        logits, dictionary, scaled_w=1, batch_input_width=1
    )
    return text, conf


def ctc_greedy_decode_with_positions(
    logits: np.ndarray,
    dictionary: list[str],
    scaled_w: int,
    batch_input_width: int,
) -> tuple[str, float, list[dict]]:
    """CTC greedy decode with per-character positions.

    Mirrors the Rust `ctc_greedy_decode_with_positions`:
    - Per-step argmax over the vocabulary (0 is the CTC blank).
    - Drop blanks and collapse consecutive repeats.
    - For each non-blank emission at step t, record (char, x, confidence)
      where `x = round(t * batch_input_width / T)` (pixel column in the
      recogniser's batched-input tensor), clamped to `scaled_w - 1` so
      emissions from the right-pad zone don't produce coordinates
      beyond the valid line crop.
    """
    idxs = logits.argmax(axis=-1)
    probs = logits.max(axis=-1)
    T = int(logits.shape[0])
    denom = max(T, 1)
    text: list[str] = []
    kept_probs: list[float] = []
    per_char: list[dict] = []
    prev = 0
    max_x = max(int(scaled_w) - 1, 0)
    for t, (idx, p) in enumerate(zip(idxs, probs)):
        idx = int(idx)
        if idx != 0 and idx != prev:
            d = idx - 1
            if 0 <= d < len(dictionary):
                ch = dictionary[d]
                text.append(ch)
                kept_probs.append(float(p))
                x_padded = int(round(t * batch_input_width / denom))
                x = min(x_padded, max_x)
                per_char.append({"char": ch, "x": x, "confidence": float(p)})
        prev = idx
    conf = float(np.mean(kept_probs)) if kept_probs else 0.0
    return "".join(text), conf, per_char


def recognize_crops(sess: ort.InferenceSession, dictionary: list[str],
                    crops: list[np.ndarray], batch_size: int = 8
                    ) -> list[dict]:
    """Recognise each crop and return a list of dicts with text, conf,
    per_char trace, and the per-sample scaled_w / batch_input_width used
    to map char positions back to the original line width.
    """
    if not crops:
        return []
    # Aspect-ratio-sorted batching so crops in the same ONNX call have
    # similar widths — cuts padded-zero waste roughly in half vs. input order.
    indices = sorted(range(len(crops)),
                     key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
    results: list[dict] = [
        {"text": "", "confidence": 0.0, "per_char": [],
         "scaled_w": 0, "batch_input_width": 0}
    ] * len(crops)
    # ^ note: the shared-dict trick above is fine because we overwrite
    # every entry below by index.

    input_name = sess.get_inputs()[0].name
    for i in range(0, len(indices), batch_size):
        chunk = indices[i:i + batch_size]
        per = [preprocess_for_recognition(crops[j]) for j in chunk]
        max_w = max(w for _, w in per)
        batch = np.zeros((len(chunk), 3, REC_INPUT_HEIGHT, max_w), dtype=np.float32)
        for k, (t, w) in enumerate(per):
            batch[k, :, :, :w] = t
        out = sess.run(None, {input_name: batch})[0]
        # out: (N, T, V)
        if out.shape[-1] != len(dictionary) + 1:
            raise RuntimeError(
                f"dict/model vocab mismatch: dict={len(dictionary)} (+blank=1), "
                f"model_vocab={out.shape[-1]}"
            )
        for k, j in enumerate(chunk):
            scaled_w = per[k][1]
            text, conf, per_char = ctc_greedy_decode_with_positions(
                out[k], dictionary, scaled_w=scaled_w, batch_input_width=max_w
            )
            results[j] = {
                "text": text,
                "confidence": conf,
                "per_char": per_char,
                "scaled_w": int(scaled_w),
                "batch_input_width": int(max_w),
            }
    return results


def rotate_180(crop: np.ndarray) -> np.ndarray:
    return cv2.rotate(crop, cv2.ROTATE_180)


# ── Detection-level projection ─────────────────────────────────


def split_line_into_words(box: dict) -> list[dict]:
    left, top, width, height = box["box"]
    h = max(height, 1)
    estimated = round(width / (h * AVG_WORD_ASPECT))
    n = max(1, min(12, int(estimated)))
    if n == 1:
        return [{**box, "box": [left, top, width, height]}]
    step = width // n
    remainder = width % n
    cursor = left
    out = []
    for i in range(n):
        extra = 1 if i < remainder else 0
        ww = step + extra
        out.append({**box, "box": [cursor, top, ww, height]})
        cursor += ww
    return out


def horiz_overlap_ratio(a: list[int], b: list[int]) -> float:
    a_left, _, a_w, _ = a
    b_left, _, b_w, _ = b
    overlap = max(0, min(a_left + a_w, b_left + b_w) - max(a_left, b_left))
    narrower = min(a_w, b_w)
    return overlap / narrower if narrower > 0 else 0.0


def merge_lines_into_blocks(boxes: list[dict]) -> list[dict]:
    if not boxes:
        return boxes
    ordered = sorted(boxes,
                     key=lambda b: (b["box"][1] + b["box"][3] // 2, b["box"][0]))
    heights = sorted(b["box"][3] for b in ordered)
    median_h = max(heights[len(heights) // 2], 1)
    max_gap = median_h * 1.2

    blocks: list[dict] = []
    cur = None
    score_sum = 0.0
    count = 0
    lines_in_block: list[str] = []

    for line in ordered:
        if cur is None:
            cur = {**line, "box": list(line["box"])}
            score_sum = line["score"]; count = 1
            lines_in_block = [line.get("ocr_text", "")]
            continue
        v_gap = line["box"][1] - (cur["box"][1] + cur["box"][3])
        h_ok = horiz_overlap_ratio(cur["box"], line["box"]) >= 0.30
        if v_gap <= max_gap and h_ok:
            l1, t1, w1, h1 = cur["box"]
            l2, t2, w2, h2 = line["box"]
            left = min(l1, l2); top = min(t1, t2)
            right = max(l1 + w1, l2 + w2); bottom = max(t1 + h1, t2 + h2)
            score_sum += line["score"]; count += 1
            cur["box"] = [left, top, right - left, bottom - top]
            cur["score"] = round(score_sum / count, 4)
            lines_in_block.append(line.get("ocr_text", ""))
        else:
            cur["ocr_text"] = "\n".join(lines_in_block)
            blocks.append(cur)
            cur = {**line, "box": list(line["box"])}
            score_sum = line["score"]; count = 1
            lines_in_block = [line.get("ocr_text", "")]
    if cur is not None:
        cur["ocr_text"] = "\n".join(lines_in_block)
        blocks.append(cur)
    return blocks


def split_line_into_words_by_whitespace(
    text: str,
    per_char: list[dict],
    line_confidence: float,
    line_width_px: int,
    scaled_w: int,
) -> list[dict]:
    """Whitespace-aware word splitter — Python mirror of the Rust
    `text_projection::split_line_into_words_by_whitespace`.

    Returns a list of `{"text", "x_in_line", "width", "confidence"}` dicts,
    one per human-readable word. The returned `x_in_line` is in the
    original line-crop pixel frame; callers translate to image
    coordinates via the line's `left`.
    """
    trimmed = text.strip()
    if not trimmed or line_confidence < WORD_SPLIT_LOW_CONFIDENCE or not per_char:
        return [{
            "text": trimmed,
            "x_in_line": 0,
            "width": max(line_width_px, 1),
            "confidence": line_confidence,
        }]

    denom = max(scaled_w, 1)
    scale = max(line_width_px, 1) / denom
    line_w_clamp = max(line_width_px, 1)

    def x_to_line(x):
        return min(int(round(x * scale)), max(line_w_clamp - 1, 0))

    # `text` and `per_char` come from the same CTC decode pass and are
    # 1:1 aligned (including whitespace, because PaddleOCR's dictionary
    # contains the space character). Straight zip is the right primitive.
    chars: list[tuple[str, int, float]] = []
    for i, c in enumerate(text):
        if i < len(per_char):
            ci = per_char[i]
            chars.append((c, int(ci["x"]), float(ci["confidence"])))
        else:
            last_x = chars[-1][1] if chars else 0
            chars.append((c, last_x, 0.0))

    script_is_cjk = cjk_ratio(trimmed) >= CJK_RATIO_THRESHOLD

    # Median gap between consecutive non-whitespace emissions.
    gaps = []
    prev_x = None
    for c, x, _ in chars:
        if c.isspace():
            continue
        if prev_x is not None:
            g = x - prev_x
            if g > 0:
                gaps.append(g)
        prev_x = x
    gaps.sort()
    median_gap = max(gaps[len(gaps) // 2], 1) if gaps else max(scaled_w // max(len(chars), 1), 1)
    gap_threshold = int(median_gap * WHITESPACE_GAP_STRIDE_MULTIPLIER)

    # First pass — whitespace + large-gap tokenisation.
    tokens: list[list[tuple[str, int, float]]] = []
    current: list[tuple[str, int, float]] = []
    last_nonws_x = None
    for entry in chars:
        c, x, _ = entry
        if c.isspace():
            if current:
                tokens.append(current)
                current = []
            last_nonws_x = None
            continue
        if not script_is_cjk and last_nonws_x is not None:
            if x - last_nonws_x >= gap_threshold and current:
                tokens.append(current)
                current = []
        current.append(entry)
        last_nonws_x = x
    if current:
        tokens.append(current)

    if len(tokens) == 1 and not script_is_cjk and not any(ch.isspace() for ch in trimmed):
        tok = tokens[0]
        confs = [c for _, _, c in tok if c > 0.0]
        mean_conf = sum(confs) / len(confs) if confs else line_confidence
        return [{
            "text": "".join(c for c, _, _ in tok),
            "x_in_line": 0,
            "width": line_w_clamp,
            "confidence": mean_conf,
        }]

    # CJK runs inside tokens get exploded to one word per glyph.
    final_tokens: list[list[tuple[str, int, float]]] = []
    for tok in tokens:
        if script_is_cjk:
            ascii_run: list[tuple[str, int, float]] = []
            for entry in tok:
                if is_cjk_char(entry[0]):
                    if ascii_run:
                        final_tokens.append(ascii_run)
                        ascii_run = []
                    final_tokens.append([entry])
                else:
                    ascii_run.append(entry)
            if ascii_run:
                final_tokens.append(ascii_run)
        else:
            final_tokens.append(tok)

    spans: list[dict] = []
    for i, tok in enumerate(final_tokens):
        if not tok:
            continue
        first_x = tok[0][1]
        last_x = tok[-1][1]
        confs = [c for _, _, c in tok if c > 0.0]
        mean_conf = sum(confs) / len(confs) if confs else line_confidence

        if i == 0:
            left_x = x_to_line(first_x)
        else:
            prev_last = final_tokens[i - 1][-1][1]
            left_x = x_to_line((prev_last + first_x) // 2)
        if i + 1 < len(final_tokens):
            next_first = final_tokens[i + 1][0][1]
            right_x = x_to_line((last_x + next_first) // 2)
        else:
            right_x = line_w_clamp

        width = max(right_x - left_x, 1)
        spans.append({
            "text": "".join(c for c, _, _ in tok),
            "x_in_line": left_x,
            "width": width,
            "confidence": mean_conf,
        })

    if not spans:
        return [{
            "text": trimmed,
            "x_in_line": 0,
            "width": line_w_clamp,
            "confidence": line_confidence,
        }]
    return spans


def split_line_text(text: str, n_words: int) -> list[str]:
    """CJK-aware word-level text splitter mirroring the Rust heuristic."""
    if n_words <= 1:
        return [text] if n_words == 1 else []
    trimmed = text.strip()
    if any(ch.isspace() for ch in trimmed):
        tokens = trimmed.split()
        if len(tokens) == n_words:
            return tokens
        cleaned = "".join(tokens)
    else:
        cleaned = trimmed
    if not cleaned:
        return [""] * n_words
    chars = list(cleaned)
    per = len(chars) // n_words
    remainder = len(chars) % n_words
    out = []
    cursor = 0
    for i in range(n_words):
        extra = 1 if i < remainder else 0
        take = per + extra
        end = min(cursor + take, len(chars))
        out.append("".join(chars[cursor:end]))
        cursor = end
    return out


def project_level(boxes: list[dict], level: str, debug: bool = False) -> list[dict]:
    if level == "word":
        projected: list[dict] = []
        for b in boxes:
            # Prefer the whitespace-aware splitter when we have a
            # per-character trace from the recogniser. Fall back to
            # the legacy geometric split when we don't (e.g. when
            # recognition was skipped).
            per_char = b.get("per_char")
            if per_char is not None:
                left, top, width, height = b["box"]
                spans = split_line_into_words_by_whitespace(
                    b.get("ocr_text", ""),
                    per_char,
                    b.get("score", 0.0),
                    width,
                    int(b.get("scaled_w", width)),
                )
                for span in spans:
                    projected.append({
                        **b,
                        "box": [left + span["x_in_line"], top, span["width"], height],
                        "ocr_text": span["text"],
                        "score": span["confidence"],
                    })
            else:
                words = split_line_into_words(b)
                texts = split_line_text(b.get("ocr_text", ""), len(words))
                for w, t in zip(words, texts):
                    projected.append({**w, "ocr_text": t})
    elif level == "block":
        projected = merge_lines_into_blocks(boxes)
    else:
        projected = [{**b, "box": list(b["box"])} for b in boxes]

    out = []
    for i, b in enumerate(projected, start=1):
        left, top, width, height = b["box"]
        entry: dict = {
            "region_id": f"r{i}",
            "label": b["label"],
            "class_id": b.get("class_id", 0),
            "confidence": b["score"],
            "level": level,
            "ocr_text": b.get("ocr_text", ""),
            "bbox": [left, top, left + width, top + height],
            "img_coord": [left, top, width, height],
        }
        # Emit the per-char debug block only at line level — word/block
        # projections derive from line data and would just duplicate it.
        if debug and level == "line" and b.get("per_char"):
            entry["debug"] = {"per_char": b["per_char"]}
        out.append(entry)
    return out


def crop_bgr(img_bgr: np.ndarray, box: list[int], padding_px: int = 0) -> np.ndarray:
    """Crop a bbox out of the image, optionally expanding by
    `padding_px` on each side before cropping. The expansion is
    clamped to image bounds so the padded crop never exceeds the
    source dimensions.

    The padding is applied to the recognizer's input only — the bbox
    values returned in the final JSON are the unpadded detector
    output, matching the Rust pipeline's contract.
    """
    h, w = img_bgr.shape[:2]
    orig_l, orig_t, orig_w, orig_h = box
    # Expand by padding_px on each side, clamping at image edges.
    l = max(0, orig_l - padding_px)
    t = max(0, orig_t - padding_px)
    r = min(w, orig_l + orig_w + padding_px)
    b = min(h, orig_t + orig_h + padding_px)
    if r <= l or b <= t:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img_bgr[t:b, l:r, :].copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--score", type=float, default=0.3)
    p.add_argument("--level", choices=("word", "line", "block"), default="line")
    p.add_argument("--language", default="en",
                   help="Recognizer dir name under runtime/ocr_rec/ (default: en).")
    p.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR,
                   help="Root of the shared runtime/ layout.")
    p.add_argument("--skip-recognition", action="store_true")
    p.add_argument("--skip-angle-cls", action="store_true")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop regions below this blended confidence. 0.0 keeps all. "
             "Typical usable range for PP-OCRv4: 0.6–0.8.",
    )
    p.add_argument(
        "--crop-padding",
        type=int,
        default=DEFAULT_CROP_PADDING_PX,
        help=f"Pixels to expand each detector bbox before cropping for the "
             f"recognizer (default: {DEFAULT_CROP_PADDING_PX}). Applied to the "
             f"crop only — the bbox in the returned JSON is the unpadded "
             f"detector output. Set to 0 to disable for A/B testing.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Emit a `debug.per_char` block on each line-level region with "
             "per-character {char, x, confidence} from the recognizer. "
             "Skipped at word/block level (they derive from line data).",
    )
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"✗ {args.onnx} not found — run fetch_onnx.sh first", file=sys.stderr)
        return 2
    if not args.image.exists():
        print(f"✗ {args.image} not found", file=sys.stderr)
        return 2

    labels = load_labels(args.labels)
    img_pil = Image.open(args.image).convert("RGB")
    tensor, model_w, model_h, orig_w, orig_h = resize_normalize(img_pil)

    det_sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    det_input = det_sess.get_inputs()[0].name
    det_t0 = time.perf_counter()
    raw = det_sess.run(None, {det_input: tensor})[0]  # (1, 1, H, W) prob map
    boxes = db_postprocess(raw, args.score, model_w, model_h, orig_w, orig_h)
    det_ms = int((time.perf_counter() - det_t0) * 1000)

    # Detector is the only mandatory model; classifier + recognizer are
    # optional and each failure logs a single warning.
    print(f"\ndetector: pp-ocrv4-det (loaded {args.onnx.name})", file=sys.stderr)

    cls_path = args.runtime_dir / "ocr_cls" / "ocr_cls.onnx"
    cls_sess = None
    if args.skip_angle_cls:
        print("angle-cls: skipped (--skip-angle-cls)", file=sys.stderr)
    elif cls_path.exists():
        cls_sess = ort.InferenceSession(str(cls_path), providers=["CPUExecutionProvider"])
        print(f"angle-cls: pp-ocr-cls (loaded {cls_path.name})", file=sys.stderr)
    else:
        print(f"angle-cls: NOT FOUND at {cls_path} — skipping", file=sys.stderr)

    rec_sess = None
    dictionary: list[str] = []
    rec_dir = args.runtime_dir / "ocr_rec" / args.language
    rec_onnx = rec_dir / "ocr_rec.onnx"
    rec_dict = next(iter(sorted(rec_dir.glob("ppocr_keys_*.txt"))), None) if rec_dir.exists() else None
    if rec_dict is None and rec_dir.exists():
        rec_dict = next(iter(sorted(rec_dir.glob("*.txt"))), None)
    if args.skip_recognition:
        print(f"recognizer: skipped (--skip-recognition), language={args.language}", file=sys.stderr)
    elif rec_onnx.exists() and rec_dict is not None:
        rec_sess = ort.InferenceSession(str(rec_onnx), providers=["CPUExecutionProvider"])
        dictionary = parse_dictionary(rec_dict)
        print(f"recognizer: pp-ocrv4-rec[{args.language}] (loaded {rec_onnx.name}, "
              f"dict={rec_dict.name}, |V|={len(dictionary) + 1})", file=sys.stderr)
    else:
        print(f"recognizer: NOT FOUND for `{args.language}` under {rec_dir} — "
              f"ocr_text will be empty", file=sys.stderr)

    # Crop each line-level box in the original image, optionally classify
    # + rotate, then recognise. Results are attached back to `boxes`.
    img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    # Padding is applied to the recognizer crop only; the bbox in the
    # returned JSON stays at the unpadded detector coordinates.
    crop_pad = max(0, int(args.crop_padding))
    crops = [crop_bgr(img_bgr, b["box"], padding_px=crop_pad) for b in boxes]
    cls_ms = 0
    if cls_sess is not None and crops:
        cls_t0 = time.perf_counter()
        angles = classify_angles(cls_sess, crops)
        crops = [rotate_180(c) if a == "180" else c for c, a in zip(crops, angles)]
        cls_ms = int((time.perf_counter() - cls_t0) * 1000)
    rec_ms = 0
    if rec_sess is not None and crops:
        rec_t0 = time.perf_counter()
        rec_results = recognize_crops(rec_sess, dictionary, crops)
        rec_ms = int((time.perf_counter() - rec_t0) * 1000)
        for b, r in zip(boxes, rec_results):
            b["ocr_text"] = r["text"]
            b["score"] = round((b["score"] + r["confidence"]) * 0.5, 4)
            # Stash the per-char trace + batch metadata for the
            # whitespace-aware word splitter (downstream) and the
            # optional --debug per-char block.
            b["per_char"] = r["per_char"]
            b["scaled_w"] = r["scaled_w"]
            b["batch_input_width"] = r["batch_input_width"]
    else:
        for b in boxes:
            b["ocr_text"] = ""

    # Apply confidence filter before level projection so word/block boxes
    # inherit the filtered set correctly.
    if args.min_confidence > 0.0:
        boxes = [b for b in boxes if b["score"] >= args.min_confidence]

    projected = project_level(boxes, args.level, debug=args.debug)
    payload = {
        "page_index": 0,
        "detection_level": args.level,
        "detection_model": "pp-ocrv4-det",
        "recognition_model": "pp-ocrv4-rec" if rec_sess is not None else None,
        "classification_model": "pp-ocr-cls" if cls_sess is not None else None,
        "language": args.language,
        "timings": {
            "detection_ms": det_ms,
            "classification_ms": cls_ms,
            "recognition_ms": rec_ms,
            "total_ms": det_ms + cls_ms + rec_ms,
        },
        "boxes": projected,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"\n{len(projected)} {args.level}-level regions "
        f"(from {len(boxes)} line detections, language={args.language}, "
        f"det={det_ms}ms cls={cls_ms}ms rec={rec_ms}ms)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
