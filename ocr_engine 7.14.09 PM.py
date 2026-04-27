"""
ocr_engine.py
-------------
Wraps Tesseract OCR via pytesseract.
Returns:
  - raw_text         : full extracted string from the receipt
  - word_confidences : list of per-word confidence scores (0–100 from Tesseract)

Tesseract's --psm 6 treats the image as a single uniform block of text,
which works well for receipts. You can also try --psm 4 (single column)
if results are poor.
"""

import pytesseract
import numpy as np
from typing import Tuple, List


# ── OCR Configuration ─────────────────────────────────────────────────────────
# Page Segmentation Mode 6 = "Assume a single uniform block of text"
# OEM 3 = Use LSTM neural net engine (most accurate)
# Use English + Malay + Chinese (simplified) — covers this dataset well
# Add more languages with '+' if needed e.g. "eng+msa+chi_sim+chi_tra"
TESSERACT_CONFIG = "--psm 6 --oem 3"
TESSERACT_LANG = "eng+msa+chi_sim"


def run_ocr(image: np.ndarray) -> Tuple[str, List[float]]:
    """
    Run Tesseract on a preprocessed image.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale/binary image from preprocessor.py

    Returns
    -------
    raw_text : str
        All text extracted from the receipt as a single string.
    word_confidences : List[float]
        Per-word confidence scores normalized to 0–1.
        (Tesseract natively returns 0–100; we divide by 100.)
    """
    # ── Get raw text ──────────────────────────────────────────────────────────
    raw_text: str = pytesseract.image_to_string(
        image,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG
    )

    # ── Get detailed data including confidence per word ───────────────────────
    # image_to_data returns a TSV-style string with columns:
    # level, page_num, block_num, par_num, line_num, word_num,
    # left, top, width, height, conf, text
    data = pytesseract.image_to_data(
        image,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    # Extract confidence scores for actual words (conf = -1 means no text)
    word_confidences: List[float] = []
    for conf, text in zip(data["conf"], data["text"]):
        if str(conf).strip() != "-1" and str(text).strip():
            try:
                word_confidences.append(float(conf) / 100.0)
            except ValueError:
                pass

    return raw_text, word_confidences


def average_confidence(word_confidences: List[float]) -> float:
    """
    Compute the mean OCR confidence for the whole page.
    Used as a fallback confidence estimate when field-level detection is weak.
    """
    if not word_confidences:
        return 0.0
    return round(sum(word_confidences) / len(word_confidences), 4)
