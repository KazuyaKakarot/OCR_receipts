"""
extractor.py
------------
Extracts structured fields from raw OCR text.

For each field we assign a confidence score (0.0–1.0) based on:
  1. Average OCR confidence of surrounding words
  2. Pattern validation (does the value look like a real date / currency?)
  3. Keyword heuristics (did we find it near "Total", "Date", etc.?)

Output schema per field:
    {
        "value": "...",
        "confidence": 0.93
    }
"""

import re
from typing import List, Tuple, Optional


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Date: matches DD/MM/YYYY, MM-DD-YYYY, YYYY.MM.DD, "12 Jan 2024", etc.
DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",           # 12/03/2024
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",             # 2024/03/12
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b",  # 12 Jan 2024
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",  # Jan 12, 2024
]

# Total amount: a currency-like number, optionally preceded by keywords
TOTAL_PATTERNS = [
    # Strong: keyword directly before a currency amount
    r"(?:total|grand\s*total|amount\s*due|balance\s*due|net\s*total|total\s*amount)"
    r"\s*[:\-]?\s*"
    r"((?:rs\.?|rm\.?|inr|myr|₹|\$|€|£|usd)?\s*[\d,]{1,7}\.?\d{0,2})",

    # Weaker: currency symbol followed by a reasonable number (max 7 digits before decimal)
    r"((?:rs\.?|rm\.?|inr|myr|₹|\$|€|£)\s*[\d,]{1,7}\.?\d{0,2})",
]

# Sanity cap: no single receipt total should exceed this (adjust if needed)
MAX_PLAUSIBLE_TOTAL = 100_000.0

# Item line: "ITEM NAME  <spaces>  12.50" — captures name + price
ITEM_PATTERN = re.compile(
    r"^(.+?)\s{2,}((?:rs\.?|inr|₹|\$|€|£|usd)?\s*[\d,]+\.\d{2})\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Keywords that suggest a line contains the store name (near the top)
STORE_KEYWORDS = [
    "store", "shop", "mart", "supermarket", "restaurant", "cafe",
    "hotel", "pharmacy", "medical", "bakery", "pvt", "ltd", "inc",
    "co.", "enterprises", "retail", "foods", "fresh"
]


# ── Helper functions ───────────────────────────────────────────────────────────

def _avg_conf(word_confidences: List[float]) -> float:
    """Mean of a list of confidence scores."""
    return round(sum(word_confidences) / len(word_confidences), 4) if word_confidences else 0.5


def _search(patterns: List[str], text: str, flags=re.IGNORECASE) -> Optional[str]:
    """Try each regex pattern in order; return first match group 1, or None."""
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None


def _pattern_confidence(value: Optional[str], patterns: List[str]) -> float:
    """
    How well does the extracted value match the expected pattern?
    Returns 1.0 if it matches perfectly, 0.5 if found without validation,
    0.3 if nothing was found.
    """
    if value is None:
        return 0.3
    for pat in patterns:
        if re.fullmatch(pat.strip(), value.strip(), re.IGNORECASE):
            return 1.0
    return 0.75  # Found via heuristic but doesn't fully match expected format


# ── Store name extraction ──────────────────────────────────────────────────────

def _is_garbage_line(line: str) -> bool:
    """
    Returns True if a line looks like OCR noise rather than a real store name.
    Rules:
      - Less than 3 alphabetic characters
      - More than 40% of characters are symbols/punctuation
      - Looks like a barcode or ID (all digits)
    """
    alpha_count = sum(1 for c in line if c.isalpha())
    symbol_count = sum(1 for c in line if not c.isalnum() and not c.isspace())

    if alpha_count < 3:
        return True
    if len(line) > 0 and symbol_count / len(line) > 0.40:
        return True
    if line.strip().isdigit():
        return True
    return False


def extract_store_name(lines: list, avg_ocr_conf: float) -> dict:
    """
    Strategy:
      1. Filter out garbage lines first.
      2. Among clean lines, prefer those with known store keywords.
      3. Fall back to the first clean non-empty line.
    """
    clean_lines = [l.strip() for l in lines if l.strip() and not _is_garbage_line(l.strip())]

    if not clean_lines:
        return {"value": "Unknown", "confidence": 0.1}

    # Check first 5 clean lines for known store keywords
    for line in clean_lines[:5]:
        if any(kw in line.lower() for kw in STORE_KEYWORDS):
            conf = round(min(avg_ocr_conf * 1.1, 1.0), 4)
            return {"value": line, "confidence": conf}

    # Fallback: first clean line
    value = clean_lines[0]
    conf = round(avg_ocr_conf * 0.85, 4)
    return {"value": value, "confidence": conf}


# ── Date extraction ────────────────────────────────────────────────────────────

DATE_VALIDATION = re.compile(
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|"
    r"(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})|"
    r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})",
    re.IGNORECASE
)

def extract_date(text: str, avg_ocr_conf: float) -> dict:
    """
    Try all date patterns. Validate the result with DATE_VALIDATION.
    Also boost confidence if the line contains 'date' keyword.
    """
    text_lower = text.lower()
    value = _search(DATE_PATTERNS, text)

    if value is None:
        return {"value": None, "confidence": 0.2}

    # Check if 'date' keyword is nearby → higher confidence
    keyword_boost = 0.1 if "date" in text_lower else 0.0

    # Is it a clean, validatable date format?
    if DATE_VALIDATION.fullmatch(value.strip()):
        conf = round(min(avg_ocr_conf + 0.1 + keyword_boost, 1.0), 4)
    else:
        conf = round(min(avg_ocr_conf * 0.8 + keyword_boost, 1.0), 4)

    return {"value": value, "confidence": conf}


# ── Total amount extraction ────────────────────────────────────────────────────

CURRENCY_VALIDATION = re.compile(
    r"(?:rs\.?|inr|₹|\$|€|£)?\s*[\d,]+\.?\d{0,2}",
    re.IGNORECASE
)

def extract_total(text: str, avg_ocr_conf: float) -> dict:
    """
    Search for total amount near keywords like 'Total', 'Grand Total', 'Amount Due'.
    Validates as currency and rejects implausibly large numbers (barcodes, phone numbers).
    """
    value = _search(TOTAL_PATTERNS, text)

    if value is None:
        return {"value": None, "confidence": 0.2}

    value = re.sub(r"\s+", " ", value).strip()

    # Plausibility check: parse and reject if it exceeds a sane receipt total
    try:
        numeric = float(re.sub(r"[^\d.]", "", value.replace(",", "")))
        if numeric > MAX_PLAUSIBLE_TOTAL or numeric <= 0:
            return {"value": None, "confidence": 0.15}
    except ValueError:
        return {"value": None, "confidence": 0.15}

    if CURRENCY_VALIDATION.fullmatch(value.strip()):
        conf = round(min(avg_ocr_conf + 0.15, 1.0), 4)
    else:
        conf = round(avg_ocr_conf * 0.75, 4)

    return {"value": value, "confidence": conf}


# ── Item list extraction ───────────────────────────────────────────────────────

def extract_items(text: str, avg_ocr_conf: float) -> List[dict]:
    """
    Find lines that look like 'ITEM NAME    12.50'.
    We look for lines with 2+ spaces between text and a price.
    """
    items = []
    matches = ITEM_PATTERN.findall(text)

    for name, price in matches:
        name = name.strip()
        price = price.strip()

        # Skip lines that are clearly headers or totals
        skip_keywords = ["total", "subtotal", "tax", "vat", "discount", "change", "cash"]
        if any(kw in name.lower() for kw in skip_keywords):
            continue

        # Per-item confidence: slightly lower than page average (items are harder)
        item_conf = round(avg_ocr_conf * 0.9, 4)
        items.append({
            "name": {"value": name, "confidence": item_conf},
            "price": {"value": price, "confidence": item_conf}
        })

    return items


# ── Master extraction function ─────────────────────────────────────────────────

def extract_fields(raw_text: str, word_confidences: List[float]) -> dict:
    """
    Given raw OCR text and Tesseract's per-word confidence scores,
    return a fully structured dict with all fields and their confidence scores.

    This is what the main pipeline calls.
    """
    avg_conf = _avg_conf(word_confidences)
    lines = raw_text.splitlines()

    store = extract_store_name(lines, avg_conf)
    date = extract_date(raw_text, avg_conf)
    total = extract_total(raw_text, avg_conf)
    items = extract_items(raw_text, avg_conf)

    return {
        "store_name": store,
        "date": date,
        "items": items,
        "total_amount": total,
        "ocr_page_confidence": round(avg_conf, 4),
    }
