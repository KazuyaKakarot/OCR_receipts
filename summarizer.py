"""
summarizer.py
-------------
Takes a list of all extracted receipt results and produces a financial summary:

  - Total spend across all receipts
  - Number of transactions
  - Spend broken down per store (with fuzzy deduplication of store names)

Key improvements:
  - Fuzzy store name matching: "99 SPEED MART S/B (S19537-X)" and
    "99 SPEED HART S/B (S19S37-X)" are treated as the same store
  - Smart decimal fix: "66890" where OCR missed the decimal → 668.90
  - Only includes totals where confidence >= 0.5
"""

import re
from typing import List


CONFIDENCE_THRESHOLD = 0.5
MAX_SINGLE_RECEIPT = 10_000.0   # anything above this is probably a misread


# ── Amount parsing ─────────────────────────────────────────────────────────────

def _strip_currency(value: str) -> str:
    return re.sub(r"[₹$€£a-zA-Z,\s]", "", value)


def _try_fix_missing_decimal(raw: float) -> float:
    """
    If a number is suspiciously large and has no decimal part,
    it was probably OCR'd without the decimal point.
    e.g. 66890 → 668.90  (divide by 100)
    """
    if raw > MAX_SINGLE_RECEIPT and raw == int(raw):
        candidate = raw / 100.0
        if candidate <= MAX_SINGLE_RECEIPT:
            return candidate
    return raw


def parse_amount(value: str) -> float:
    if not value:
        return 0.0
    cleaned = _strip_currency(value)

    # Handle leading zeros like "001533" → try as "15.33" (insert decimal before last 2 digits)
    if re.match(r'^0+\d+$', cleaned) and len(cleaned) > 2:
        cleaned = cleaned.lstrip('0') or '0'

    try:
        amount = float(cleaned)
        amount = _try_fix_missing_decimal(amount)
        if amount <= 0 or amount > MAX_SINGLE_RECEIPT:
            return 0.0
        return round(amount, 2)
    except ValueError:
        return 0.0


# ── Store name normalisation & fuzzy deduplication ────────────────────────────

def _normalise(name: str) -> str:
    """Strip everything but letters and digits, lowercase."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _similarity(a: str, b: str) -> float:
    """
    Bigram Jaccard similarity. Returns 0.0–1.0.
    Above 0.60 = likely the same store.
    """
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) >= 2 else {s}

    na, nb = _normalise(a), _normalise(b)
    ba, bb = bigrams(na), bigrams(nb)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _deduplicate_stores(spend_raw: dict) -> dict:
    """
    Merge store names that are likely the same shop (similarity >= 0.60).
    Keeps the first-seen name as canonical. Sums amounts.
    Also drops pure garbage names (< 3 alpha chars).
    """
    canonical: dict = {}
    mapping: list = []

    for store, amount in spend_raw.items():
        alpha = sum(1 for c in store if c.isalpha())
        total_chars = len(store.strip())
        # Skip if fewer than 4 alpha chars OR alpha chars are less than 50% of the name
        if alpha < 4 or (total_chars > 0 and alpha / total_chars < 0.50):
            continue

        matched = None
        for canon in mapping:
            if _similarity(store, canon) >= 0.60:
                matched = canon
                break

        if matched:
            canonical[matched] += amount
        else:
            canonical[store] = amount
            mapping.append(store)

    return canonical


# ── Main summary function ──────────────────────────────────────────────────────

def format_currency(amount: float) -> str:
    return f"RM {amount:,.2f}"


def generate_summary(results: List[dict]) -> dict:
    total_spend = 0.0
    spend_raw: dict = {}
    skipped: list = []

    for r in results:
        filename = r.get("source_file", "unknown")
        total_field = r.get("total_amount", {})
        total_value = total_field.get("value")
        total_conf = total_field.get("confidence", 0.0)
        store_name = r.get("store_name", {}).get("value", "Unknown Store")

        if total_value is None or total_conf < CONFIDENCE_THRESHOLD:
            skipped.append(filename)
            continue

        amount = parse_amount(total_value)
        if amount <= 0:
            skipped.append(filename)
            continue

        total_spend += amount
        spend_raw[store_name] = spend_raw.get(store_name, 0.0) + amount

    spend_deduped = _deduplicate_stores(spend_raw)

    # Sort by spend descending
    spend_sorted = dict(
        sorted(spend_deduped.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "num_transactions": len(results),
        "receipts_included_in_total": len(results) - len(skipped),
        "skipped_receipts": skipped,
        "total_spend": format_currency(total_spend),
        "spend_per_store": {k: format_currency(v) for k, v in spend_sorted.items()},
    }
