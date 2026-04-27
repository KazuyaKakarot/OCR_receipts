"""
demo.py
-------
Demonstrates the pipeline WITHOUT needing real receipt images.
It uses a synthetic text receipt to show exactly what the JSON output looks like.

Run this to verify everything is working:
    python demo.py
"""

import json
from extractor import extract_fields
from summarizer import generate_summary


# A synthetic receipt text simulating what Tesseract might return
SAMPLE_RECEIPT = """
RELIANCE FRESH
MG Road, Bengaluru
GSTIN: 29AABCR1234A1Z5

Date: 15/03/2024        Time: 14:32

------------------------------
Bread Brown 400g       45.00
Amul Milk 1L           68.00
Eggs (12 pcs)         110.00
Bisleri 1L             20.00
Lay's Classic 26g      20.00
------------------------------
Subtotal              263.00
GST (5%)               13.15
Total                 276.15

Cash                  300.00
Change                 23.85

Thank you for shopping!
"""


def run_demo():
    print("=" * 55)
    print("  DEMO MODE — Synthetic Receipt")
    print("=" * 55)
    print("\nSample Receipt Text:")
    print(SAMPLE_RECEIPT)

    # Simulate OCR confidence: pretend Tesseract gave us 88% average
    simulated_confidences = [0.88] * 50

    result = extract_fields(SAMPLE_RECEIPT, simulated_confidences)
    result["source_file"] = "demo_receipt.jpg"
    result["warnings"] = []

    for field in ["store_name", "date", "total_amount"]:
        if field in result and result[field]["confidence"] < 0.70:
            result["warnings"].append(
                f"Low confidence on '{field}': {result[field]['confidence']:.2f}"
            )

    print("\n📦 Extracted JSON Output:")
    print(json.dumps(result, indent=2))

    summary = generate_summary([result])
    print("\n📊 Financial Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_demo()
