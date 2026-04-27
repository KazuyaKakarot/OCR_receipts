"""
pipeline.py
-----------
Main entry point. Run this file to process all receipts in the /receipts folder.

Usage:
    python pipeline.py

It will:
1. Preprocess each image
2. Run Tesseract OCR
3. Extract fields with confidence scores
4. Save JSON output per receipt
5. Generate a financial summary
"""

import os
import json
from preprocessor import preprocess_image
from ocr_engine import run_ocr
from extractor import extract_fields
from summarizer import generate_summary

# ── Folder paths ──────────────────────────────────────────────────────────────
RECEIPTS_DIR = "receipts"          # Put your receipt images here
JSON_OUTPUT_DIR = "outputs/json"   # One JSON per receipt
SUMMARY_OUTPUT = "outputs/summary/financial_summary.json"


def process_receipt(image_path: str) -> dict:
    """
    Full pipeline for a single receipt image.
    Returns a structured dict with all extracted fields + confidence scores.
    """
    filename = os.path.basename(image_path)
    print(f"\n📄 Processing: {filename}")

    # Step 1: Preprocess the image (denoise, deskew, fix contrast)
    preprocessed = preprocess_image(image_path)

    # Step 2: Run Tesseract OCR → get raw text + word-level confidence scores
    raw_text, word_confidences = run_ocr(preprocessed)

    # Step 3: Extract structured fields with confidence scores
    result = extract_fields(raw_text, word_confidences)
    result["source_file"] = filename

    # Step 4: Flag low-confidence fields (< 0.70)
    result["warnings"] = []
    for field in ["store_name", "date", "total_amount"]:
        if field in result and result[field]["confidence"] < 0.70:
            result["warnings"].append(
                f"Low confidence on '{field}': {result[field]['confidence']:.2f}"
            )

    return result


def save_json(data: dict, output_dir: str, filename: str):
    """Save extracted data as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   ✅ Saved → {out_path}")
    return out_path


def main():
    print("=" * 55)
    print("  CARBON CRUNCH — Receipt OCR Pipeline")
    print("=" * 55)

    # Collect all image files from receipts/ folder
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        os.path.join(RECEIPTS_DIR, f)
        for f in os.listdir(RECEIPTS_DIR)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    if not image_files:
        print(f"\n⚠️  No images found in '{RECEIPTS_DIR}/' folder.")
        print("   Add your receipt images (.jpg/.png/.tiff) and re-run.")
        return

    print(f"\nFound {len(image_files)} receipt(s) to process.")

    all_results = []
    for img_path in sorted(image_files):
        try:
            result = process_receipt(img_path)
            save_json(result, JSON_OUTPUT_DIR, result["source_file"])
            all_results.append(result)

            # Print a quick preview in terminal
            sn = result.get("store_name", {}).get("value", "Unknown")
            dt = result.get("date", {}).get("value", "Unknown")
            ta = result.get("total_amount", {}).get("value", "Unknown")
            print(f"   Store: {sn} | Date: {dt} | Total: {ta}")
            if result["warnings"]:
                for w in result["warnings"]:
                    print(f"   ⚠️  {w}")

        except Exception as e:
            print(f"   ❌ Error processing {img_path}: {e}")

    # Generate financial summary across all receipts
    summary = generate_summary(all_results)
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("  FINANCIAL SUMMARY")
    print("=" * 55)
    print(f"  Receipts processed : {summary['num_transactions']}")
    print(f"  Total spend        : {summary['total_spend']}")
    print(f"  Spend per store    :")
    for store, amount in summary["spend_per_store"].items():
        print(f"      {store:30s}  {amount}")
    print(f"\n  Summary saved → {SUMMARY_OUTPUT}")
    print("=" * 55)


if __name__ == "__main__":
    main()
