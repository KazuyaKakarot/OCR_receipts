# 🧾 Carbon Crunch — Receipt OCR Pipeline

A complete OCR pipeline for extracting structured data from receipt images, with confidence-aware outputs and financial summarization.

## 📁 Project Structure

```
receipt_ocr/
│
├── pipeline.py          ← Main script — run this to process all receipts
├── demo.py              ← Quick demo (no images needed)
├── preprocessor.py      ← Image cleaning: noise, skew, contrast
├── ocr_engine.py        ← Tesseract OCR wrapper
├── extractor.py         ← Field extraction + confidence scoring
├── summarizer.py        ← Financial summary generation
│
├── receipts/            ← ⬅ PUT YOUR RECEIPT IMAGES HERE
├── outputs/
│   ├── json/            ← One .json file per receipt
│   └── summary/         ← financial_summary.json
│
├── docs/
│   └── documentation.md ← Full technical write-up
│
└── requirements.txt
```

## ⚡ Quick Start

```bash
# 1. Install Tesseract
sudo apt-get install tesseract-ocr        # Linux
brew install tesseract                    # macOS

# 2. Install Python packages
pip install -r requirements.txt

# 3. Try the demo (no images needed)
python demo.py

# 4. Process real receipts
#    → Copy images into receipts/ folder
#    → Then run:
python pipeline.py
```

## 📊 What It Extracts

| Field | With Confidence Score |
|---|---|
| Store / Vendor Name | ✅ |
| Date of Transaction | ✅ |
| List of Items | ✅ |
| Item Prices | ✅ |
| Total Amount | ✅ |

Low-confidence fields (< 0.70) are automatically flagged with warnings.
