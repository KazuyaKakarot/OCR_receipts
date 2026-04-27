# Carbon Crunch — Receipt OCR Pipeline
## Technical Documentation

---

## 1. Approach

The pipeline is structured as five independent, composable modules that form an end-to-end OCR system for receipt data extraction.

### Pipeline Flow

```
Receipt Image
     ↓
[preprocessor.py]  → Fix noise, contrast, skew
     ↓
[ocr_engine.py]    → Tesseract OCR → raw text + per-word confidence scores
     ↓
[extractor.py]     → Extract fields + assign confidence scores
     ↓
[summarizer.py]    → Aggregate financial summary
     ↓
JSON Output per receipt + financial_summary.json
```

### Confidence Scoring Strategy

Each field gets a composite confidence score (0.0–1.0) combining three signals:

| Signal | Description |
|---|---|
| OCR confidence | Average of Tesseract's per-word scores for the page |
| Pattern validation | Does the extracted value match a known format? (date regex, currency regex) |
| Keyword heuristics | Was the field found near a strong keyword like "Total" or "Date"? |

Fields below 0.70 are flagged as warnings in the output JSON.

---

## 2. Tools Used

| Tool | Purpose |
|---|---|
| **Tesseract OCR** | Text recognition engine (open-source, battle-tested) |
| **pytesseract** | Python wrapper for Tesseract |
| **OpenCV (cv2)** | Image preprocessing (contrast, deskew, binarization) |
| **NumPy** | Image array manipulation |
| **Python re** | Regex-based field extraction and validation |
| **json** | Structured output serialization |

---

## 3. Preprocessing Techniques

### 3.1 Contrast Enhancement (CLAHE)
Receipts often have uneven lighting (flash glare, shadows). CLAHE (Contrast Limited Adaptive Histogram Equalization) applies localized contrast correction, which handles lighting gradients without over-brightening uniform areas.

### 3.2 Noise Removal
A 3×3 Gaussian blur smooths pixel-level noise (camera grain, scanner artifacts) while preserving edge sharpness on characters.

### 3.3 Binarization (Otsu's Method)
Converts the grayscale image to pure black-and-white. Otsu's algorithm automatically selects the optimal threshold, making it robust to varying brightness levels across different receipts.

### 3.4 Deskewing
Calculates the skew angle using `cv2.minAreaRect()` on detected text pixels, then applies a rotation correction. Only activates when skew > 0.5° to avoid degrading already-straight images.

### 3.5 Upscaling
Images narrower than 1000px are doubled in size before OCR. Tesseract's accuracy degrades significantly on small images; upscaling helps the neural network identify characters correctly.

---

## 4. Challenges Faced

**1. Store name detection**
Receipts have no consistent label for the store name — it's just at the top. The solution uses a two-step heuristic: first check the top 5 lines for known business keywords (e.g., "Pvt", "Mart", "Store"), then fall back to the first non-empty line.

**2. Currency format diversity**
Receipts may use ₹, Rs., INR, $, € or no symbol at all. Extraction uses a union regex that handles all major formats.

**3. Item-line parsing**
Tesseract doesn't preserve column alignment perfectly. The item extractor uses "2 or more spaces" as a delimiter between item name and price, which is a reasonable heuristic for most tabular receipt layouts.

**4. Low-quality image handling**
For very noisy or partial receipts, OCR confidence drops significantly. These are surfaced via the `warnings` array in each JSON and are excluded from the financial summary.

---

## 5. Possible Improvements

| Area | Improvement |
|---|---|
| OCR accuracy | Fine-tune Tesseract on a receipt-specific dataset or switch to EasyOCR for printed receipts |
| Store name | Use a dictionary of known store names to validate / correct fuzzy matches |
| Items | Use line-height clustering to better reconstruct tabular structure |
| Confidence | Train a lightweight classifier on field patterns rather than rule-based scoring |
| Edge cases | Add PDF support via pdf2image for digital receipts |
| UI | Build a simple web interface (Flask/Streamlit) to upload receipts and view results |

---

## 6. Running the Project

### Installation

```bash
# 1. Install Tesseract (system-level)
#    Ubuntu/Debian:
sudo apt-get install tesseract-ocr

#    macOS:
brew install tesseract

#    Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install Python dependencies
pip install -r requirements.txt
```

### Running

```bash
# Demo mode (no images needed — uses synthetic receipt)
python demo.py

# Full pipeline (add your images to receipts/ first)
python pipeline.py
```

### Output Files

```
outputs/
  json/
    receipt_001.json     ← structured extraction per receipt
    receipt_002.json
    ...
  summary/
    financial_summary.json   ← aggregate spend report
```

---

## 7. Output Format Reference

### Per-Receipt JSON

```json
{
  "source_file": "receipt_001.jpg",
  "store_name": {
    "value": "RELIANCE FRESH",
    "confidence": 0.91
  },
  "date": {
    "value": "15/03/2024",
    "confidence": 0.95
  },
  "items": [
    {
      "name": { "value": "Bread Brown 400g", "confidence": 0.87 },
      "price": { "value": "45.00", "confidence": 0.87 }
    }
  ],
  "total_amount": {
    "value": "276.15",
    "confidence": 0.96
  },
  "ocr_page_confidence": 0.88,
  "warnings": []
}
```

### Financial Summary JSON

```json
{
  "num_transactions": 10,
  "receipts_included_in_total": 9,
  "skipped_receipts": ["blurry_receipt.jpg"],
  "total_spend": "₹ 4,231.50",
  "spend_per_store": {
    "RELIANCE FRESH": "₹ 1,450.00",
    "BIG BAZAAR": "₹ 2,781.50"
  }
}
```

---


