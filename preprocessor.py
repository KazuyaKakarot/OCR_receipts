"""
preprocessor.py
---------------
Prepares receipt images for OCR by fixing common real-world issues:
  - Noise and blur      → Gaussian blur + thresholding
  - Skew / rotation     → Deskew using projection profiling
  - Contrast issues     → CLAHE (adaptive histogram equalization)

Returns a preprocessed grayscale image (numpy array) ready for Tesseract.
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load image from disk. Handles both color and grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def fix_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This brightens dark receipts and evens out lighting gradients.
    Think of it as 'auto-leveling' the image locally, not globally.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """
    Apply a gentle Gaussian blur to reduce noise/grain.
    Kernel size (3,3) is small so we don't blur actual text.
    """
    return cv2.GaussianBlur(gray, (3, 3), 0)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Convert to black-and-white using Otsu's thresholding.
    Tesseract works best on clean B&W images.
    Otsu automatically picks the best threshold value.
    """
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Detect and correct rotation in scanned/photographed receipts.
    Uses the angle of detected text lines to rotate the image back to level.

    How it works:
      1. Find all white pixels (text pixels after binarization)
      2. Fit a bounding rectangle around them → gives us the skew angle
      3. Rotate the whole image to correct that angle
    """
    # Find coordinates of all non-zero (text) pixels
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return binary  # Not enough content to deskew

    # minAreaRect returns the angle of the bounding box
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect returns angles in (-90, 0]; convert to a usable rotation
    if angle < -45:
        angle = 90 + angle  # e.g., -80° becomes +10°

    # Only rotate if skew is significant (> 0.5°) to avoid unnecessary blur
    if abs(angle) < 0.5:
        return binary

    # Build rotation matrix and apply it
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def upscale_if_small(img: np.ndarray, min_width: int = 1000) -> np.ndarray:
    """
    Tesseract accuracy improves significantly on larger images.
    If the image is narrower than min_width pixels, scale it up 2×.
    """
    h, w = img.shape[:2]
    if w < min_width:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    return img


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Master preprocessing function. Call this with any receipt image path.
    Returns a clean binary image ready for OCR.

    Pipeline:
        Load → Grayscale → Fix contrast → Upscale → Denoise → Binarize → Deskew
    """
    img = load_image(image_path)
    gray = to_grayscale(img)
    gray = fix_contrast(gray)
    gray = upscale_if_small(gray)
    gray = remove_noise(gray)
    binary = binarize(gray)
    binary = deskew(binary)
    return binary
