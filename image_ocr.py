import os
import easyocr
import numpy as np
import warnings

warnings.filterwarnings('ignore') # to hide verbose warnings from torch

# Global variable for lazy loading
_ocr_instance = None

def get_ocr_instance():
    global _ocr_instance
    if _ocr_instance is None:
        print("[Image] Loading EasyOCR Engine (Lazy Load)...")
        # Initialize easyocr, use CPU by default to save RAM alongside Phi-3
        _ocr_instance = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_instance

def extract_text_from_image(image_path):
    """
    Extract text and metadata from image using EasyOCR.
    """
    if not os.path.exists(image_path):
        return "", {"error": "File not found"}

    try:
        reader = get_ocr_instance()
        # Returns a list of tuples: (bounding_box, text, confidence)
        result = reader.readtext(image_path)
        
        if not result:
            return "", {"num_text_regions": 0, "mean_confidence": 0.0}

        extracted_text = []
        confidences = []
        text_regions = []

        for item in result:
            bbox, text, confidence = item
            extracted_text.append(text)
            confidences.append(float(confidence))
            text_regions.append(bbox)

        text_output = " ".join(extracted_text)

        metadata = {
            "num_text_regions": len(text_regions),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "status": "success"
        }

        return text_output, metadata

    except Exception as e:
        return "", {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"')
    text, meta = extract_text_from_image(image_path)
    if "error" in meta:
        print(f"[Error] {meta['error']}")
    else:
        print("\n[Image] Extracted Text:")
        print(text if text else "[No text detected]")
        print("\n[OCR] Metadata:")
        print(meta)