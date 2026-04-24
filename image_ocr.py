import os
import easyocr
import numpy as np
import warnings
import cv2

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

def enhance_image_for_hidden_text(image_path):
    """
    Advanced Computer Vision techniques to reveal:
    1. White text on white background (CLAHE contrast amplification)
    2. Extremely small text (2.5x Cubic Upscaling)
    3. Hidden steganography-like text (Adaptive Histogram Eq)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 1. Upscale by 2.5x for tiny micro-text detection
        h, w = img.shape[:2]
        # Limit scaling for extremely large images to prevent out-of-memory errors
        scale = 1.5 if max(h, w) > 2000 else 2.5
            
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply extreme CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This forces contrast differences between pixels that are mathematically close 
        # (e.g., #FFFFFF vs #FDFDFD), exposing "invisible" white text.
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. Sharpen image to make edges (text boundaries) pop out vividly
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        temp_path = image_path + "_cv2_enhanced.png"
        cv2.imwrite(temp_path, sharpened)
        return temp_path
    except Exception as e:
        print(f"[CV Error] Preprocessing failed: {e}")
        return None

def extract_text_from_image(image_path):
    """
    Extract text and metadata from image using EasyOCR, doing a multi-pass
    approach with Computer Vision enhancements to detect hidden/tiny text.
    """
    if not os.path.exists(image_path):
        return "", {"error": "File not found"}

    try:
        reader = get_ocr_instance()
        extracted_text_blocks = []
        all_confidences = []
        all_regions = []

        # -- PASS 1: Standard extraction for obvious text --
        # Lowered text_threshold to catch fainter text natively
        result_std = reader.readtext(image_path, text_threshold=0.4, low_text=0.3)
        
        for item in result_std:
            bbox, text, confidence = item
            extracted_text_blocks.append(text)
            all_confidences.append(float(confidence))
            all_regions.append(bbox)

        # -- PASS 2: CV Enhancement for Hidden / Micro-Text --
        enhanced_path = enhance_image_for_hidden_text(image_path)
        
        if enhanced_path and os.path.exists(enhanced_path):
            # Use extreme sensitivity on the enhanced image
            result_enh = reader.readtext(enhanced_path, text_threshold=0.1, low_text=0.2)
            
            for item in result_enh:
                bbox, text, confidence = item
                # Append only if this hidden text wasn't already caught perfectly in pass 1
                if text.strip() not in " ".join(extracted_text_blocks):
                    extracted_text_blocks.append(text)
                    all_confidences.append(float(confidence))
            
            # Clean up temporary enhanced image
            try:
                os.remove(enhanced_path)
            except:
                pass

        if not extracted_text_blocks:
            return "", {"num_text_regions": 0, "mean_confidence": 0.0}

        text_output = " ".join(extracted_text_blocks)

        metadata = {
            "num_text_regions": len(all_regions) + (len(extracted_text_blocks) - len(result_std)),
            "mean_confidence": float(np.mean(all_confidences)) if all_confidences else 0.0,
            "min_confidence": float(np.min(all_confidences)) if all_confidences else 0.0,
            "status": "success",
            "hidden_text_scan": "enabled"
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