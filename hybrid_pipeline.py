import os
from cot_testing import MeticulousCoTDetector
from cod_prefilter import run_cod_llm
import joblib
from normalization import normalize_input
from image_ocr import extract_text_from_image
from speech_to_text import transcribe_audio

class FullHybridPipeline:
    def __init__(self):
        print("[Core] Loading Cloud-Accelerated Reasoning Engines...")
        self.cot_detector = MeticulousCoTDetector()
        
        print("[Core] Loading ML Classifiers...")
        self.clf = joblib.load("models/xgboost_injection_classifier.pkl")
        self.embedder = joblib.load("models/embedder.pkl")
        
        print("[System] Hybrid Defense Core READY!")
        print("[Info] OCR and STT will load only when needed (Lazy Loading).")
    
    def predict_multimodal(self, input_data: str, client, model_name="gemini-1.5-flash"):
        """
        Detects if input is a text string or a file path (image/audio).
        Extracts content and runs prediction via Gemini API.
        """
        # 1. Detect if input is a local file
        if os.path.exists(input_data):
            ext = os.path.splitext(input_data)[1].lower()
            
            # --- IMAGE ROUTE ---
            if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                print(f"[Image] Detected Image File. Running OCR...")
                text, meta = extract_text_from_image(input_data)
                if not text:
                    return "⚠️ FAILED", "OCR-Empty", 0.0, ""
                print(f"[Image] Extracted from Image: '{text[:100]}...'")
                return self.predict(text, client, model_name)

            # --- AUDIO ROUTE ---
            elif ext in [".mp3", ".wav", ".m4a", ".flac"]:
                print(f"[Audio] Detected Audio File. Running Transcription...")
                text, meta = transcribe_audio(input_data)
                if not text:
                    return "⚠️ FAILED", "STT-Empty", 0.0, ""
                print(f"[Audio] Transcribed Audio: '{text[:100]}...'")
                return self.predict(text, client, model_name)

        # --- DEFAULT TEXT ROUTE ---
        return self.predict(input_data, client, model_name)
    
    def predict(self, prompt: str, client, model_name="gemini-1.5-flash"):
        print(f"\n[Search] Original: '{prompt[:60]}...'")
        
        # 🧹 Normalize Input First
        prompt = normalize_input(prompt)
        print(f"[Clean] Full Normalized Prompt: '{prompt}'")
        
        # 1️⃣ CoD (Tier 1: Early Identification using Cloud LLM)
        analysis, cod_label, cod_risk = run_cod_llm(prompt, client=client, model_name=model_name)
        print(f"   [CoD Tier 1] Screening: {cod_risk*100:.0f}% Risk ({cod_label})")
        
        # New Strict Logic:
        # A. If risk < 25% and SAFE -> Approve
        if cod_risk < 0.25 and cod_label == "SAFE":
            print(f"   [Pass] Low Risk Match (<0.25) + SAFE -> APPROVED (CoD-FastPath)")
            return "✅ APPROVED", "CoD-FastPath", 1-cod_risk, prompt

        # B. If risk > 75% and SUSPICIOUS -> Block
        if cod_risk > 0.75 and cod_label == "SUSPICIOUS":
            print(f"   [Block] High Risk Match (>0.75) + SUSPICIOUS -> BLOCKED (CoD-Strict)")
            return "🚫 BLOCKED", "CoD-Strict", cod_risk, prompt
            
        # C. Everything else (0.25 - 0.75) -> XGBoost
        print(f"   [Wait] CoD Uncertain (Risk {cod_risk*100:.0f}%) -> Escalating to XGBoost")
        
        # 2️⃣ XGBoost (Tier 2: Statistical Verification)
        # Taking CoD risk score into account for feature engineering
        feature = f"{prompt} [CoD-Risk: {cod_risk:.2f}]"
        emb = self.embedder.encode([feature])
        xgb_risk = self.clf.predict_proba(emb)[0][1]
        xgb_pred = self.clf.predict(emb)[0] # 0 = Safe, 1 = Injection
        
        # New Logic:
        # A. If XGBoost classifies as INJECTION -> Block (any risk %)
        if xgb_pred == 1:
            print(f"   [Block] XGBoost Classified as INJECTION ({xgb_risk:.2f}) -> BLOCKED")
            return "🚫 BLOCKED", "XGBoost-Injection", xgb_risk, prompt

        # B. If XGBoost classifies as SAFE but risk > 0.85 -> Escalate to CoT
        if xgb_pred == 0 and xgb_risk > 0.85:
            print(f"   [Alert] XGBoost SAFE but High Risk ({xgb_risk:.2f}) -> Escalating to CoT")
            
            # 3️⃣ CoT (Tier 3: Deep Audit)
            cot = self.cot_detector.detect_injection_cot(prompt, client=client, model_name=model_name)
            print(f"   [CoT Tier 3] Verdict: {cot['verdict']}")
            return ("🚫 BLOCKED" if cot['verdict'] == "INJECTION" else "✅ APPROVED", 
                   f"CoT-{cot['verdict']}", cot['confidence'], prompt)
        
        # C. If XGBoost classifies as SAFE and risk <= 0.85 -> Approve
        print(f"   [Pass] XGBoost Classified as SAFE and Risk is manageable ({xgb_risk:.2f}) -> APPROVED")
        return "✅ APPROVED", "XGBoost-Safe", 1-xgb_risk, prompt

# TEST
if __name__ == "__main__":
    try:
        pipeline = FullHybridPipeline()
    except Exception as e:
        print(f"\n❌ FATAL: Failed to initialize pipeline components.")
        print(f"Error: {e}")
        print("\nPossible reasons:")
        print("1. Out of RAM (Need ~8GB+)")
        print("2. Models failed to download")
        print("3. Missing model files in 'models/' folder")
        os._exit(1)
    
    print("\n" + "="*50)
    print("Hybrid Prompt Detection Pipeline!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            line = input("\nEnter prompt or file path: ")
            if line.strip().lower() in ['exit', 'quit']:
                print("Exiting pipeline...")
                break
            
            # Clean input (remove quotes from file paths)
            user_input = line.strip().strip('"')
            
            if not user_input:
                continue
            
            # Note for CLI Usage: A Gemini Client needs to be provided here.
            # We are assuming runtime is entirely from Streamlit now.
            print("CLI Direct Testing is disabled while configured for Cloud Mode.")
            print("Please use the Streamlit Web Interface (app.py) to test!")
            break
        except KeyboardInterrupt:
            print("\nExiting pipeline...")
            break
        except Exception as e:
            print(f"\nError processing prompt: {e}\n")
