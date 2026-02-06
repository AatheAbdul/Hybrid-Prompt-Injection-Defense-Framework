from cod_prefilter import run_cod_llm
from normalization import normalize_input
import joblib
import numpy as np

class InjectionPipeline:
    def __init__(self):
        self.clf = joblib.load("models/xgboost_injection_classifier.pkl")
        self.embedder = joblib.load("models/embedder.pkl")
        print("🚀 Hybrid Pipeline: CoD → XGBoost → [CoT if needed]")
    
    def predict(self, prompt: str):
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing: {prompt[:80]}...")
        
        # STEP 1: CoD (1-2 sec)
        print("1️⃣  CoD Pre-filter running...")
        clean_prompt = normalize_input(prompt)
        draft_analysis, draft_label = run_cod_llm(clean_prompt)
        print(f"   CoD: {draft_label} | {draft_analysis[:60]}...")
        
        if draft_label == "SUSPICIOUS":
            print("   🚫 BLOCKED BY COD → NO CoT NEEDED")
            return {"decision": "🚫 BLOCKED", "stage": "CoD", "cot_needed": False}
        
        # STEP 2: XGBoost (10ms)
        print("2️⃣  XGBoost Classifier running...")
        feature_text = f"{prompt} [CoD: {draft_analysis}]"
        embedding = self.embedder.encode([feature_text])
        score = self.clf.predict_proba(embedding)[0][1]
        print(f"   XGBoost: {score:.1%} injection risk")
        
        if score > 0.65:
            print("   🚫 BLOCKED BY XGBOOST → NO CoT NEEDED")
            return {"decision": "🚫 BLOCKED", "stage": "XGBoost", "cot_needed": False}
        
        # STEP 3: NEEDS CoT (expensive)
        print("3️⃣  ✅ SAFE → PASSING TO Fine-grained CoT Detector")
        print("   💡 (In production: Meticulous Thought Defender CoT)")
        return {"decision": "⚠️  CoT NEEDED", "stage": "CoT", "cot_needed": True}

# INTERACTIVE LOOP
if __name__ == "__main__":
    pipeline = InjectionPipeline()
    
    print("\n🎮 INTERACTIVE PIPELINE TEST")
    print("Type prompts (Ctrl+C to exit)")
    
    try:
        while True:
            prompt = input("\n💬 Enter prompt: ").strip()
            if not prompt: continue
            
            result = pipeline.predict(prompt)
            print(f"\n🎯 FINAL: {result['decision']} ({result['stage']})")
            print(f"   {'→ CoT Detector' if result['cot_needed'] else '→ Main LLM'}")
            
    except KeyboardInterrupt:
        print("\n👋 Pipeline test complete!")
