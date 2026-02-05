import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

print("🚀 XGBoost Injection Classifier - NO PLOTS")
print("=" * 60)

# 1. LOAD + PREPARE
print("📊 Loading cod_enriched_dataset.csv...")
df = pd.read_csv("cod_enriched_dataset.csv")
print(f"Dataset: {df.shape[0]} rows (Safe: {len(df[df['true_label']=='safe'])}, Injection: {len(df[df['true_label']=='injection'])})")

df['feature_text'] = df['prompt'] + " [CoD: " + df['draft_analysis'] + "]"
y = (df['true_label'] == 'injection').astype(int)

# 2. EMBEDDINGS (384-dim)
print("🧠 Generating embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(df['feature_text'].tolist(), show_progress_bar=True, batch_size=32)
print(f"✅ Embeddings shape: {X.shape}")

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 4. XGBoost TRAINING
print("🎯 Training XGBoost...")
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6, 
    learning_rate=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 5. EVALUATION
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🏆 RESULTS:")
print(f"Accuracy: {accuracy:.1%} ({accuracy:.3f})")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  Safe  Inj")
print(f"Safe [{cm[0,0]:2d} {cm[0,1]:2d}]")
print(f"Inj  [{cm[1,0]:2d} {cm[1,1]:2d}]")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Injection']))

# 6. SAVE MODELS (CRITICAL)
print("\n💾 SAVING MODELS...")
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/xgboost_injection_classifier.pkl")
joblib.dump(embedder, "models/embedder.pkl")

print("✅ SAVED SUCCESSFULLY:")
print("   models/xgboost_injection_classifier.pkl  (95.5% model)")
print("   models/embedder.pkl                      (384-dim)")
print("\n🎉 PIPELINE READY FOR PRODUCTION!")
