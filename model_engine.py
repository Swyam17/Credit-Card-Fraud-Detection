"""
model_engine.py
Robust Machine Learning training and inference engine for Credit Card Fraud Detection.
Uses Balanced Random Forest with auto-scaling and fallback oversampling.
Exports model, scaler, and rich evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Optional SMOTE with graceful fallback to minority oversampling
def balance_training_data(X_train, y_train, random_state=42):
    """Balances dataset using SMOTE if installed, otherwise uses minority oversampling."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print("[*] Resampled data using SMOTE.")
        return X_res, y_res
    except ImportError:
        # Fallback: Robust Minority Oversampling
        print("[*] imblearn not found. Applying built-in minority oversampling...")
        X_train_df = pd.DataFrame(X_train)
        y_train_s = pd.Series(y_train, name='Class')
        
        fraud_idx = y_train_s[y_train_s == 1].index
        normal_idx = y_train_s[y_train_s == 0].index
        
        # Target ratio: 1:3 for balanced stability without overfitting
        target_fraud_count = len(normal_idx) // 2
        oversampled_fraud_idx = np.random.choice(fraud_idx, size=target_fraud_count, replace=True)
        
        combined_idx = np.concatenate([normal_idx, oversampled_fraud_idx])
        np.random.shuffle(combined_idx)
        
        X_res = X_train_df.loc[combined_idx].values
        y_res = y_train_s.loc[combined_idx].values
        return X_res, y_res

def train_fraud_model(data_path="creditcard.csv", output_dir="."):
    """Trains the fraud classification model and saves artifacts."""
    print(f"[*] Loading dataset from '{data_path}'...")
    if not os.path.exists(data_path):
        from generate_dataset import save_datasets
        save_datasets()
        
    df = pd.read_csv(data_path)
    print(f"[*] Dataset loaded: {len(df)} records, {df['Class'].sum()} fraudulent.")
    
    # 1. Scaling Amount and Time
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[['Amount', 'Time']])
    
    # Prepare feature matrix X (V1 to V28, Scaled_Amount, Scaled_Time)
    v_cols = [f'V{i}' for i in range(1, 29)]
    X_v = df[v_cols].values
    X = np.hstack([X_v, scaled_feats])
    y = df['Class'].values
    
    # 2. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Balancing
    X_train_res, y_train_res = balance_training_data(X_train, y_train)
    
    # 4. Train Model
    print("[*] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("-" * 45)
    print("MODEL EVALUATION RESULTS:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix: {cm}")
    print("-" * 45)
    
    # Feature Importances (Top 10)
    feat_names = v_cols + ['Scaled_Amount', 'Scaled_Time']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [{"feature": feat_names[i], "importance": round(float(importances[i]), 4)} for i in sorted_idx[:10]]
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "total_test_samples": len(y_test),
        "test_fraud_count": int(np.sum(y_test)),
        "top_features": top_features,
        "dataset_total": len(df),
        "dataset_fraud": int(df['Class'].sum()),
        "dataset_normal": int(len(df) - df['Class'].sum()),
        "fraud_percentage": round(float(df['Class'].sum() / len(df) * 100), 2)
    }
    
    # Save Model, Scaler, and Metrics
    target_dirs = [output_dir]
    if os.path.exists("Credit-Card-Fraud-Detection-main") and output_dir != "Credit-Card-Fraud-Detection-main":
        target_dirs.append("Credit-Card-Fraud-Detection-main")
        
    for out in target_dirs:
        model_path = os.path.join(out, "fraud_model.joblib")
        scaler_path = os.path.join(out, "scaler.joblib")
        metrics_path = os.path.join(out, "model_metrics.json")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"[OK] Model saved to '{model_path}'")
        print(f"[OK] Scaler saved to '{scaler_path}'")
        print(f"[OK] Metrics saved to '{metrics_path}'")
        
    # Also save plots if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Confusion Matrix Plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=120)
        plt.close()
        
        # 2. Fraud Distribution Plot
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=df, palette=['#10b981', '#ef4444'])
        plt.title('Distribution of Fraud vs. Normal Transactions')
        plt.xticks([0, 1], ['Legitimate (0)', 'Fraudulent (1)'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fraud_distribution.png'), dpi=120)
        plt.close()
        print("[OK] Evaluation plots generated.")
    except Exception as e:
        print(f"[!] Plot generation note: {e}")
        
    return model, scaler, metrics

# Singleton predictor helper
_CACHED_MODEL = None
_CACHED_SCALER = None
_CACHED_METRICS = None

def get_engine():
    """Returns initialized model, scaler, and metrics, loading or training if necessary."""
    global _CACHED_MODEL, _CACHED_SCALER, _CACHED_METRICS
    if _CACHED_MODEL is not None and _CACHED_SCALER is not None:
        return _CACHED_MODEL, _CACHED_SCALER, _CACHED_METRICS
        
    model_path = "fraud_model.joblib"
    scaler_path = "scaler.joblib"
    metrics_path = "model_metrics.json"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[*] Model files not found. Auto-training now...")
        _CACHED_MODEL, _CACHED_SCALER, _CACHED_METRICS = train_fraud_model()
    else:
        try:
            _CACHED_MODEL = joblib.load(model_path)
            _CACHED_SCALER = joblib.load(scaler_path)
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    _CACHED_METRICS = json.load(f)
            else:
                _CACHED_METRICS = {}
            print("[OK] Cached model and scaler loaded.")
        except Exception as e:
            print(f"[!] Error loading existing model: {e}. Retraining...")
            _CACHED_MODEL, _CACHED_SCALER, _CACHED_METRICS = train_fraud_model()
            
    return _CACHED_MODEL, _CACHED_SCALER, _CACHED_METRICS

def predict_transaction(amount, hour=12, merchant_cat="Grocery", distance=10.0, is_foreign=0, failed_pin=0, device_trust=90, pca_vector=None):
    """
    Computes fraud probability, risk score, classification, and explanation factors.
    Works with both rich transactional inputs and standard PCA vectors.
    """
    model, scaler, _ = get_engine()
    
    # If no PCA vector is provided, estimate synthetic PCA features from the transaction risk factors
    if pca_vector is None or len(pca_vector) != 28:
        # Heuristic risk alignment for synthetic PCA vector
        risk_factor = 0.0
        if amount > 1000:
            risk_factor += min(2.5, (amount - 1000) / 1000.0)
        if hour in [1, 2, 3, 4]:
            risk_factor += 1.5
        if merchant_cat in ["Crypto & Trading", "Luxury & Jewelry"]:
            risk_factor += 1.8
        elif merchant_cat in ["Electronics", "Travel & Airlines"]:
            risk_factor += 0.8
        if distance > 200:
            risk_factor += min(2.0, distance / 500.0)
        if is_foreign:
            risk_factor += 1.5
        if failed_pin > 0:
            risk_factor += failed_pin * 1.2
        if device_trust < 50:
            risk_factor += (50 - device_trust) / 25.0
            
        # Scale PCA components according to risk factor
        base_v = np.random.normal(0, 0.5, 28)
        # Shift known predictive components
        base_v[0] -= risk_factor * 0.8   # V1
        base_v[2] -= risk_factor * 1.0   # V3
        base_v[3] += risk_factor * 0.9   # V4
        base_v[9] -= risk_factor * 1.1   # V10
        base_v[10] += risk_factor * 0.9  # V11
        base_v[11] -= risk_factor * 1.2  # V12
        base_v[13] -= risk_factor * 1.4  # V14
        base_v[16] -= risk_factor * 1.3  # V17
        pca_vector = base_v.tolist()
        
    # Scale Amount and Time (Hour * 3600)
    time_sec = float(hour * 3600)
    scaled_vals = scaler.transform([[float(amount), time_sec]])[0]
    scaled_amount, scaled_time = scaled_vals[0], scaled_vals[1]
    
    # Feature vector: 28 PCA + Scaled_Amount + Scaled_Time = 30 features
    features = np.array(pca_vector + [scaled_amount, scaled_time]).reshape(1, -1)
    
    prob = float(model.predict_proba(features)[0][1])
    is_fraud = int(prob >= 0.50)
    
    # Determine Risk Level
    if prob >= 0.75:
        risk_level = "CRITICAL"
        risk_color = "#ef4444"
    elif prob >= 0.50:
        risk_level = "HIGH"
        risk_color = "#f97316"
    elif prob >= 0.25:
        risk_level = "MEDIUM"
        risk_color = "#eab308"
    else:
        risk_level = "LOW"
        risk_color = "#10b981"
        
    # Generate human-readable risk explanation reasons
    risk_reasons = []
    if amount > 1500:
        risk_reasons.append(f"High transaction amount (${amount:,.2f}) deviates significantly from standard retail baseline.")
    if hour in [0, 1, 2, 3, 4, 5]:
        risk_reasons.append(f"Transaction conducted at {hour:02d}:00 (unusual nighttime hours for cardholder).")
    if merchant_cat in ["Crypto & Trading", "Luxury & Jewelry"]:
        risk_reasons.append(f"High-risk merchant industry category: '{merchant_cat}'.")
    if distance > 250:
        risk_reasons.append(f"Unusual location anomaly: {distance:,.0f} km away from billing address.")
    if is_foreign:
        risk_reasons.append("Cross-border / international transaction detected.")
    if failed_pin > 0:
        risk_reasons.append(f"Security Alert: {failed_pin} failed PIN attempt(s) recorded prior to transaction.")
    if device_trust < 50:
        risk_reasons.append(f"Untrusted device profile (Trust Score: {device_trust}/100).")
        
    if not risk_reasons:
        risk_reasons.append("Transaction matches verified cardholder spending pattern, normal geolocation, and trusted device telemetry.")
        
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob, 4),
        "fraud_percentage": round(prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_reasons": risk_reasons,
        "recommended_action": "BLOCK_TRANSACTION" if prob >= 0.75 else ("CHALLENGE_OTP" if prob >= 0.50 else "APPROVE")
    }

if __name__ == "__main__":
    train_fraud_model()
