"""
fraud_detection.py
AI Engine for Credit Card Fraud Detection.
Analyzes the custom dataset, handles class imbalance, trains the Random Forest model,
evaluates precision, recall, ROC-AUC, and exports production model artifacts.
"""

import os
import sys
from model_engine import train_fraud_model

def main():
    print("=" * 60)
    print("   CREDIT CARD FRAUD DETECTION - AI TRAINING ENGINE")
    print("=" * 60)
    
    # Check dataset existence
    if not os.path.exists('creditcard.csv'):
        print("[*] Dataset 'creditcard.csv' not found. Generating custom dataset first...")
        import generate_dataset
        generate_dataset.save_datasets()
        
    # Execute training
    model, scaler, metrics = train_fraud_model(data_path='creditcard.csv', output_dir='.')
    
    print("\n[SUCCESS] AI Training Pipeline completed successfully!")
    print(f"Model Accuracy: {metrics['accuracy'] * 100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Model files generated: fraud_model.joblib, scaler.joblib, model_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
