import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os
import joblib

# Confirm script started
with open('script_status.txt', 'w') as f:
    f.write('Python script has started and is importing libraries...')

# Set visual style
sns.set_style("whitegrid")

def load_data(file_path='creditcard.csv'):
    """Load the dataset or create a synthetic one if missing."""
    if not os.path.exists(file_path):
        print(f"\n[!] Dataset {file_path} not found.")
        print("[!] Generating synthetic data for demonstration purposes...")
        
        # Creating a basic version of the credit card dataset structure
        # (V1-V28 PCA features, Amount, Time, and Class)
        # Reduced synthetic data generation to 1,000 rows for speed
        np.random.seed(42)
        rows = 1000
        data = {
            'Time': np.linspace(0, 172800, rows), # 48 hours in seconds
            'Amount': np.random.uniform(1, 1000, rows),
            **{f'V{i}': np.random.randn(rows) for i in range(1, 29)},
            'Class': np.concatenate([np.zeros(980), np.ones(20)]) # 2% fraud
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"[*] Synthetic dataset '{file_path}' created with {len(df)} records.\n")
    
    df = pd.read_csv(file_path)
    
    # Ensure the dataset is not too large for this demo
    if len(df) > 1000:
        print(f"[*] Trimming large dataset from {len(df)} to 1,000 rows...")
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
        
    print(f"Dataset loaded with {len(df)} records.")
    return df

def analyze_frauds(df):
    """Analyze and print the count and percentage of fraudulent transactions."""
    class_counts = df['Class'].value_counts()
    normal = class_counts[0]
    fraud = class_counts[1]
    perc = (fraud / len(df)) * 100
    
    print("-" * 30)
    print(f"Total Transactions: {len(df)}")
    print(f"Normal Transactions: {normal}")
    print(f"Fraudulent Transactions: {fraud}")
    print(f"Fraud Percentage: {perc:.4f}%")
    print("-" * 30)
    
    # Simple Plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Distribution of Fraud vs. Normal Transactions')
    plt.savefig('fraud_distribution.png')
    print("Saved plot to 'fraud_distribution.png'")

def preprocess_data(df):
    """Normalize numeric columns and prepare features/labels."""
    scaler = StandardScaler()
    # Fit and transform Amount and Time together as a 2D array
    scaled_features = scaler.fit_transform(df[['Amount', 'Time']])
    df['Scaled_Amount'] = scaled_features[:, 0]
    df['Scaled_Time'] = scaled_features[:, 1]
    
    # Drop original Time and Amount columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y, scaler

def train_and_evaluate(X, y):
    """Handle imbalance with SMOTE and train a Random Forest classifier."""
    print("Splitting data and handling imbalance with SMOTE...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Training Random Forest Classifier (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'")
    
    # Save the model and scaler
    joblib.dump(model, 'fraud_model.joblib')
    print("Model saved to 'fraud_model.joblib'")
    
    return model

def main():
    df = load_data()
    if df is not None:
        analyze_frauds(df)
        X, y, scaler = preprocess_data(df)
        
        # Save the actual fitted scaler
        joblib.dump(scaler, 'scaler.joblib')
        print("[*] Fitted scaler saved to 'scaler.joblib'")
        
        model = train_and_evaluate(X, y)
        print("\nProject Analysis and Prediction Model Complete.")

if __name__ == "__main__":
    main()
