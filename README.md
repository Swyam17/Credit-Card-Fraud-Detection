# Credit Card Fraud Detection Project

## Overview
This project uses the **Credit Card Fraud Detection** dataset from Kaggle to detect fraudulent transactions. We build a machine learning pipeline in Python that explores the data, identifies fraudulent patterns, and builds a classification model to predict potential frauds.

## Dataset
The dataset (creditcard.csv) contains:
- `Time`: Seconds elapsed between each transaction and the first transaction.
- `V1-V28`: PCA-transformed features for privacy.
- `Amount`: Transaction amount.
- `Class`: 1 for Fraud, 0 for Normal.

**Dataset URL**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Full Project: Secure Bank Portal
The project now includes two distinct components:
1.  **AI Engine (`fraud_detection.py`)**: Analyzes the dataset and trains the machine learning model.
2.  **User Portal (`app.py`)**: A professional GUI where users enter their account details, PIΝ, and transaction amount for real-time fraud checking.

## How to Run in VS Code

### 1. Preparation
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project folder.

### 2. Train the AI Model
First, run the training script to generate the model files (`fraud_model.joblib` and `scaler.joblib`):
```bash
python fraud_detection.py
```

### 3. Launch the Bank Portal
Once the model is trained, launch the interactive user interface:
```bash
python app.py
```

## User Portal Features
- **Account Verification**: Mock database with accounts (e.g., `1234567890` with PIN `1234`).
- **Security PIN**: Standard 4-digit PIN authentication.
- **AI Prediction**: Uses a Random Forest model with SMOTE-balanced data to detect risky transactions.
- **Modern Interface**: Clean, professional Tkinter-based GUI for Windows.

## Project Structure
- `app.py`: The GUI application (Bank Portal).
- `fraud_detection.py`: The machine learning training script.
- `credit_card_fraud_detection.ipynb`: Notebook version for Google Colab researchers.
- `README.md`: Project documentation and setup instructions.
- `requirements.txt`: Necessary libraries list (joblib, pandas, etc.).

## Key Goals Addressed
- **Fraud Analysis**: Calculates how many fraudulent entries are in the dataset and their percentage of the total.
- **Predictive Modeling**: Uses SMOTE (for imbalanced data) and Random Forest to predict potential fraud.
