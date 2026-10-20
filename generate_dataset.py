"""
generate_dataset.py
Generates a realistic, high-quality, self-contained Credit Card Fraud Detection dataset
with authentic statistical distributions, real-world transaction indicators,
and standard PCA features (V1-V28) for full compatibility.
"""

import numpy as np
import pandas as pd
import os
import random

def generate_credit_card_data(n_samples=5000, fraud_ratio=0.03, random_seed=42):
    """
    Generates synthetic credit card transaction data.
    
    Parameters:
    - n_samples (int): Total number of transactions (default 5,000).
    - fraud_ratio (float): Fraction of fraudulent transactions (default 0.03 = 3%).
    - random_seed (int): Seed for reproducibility.
    
    Returns:
    - df_standard (pd.DataFrame): Compatible with Kaggle creditcard.csv (Time, V1-V28, Amount, Class).
    - df_rich (pd.DataFrame): Human-interpretable dataset with rich transaction attributes.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    print(f"[*] Generating {n_samples} transactions: {n_normal} legitimate, {n_fraud} fraudulent ({fraud_ratio*100:.1f}%)...")
    
    # 1. Accounts Pool
    accounts = [
        {"acc": "1234567890", "name": "John Doe", "home_city": "New York", "home_country": "USA"},
        {"acc": "0987654321", "name": "Jane Smith", "home_city": "Los Angeles", "home_country": "USA"},
        {"acc": "5556667770", "name": "Alice Wilson", "home_city": "Chicago", "home_country": "USA"},
        {"acc": "1112223334", "name": "Bob Brown", "home_city": "Houston", "home_country": "USA"},
        {"acc": "9998887776", "name": "Charlie Davis", "home_city": "Miami", "home_country": "USA"},
        {"acc": "4443332221", "name": "Elena Rostova", "home_city": "Seattle", "home_country": "USA"},
        {"acc": "7778889990", "name": "David Miller", "home_city": "Denver", "home_country": "USA"},
        {"acc": "3332221110", "name": "Sarah Connor", "home_city": "San Francisco", "home_country": "USA"}
    ]
    
    merchant_categories = [
        "Grocery", "Dining & Restaurants", "Gas & Fuel", "Online Retail", 
        "Electronics", "Travel & Airlines", "Luxury & Jewelry", "Crypto & Trading", "Entertainment"
    ]
    
    transaction_types = ["Chip_POS", "Contactless_NFC", "Online_CNP", "ATM_Withdrawal"]
    
    records_rich = []
    
    # Generate Normal Transactions
    for i in range(n_normal):
        acc = random.choice(accounts)
        # Normal amounts follow log-normal: typically $10 - $150, occasionally up to $800
        amount = round(float(np.random.lognormal(mean=3.6, sigma=0.85)), 2)
        amount = max(2.50, min(amount, 1200.00))
        
        # Normal hour: mostly daytime 7 AM to 11 PM
        hour_weights = [1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 11, 12, 12, 11, 10, 10, 10, 11, 12, 10, 8, 6, 4, 2]
        hour = random.choices(range(24), weights=hour_weights)[0]
        
        # Normal category: mostly everyday essentials
        cat_weights = [25, 20, 18, 15, 8, 5, 2, 1, 6]
        category = random.choices(merchant_categories, weights=cat_weights)[0]
        
        # Distance: mostly local (< 25km)
        distance = round(float(np.random.exponential(scale=12.0)), 1)
        is_foreign = 1 if (random.random() < 0.02 and category in ["Travel & Airlines", "Online Retail"]) else 0
        
        # Failed PINs: almost always 0, rarely 1
        failed_pins = 1 if random.random() < 0.03 else 0
        
        # Transaction Type
        tx_type = "Online_CNP" if category in ["Online Retail", "Crypto & Trading"] else random.choice(transaction_types)
        
        # Device Trust: High for normal users
        device_trust = int(np.clip(np.random.normal(92, 6), 50, 100))
        
        # Time in seconds (over 48 hour span)
        time_sec = float(hour * 3600 + np.random.uniform(0, 3600) + random.choice([0, 86400]))
        
        records_rich.append({
            "Transaction_ID": f"TXN-{100000 + i}",
            "Account_Number": acc["acc"],
            "Customer_Name": acc["name"],
            "Amount": amount,
            "Hour": hour,
            "Time": time_sec,
            "Merchant_Category": category,
            "Transaction_Type": tx_type,
            "Distance_KM": distance,
            "Is_Foreign": is_foreign,
            "Failed_PIN": failed_pins,
            "Device_Trust": device_trust,
            "Class": 0
        })
        
    # Generate Fraudulent Transactions
    for j in range(n_fraud):
        acc = random.choice(accounts)
        fraud_type = random.choice(["high_ticket", "overseas_spree", "odd_hours_crypto", "card_testing", "atm_drain"])
        
        if fraud_type == "high_ticket":
            amount = round(float(np.random.uniform(1200, 4800)), 2)
            category = random.choice(["Luxury & Jewelry", "Electronics", "Online Retail"])
            hour = random.choice([0, 1, 2, 3, 4, 22, 23])
            distance = round(float(np.random.uniform(150, 2500)), 1)
            is_foreign = 1 if random.random() < 0.6 else 0
            failed_pins = random.choice([1, 2, 3])
            tx_type = "Online_CNP"
            device_trust = int(np.random.uniform(10, 40))
            
        elif fraud_type == "overseas_spree":
            amount = round(float(np.random.uniform(850, 3500)), 2)
            category = random.choice(["Luxury & Jewelry", "Travel & Airlines", "Online Retail"])
            hour = random.randint(0, 23)
            distance = round(float(np.random.uniform(1200, 8000)), 1)
            is_foreign = 1
            failed_pins = random.choice([0, 1, 2])
            tx_type = "Online_CNP"
            device_trust = int(np.random.uniform(5, 35))
            
        elif fraud_type == "odd_hours_crypto":
            amount = round(float(np.random.uniform(1500, 5000)), 2)
            category = "Crypto & Trading"
            hour = random.choice([1, 2, 3, 4, 5])
            distance = round(float(np.random.uniform(200, 4000)), 1)
            is_foreign = 1 if random.random() < 0.7 else 0
            failed_pins = random.choice([0, 1, 2])
            tx_type = "Online_CNP"
            device_trust = int(np.random.uniform(10, 30))
            
        elif fraud_type == "card_testing":
            # Micro charges followed by a large withdrawal
            amount = round(float(random.choice([1.25, 2.00, 3.50, 4.99, 1499.00])), 2)
            category = "Online Retail"
            hour = random.randint(0, 23)
            distance = round(float(np.random.uniform(80, 1500)), 1)
            is_foreign = 1 if random.random() < 0.5 else 0
            failed_pins = random.choice([1, 2, 3])
            tx_type = "Online_CNP"
            device_trust = int(np.random.uniform(15, 45))
            
        else: # atm_drain
            amount = round(float(random.choice([500.00, 800.00, 1000.00, 1500.00])), 2)
            category = "Dining & Restaurants" if random.random() < 0.2 else "Electronics"
            hour = random.choice([1, 2, 3, 4])
            distance = round(float(np.random.uniform(300, 1800)), 1)
            is_foreign = 0
            failed_pins = random.choice([2, 3])
            tx_type = "ATM_Withdrawal"
            device_trust = int(np.random.uniform(20, 50))
            
        time_sec = float(hour * 3600 + np.random.uniform(0, 3600) + random.choice([0, 86400]))
        
        records_rich.append({
            "Transaction_ID": f"TXN-{200000 + j}",
            "Account_Number": acc["acc"],
            "Customer_Name": acc["name"],
            "Amount": amount,
            "Hour": hour,
            "Time": time_sec,
            "Merchant_Category": category,
            "Transaction_Type": tx_type,
            "Distance_KM": distance,
            "Is_Foreign": is_foreign,
            "Failed_PIN": failed_pins,
            "Device_Trust": device_trust,
            "Class": 1
        })
        
    df_rich = pd.DataFrame(records_rich).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # 2. Generate Standard Kaggle-compatible PCA features (V1-V28)
    # Fraud transactions exhibit distinct distribution shifts on key components
    # (specifically V1, V3, V4, V10, V11, V12, V14, V17 as documented in credit card fraud literature)
    v_data = {}
    is_fraud_mask = df_rich['Class'].values == 1
    
    for k in range(1, 29):
        col = f"V{k}"
        # Base normal values
        vals = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
        
        # Add realistic fraud signatures to known sensitive components
        if k == 1:   # Lower for fraud
            vals[is_fraud_mask] = np.random.normal(loc=-2.5, scale=1.8, size=n_fraud)
        elif k == 2: # Higher for fraud
            vals[is_fraud_mask] = np.random.normal(loc=1.8, scale=1.5, size=n_fraud)
        elif k == 3: # Much lower for fraud
            vals[is_fraud_mask] = np.random.normal(loc=-3.8, scale=2.1, size=n_fraud)
        elif k == 4: # Positive spike for fraud
            vals[is_fraud_mask] = np.random.normal(loc=3.2, scale=1.6, size=n_fraud)
        elif k == 9: # Negative shift
            vals[is_fraud_mask] = np.random.normal(loc=-1.5, scale=1.2, size=n_fraud)
        elif k == 10: # Strong negative indicator
            vals[is_fraud_mask] = np.random.normal(loc=-4.2, scale=2.0, size=n_fraud)
        elif k == 11: # Strong positive indicator
            vals[is_fraud_mask] = np.random.normal(loc=3.5, scale=1.8, size=n_fraud)
        elif k == 12: # Strong negative indicator
            vals[is_fraud_mask] = np.random.normal(loc=-4.5, scale=1.9, size=n_fraud)
        elif k == 14: # Most critical fraud indicator
            vals[is_fraud_mask] = np.random.normal(loc=-5.1, scale=2.2, size=n_fraud)
        elif k == 16: # Negative indicator
            vals[is_fraud_mask] = np.random.normal(loc=-2.8, scale=1.7, size=n_fraud)
        elif k == 17: # Strong negative indicator
            vals[is_fraud_mask] = np.random.normal(loc=-4.8, scale=2.1, size=n_fraud)
        elif k == 18: # Negative indicator
            vals[is_fraud_mask] = np.random.normal(loc=-1.9, scale=1.4, size=n_fraud)
            
        v_data[col] = vals
        
    df_standard = pd.DataFrame({
        "Time": df_rich["Time"],
        **v_data,
        "Amount": df_rich["Amount"],
        "Class": df_rich["Class"]
    })
    
    # Reorder columns to match standard Kaggle structure: Time, V1...V28, Amount, Class
    cols_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df_standard = df_standard[cols_order]
    
    return df_standard, df_rich

def save_datasets(target_dirs=None):
    """Saves both the standard and rich datasets in all specified locations."""
    if target_dirs is None:
        target_dirs = [
            ".",
            "Credit-Card-Fraud-Detection-main"
        ]
        
    df_standard, df_rich = generate_credit_card_data(n_samples=5000, fraud_ratio=0.035, random_seed=42)
    
    for d in target_dirs:
        if os.path.exists(d):
            std_path = os.path.join(d, "creditcard.csv")
            rich_path = os.path.join(d, "transactions_custom.csv")
            
            df_standard.to_csv(std_path, index=False)
            df_rich.to_csv(rich_path, index=False)
            
            print(f"[OK] Saved '{std_path}' ({len(df_standard)} rows, {len(df_standard.columns)} columns)")
            print(f"[OK] Saved '{rich_path}' ({len(df_rich)} rows, {len(df_rich.columns)} columns)")
            
    print("\n[SUCCESS] Custom dataset generation successfully completed!")
    return df_standard, df_rich

if __name__ == "__main__":
    save_datasets()
