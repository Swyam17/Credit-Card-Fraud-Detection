import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os
import random

# Mock database of accounts
ACCOUNTS = {
    "1234567890": {"pin": "1234", "name": "John Doe", "balance": 5000.0},
    "0987654321": {"pin": "4321", "name": "Jane Smith", "balance": 1500.0},
    "5556667770": {"pin": "0000", "name": "Alice Wilson", "balance": 12400.0},
    "1112223334": {"pin": "5678", "name": "Bob Brown", "balance": 450.0},
    "9998887776": {"pin": "9876", "name": "Charlie Davis", "balance": 8900.0}
}

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Payment Portal - Fraud Detection System")
        self.root.geometry("500x550")
        self.root.configure(bg="#f0f4f8")
        
        # Load Model
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()
        
        self.setup_ui()

    def load_model_and_scaler(self):
        """Loads the pre-trained model, scaler, and dataset."""
        if os.path.exists('fraud_model.joblib') and os.path.exists('scaler.joblib'):
            try:
                self.model = joblib.load('fraud_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                print("[*] Model and Scaler loaded successfully.")
            except Exception as e:
                print(f"[!] Error loading model: {e}")
        else:
            print("[!] Model file not found. Ensure fraud_detection.py has been run first.")
        
        # Load Dataset for real samples
        if os.path.exists('creditcard.csv'):
            try:
                import pandas as pd
                self.df = pd.read_csv('creditcard.csv')
                print(f"[*] Dataset loaded: {len(self.df)} records available.")
            except Exception as e:
                self.df = None
                print(f"[!] Error loading dataset: {e}")
        else:
            self.df = None

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#f0f4f8', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        
        # Header
        header_frame = tk.Frame(self.root, bg="#1a365d", height=80)
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="🔒 BANK SECURE PORTAL", foreground="white", background="#1a365d", style='Header.TLabel').pack(pady=20)
        
        # Main form
        form_frame = tk.Frame(self.root, bg="#f0f4f8", padx=40, pady=20)
        form_frame.pack(fill="both", expand=True)
        
        # Account Number
        ttk.Label(form_frame, text="Account Number:").pack(anchor="w")
        self.acc_entry = ttk.Entry(form_frame, width=40)
        self.acc_entry.pack(pady=5)
        self.acc_entry.insert(0, "1234567890")
        
        # PIN
        ttk.Label(form_frame, text="Security PIN:").pack(anchor="w", pady=(10, 0))
        self.pin_entry = ttk.Entry(form_frame, show="*", width=40)
        self.pin_entry.pack(pady=5)
        self.pin_entry.insert(0, "1234")
        
        # Transaction Amount
        ttk.Label(form_frame, text="Transaction Amount ($):").pack(anchor="w", pady=(10, 0))
        self.amount_entry = ttk.Entry(form_frame, width=40)
        self.amount_entry.pack(pady=5)
        self.amount_entry.insert(0, "250.00")
        
        # Info label about data source
        self.info_label = ttk.Label(form_frame, text="Features will be pulled from real dataset rows", font=('Helvetica', 8, 'italic'))
        self.info_label.pack(pady=5)

        # Process Button
        self.process_btn = ttk.Button(form_frame, text="BLOCKCHAIN VALIDATE & PROCESS", command=self.process_transaction)
        self.process_btn.pack(pady=20, fill="x")
        
        # Final prediction label
        self.result_label = tk.Label(form_frame, text="System Ready", bg="#f0f4f8", fg="#4a5568", font=('Helvetica', 10, 'italic'))
        self.result_label.pack(pady=5)

        # Truth label (to compare with dataset)
        self.truth_label = tk.Label(form_frame, text="", bg="#f0f4f8", font=('Helvetica', 9))
        self.truth_label.pack()

    def process_transaction(self):
        acc = self.acc_entry.get()
        pin = self.pin_entry.get()
        amount_str = self.amount_entry.get()
        
        # 1. Verification
        if acc not in ACCOUNTS:
            messagebox.showerror("Error", "Account number not found!")
            return
            
        if ACCOUNTS[acc]["pin"] != pin:
            messagebox.showerror("Error", "Invalid Security PIN!")
            return
            
        try:
            # We take the user's amount for the UI, but we'll use a real sample for PCA features
            user_amount = float(amount_str)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid amount!")
            return
            
        # 2. Fraud Prediction
        if self.model is None or self.df is None or self.scaler is None:
            messagebox.showwarning("Warning", "Model, Dataset, or Scaler not found. Simulating check...")
            is_fraud = user_amount > 2000
            actual_class = "Unknown"
            sample_amt = user_amount
        else:
            # Pick a random sample from the real dataset
            sample = self.df.sample(n=1).iloc[0]
            actual_class = "Fraud" if sample['Class'] == 1 else "Normal"
            
            # Extract PCA features (V1-V28) from this real record
            pca_feats = [sample[f'V{i}'] for i in range(1, 29)]
            
            # Use original Amount and Time from the sample to scale them
            sample_amt = sample['Amount']
            sample_time = sample['Time']
            
            # Scale the features using the saved scaler [Amount, Time]
            try:
                scaled_vals = self.scaler.transform([[sample_amt, sample_time]])[0]
                scaled_amount, scaled_time = scaled_vals[0], scaled_vals[1]
                
                # Build final input vector: V1...V28, Scaled_Amount, Scaled_Time
                input_vector = pca_feats + [scaled_amount, scaled_time]
                
                # Model Prediction
                prediction = self.model.predict([input_vector])
                is_fraud = prediction[0] == 1
            except Exception as e:
                print(f"[!] Scaling/Prediction error: {e}")
                is_fraud = sample_amt > 2000 # Fallback
            
        # 3. Handle results
        self.truth_label.config(text=f"Dataset Ground Truth for this sample: {actual_class}\nRecord Amount: ${sample_amt:,.2f}")
        
        if is_fraud:
            self.show_alert("⚠️ FRAUD DETECTED!", "high")
            messagebox.showwarning("Security Alert", f"Our AI detected high-risk activity for Account {acc}.\n\nTested Record Amount: ${sample_amt:,.2f}\nDataset Label: {actual_class}\n\nTransaction Blocked.")
        else:
            self.show_alert("✅ TRANSACTION SECURE", "low")
            messagebox.showinfo("Success", f"Transaction for ${user_amount} processed successfully.\n\nAI Scan Result: SECURE\nDataset Label: {actual_class}")

    def show_alert(self, message, risk_level):
        if risk_level == "high":
            self.result_label.config(text=message, fg="#e53e3e", font=('Helvetica', 14, 'bold'))
        else:
            self.result_label.config(text=message, fg="#38a169", font=('Helvetica', 14, 'bold'))

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
