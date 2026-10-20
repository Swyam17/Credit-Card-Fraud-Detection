# AegisGuard AI - Credit Card Fraud Detection & Banking Security Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20Random%20Forest-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Web%20App-Flask%203.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Deployment](https://img.shields.io/badge/Deploy-Render%20%7C%20HuggingFace%20%7C%20Docker-brightgreen.svg)](#cloud-deployment-guide)
[![Status](https://img.shields.io/badge/Status-Cloud%20Ready-success.svg)]()

A production-grade, self-contained AI-powered Credit Card Fraud Detection platform. Features a real-time **Risk Radar**, interactive **3D Virtual Card**, **Banking Simulator with AI Intervention**, **Batch CSV Scanner**, **Custom Dataset Synthesizer**, and **REST API**.

---

## Key Highlights & Changes

1. **Self-Contained Custom Dataset Engine (`generate_dataset.py`)**:
   - No external Kaggle download dependencies or broken links!
   - Generates mathematically authentic, reproducible transaction distributions (5,000+ rows).
   - Outputs two formats:
     - `transactions_custom.csv`: Human-interpretable real-world attributes (Customer Name, Category, Geolocation Distance, Foreign Flag, Failed PIN Attempts, Device Trust).
     - `creditcard.csv`: 100% compatible standard Kaggle format (`Time`, `V1-V28` PCA features, `Amount`, `Class`).
   - Dynamic dataset studio directly accessible from the browser UI to customize sample size (1,000 - 20,000) and fraud ratio.

2. **Cloud-Ready Web Architecture (Accessible via Deployed Link)**:
   - Replaced legacy desktop-only Tkinter GUI with a modern, glassmorphic, responsive web portal.
   - Built on Flask + Vanilla HTML5/CSS3/JavaScript + Chart.js.
   - Cloud-native PORT binding (`os.environ.get('PORT', 5000)`) compatible with **Render**, **Hugging Face Spaces**, **Railway**, **Heroku**, and **Docker**.

3. **Multi-Feature AI Radar & Explainable AI**:
   - Random Forest Classifier with minority class rebalancing and standard feature scaling.
   - Computes live fraud probability (0% - 100%) with risk levels: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
   - Explains the specific reasons why a transaction was flagged (e.g. night hours, cross-border anomaly, failed PIN attempts, abnormal dollar amount).

---

## Interactive Web Portal Features

| Module | Description |
| :--- | :--- |
| **Live Fraud Radar** | Real-time risk meter with 1-click test scenarios (Morning Coffee, Gas, 3 AM Crypto, Overseas Luxury). |
| **Banking Portal Simulator** | Live account portfolio with balance tracking and AI fraud intervention (approves safe transactions or blocks risky transfers). |
| **Model & Data Analytics** | Interactive Chart.js charts showing diurnal fraud surge, class distribution, category risk, and confusion matrix. |
| **Batch CSV Scanner** | Drag-and-drop CSV scanner that processes bulk transactions and outputs total stolen funds protected. |
| **Custom Dataset Studio** | Interactive synthesizer slider to generate and download fresh datasets on demand. |
| **REST API** | Full programmatic access via `/api/predict`, `/api/health`, and `/api/stats`. |

---

## Cloud Deployment Guide

Deploy this platform with a permanent, public HTTPS link accessible by anyone on any device.

### Option 1: 1-Click Deploy to Render.com (Recommended - Free)
1. Fork or push this repository to your **GitHub** account.
2. Sign in to [Render.com](https://render.com) and click **New +** &gt; **Web Service**.
3. Connect your repository.
4. Render will automatically detect the included `Procfile` and `render.yaml`:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Click **Create Web Service**. Within 2 minutes, your public link is live (e.g. `https://credit-card-fraud-detection.onrender.com`)!

### Option 2: Deploy to Hugging Face Spaces (Free)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space name and choose **Docker** (or **Blank**).
3. Upload or push this repository. The included `Dockerfile` will automatically configure dependencies, run model training, and expose port `5000`.
4. Your application runs with a dedicated public link.

### Option 3: Deploy via Docker
```bash
# Build the Docker image
docker build -t credit-card-fraud-ai .

# Run the container
docker run -p 5000:5000 credit-card-fraud-ai
```
Access at `http://localhost:5000`.

---

## Local Development & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Custom Dataset (Optional - Pre-generated datasets included)
```bash
python generate_dataset.py
```
This generates:
- `creditcard.csv` (Standard 31 columns: `Time`, `V1-V28`, `Amount`, `Class`)
- `transactions_custom.csv` (Rich transactional dataset)

### 4. Train the AI Model (Optional - Pre-trained model included)
```bash
python fraud_detection.py
```
Outputs:
- `fraud_model.joblib`: Random Forest Classifier
- `scaler.joblib`: Fitted Standard Scaler
- `model_metrics.json`: Accuracy, Precision, Recall, ROC-AUC, Confusion Matrix

### 5. Launch the Web Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## REST API Documentation

### `POST /api/predict`
Evaluates a single transaction and returns real-time fraud probability, classification, risk factors, and recommended action.

#### Request Example:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 4200.00,
    "hour": 3,
    "merchant_category": "Crypto & Trading",
    "distance": 2500.0,
    "is_foreign": 1,
    "failed_pin": 2,
    "device_trust": 20
  }'
```

#### Response Example:
```json
{
  "success": true,
  "prediction": {
    "is_fraud": 1,
    "fraud_probability": 0.975,
    "fraud_percentage": 97.5,
    "risk_level": "CRITICAL",
    "risk_color": "#ef4444",
    "recommended_action": "BLOCK_TRANSACTION",
    "risk_reasons": [
      "High transaction amount ($4,200.00) deviates significantly from standard retail baseline.",
      "Transaction conducted at 03:00 (unusual nighttime hours for cardholder).",
      "High-risk merchant industry category: 'Crypto & Trading'.",
      "Unusual location anomaly: 2,500 km away from billing address.",
      "Cross-border / international transaction detected.",
      "Security Alert: 2 failed PIN attempt(s) recorded prior to transaction.",
      "Untrusted device profile (Trust Score: 20/100)."
    ]
  }
}
```

### Other Endpoints:
- `GET /api/health`: Uptime monitoring endpoint for cloud platforms.
- `GET /api/stats`: Dataset distribution and model evaluation metrics.
- `GET /api/accounts`: Mock accounts for banking simulator.
- `POST /api/bank-transfer`: Processes transaction with PIN verification and AI fraud intercept.
- `POST /api/batch-predict`: Accepts multipart CSV file and returns batch predictions.
- `GET /download/transactions_custom.csv`: Direct download for rich dataset.
- `GET /download/creditcard.csv`: Direct download for Kaggle-compatible dataset.
- `GET /download/sample-batch.csv`: Direct download for sample test CSV.

---

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── app.py                     # Main web application & REST API server
├── model_engine.py            # ML training, scaler, evaluation, and prediction engine
├── generate_dataset.py        # Custom dataset generation script
├── fraud_detection.py         # Standalone AI model training pipeline
├── creditcard.csv             # Kaggle-standard format custom dataset
├── transactions_custom.csv    # Rich human-interpretable dataset
├── fraud_model.joblib         # Pre-trained Random Forest model
├── scaler.joblib              # Fitted StandardScaler
├── model_metrics.json         # Evaluation metrics & confusion matrix
├── requirements.txt           # Python dependencies
├── Procfile                   # Cloud process manager configuration (Render/Heroku)
├── render.yaml                # Render 1-click deploy blueprint
├── Dockerfile                 # Container image for Docker & Hugging Face Spaces
├── templates/
│   └── index.html             # Responsive HTML5 dashboard
└── static/
    ├── css/
    │   └── style.css          # Glassmorphism dark theme styling
    └── js/
        └── app.js             # Client controller, Chart.js & animations
```

---

## License
MIT License. Built for secure financial engineering and machine learning education.
