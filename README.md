# 💳 Credit Card Fraud Detection API

A production-ready Machine Learning API for detecting fraudulent credit card transactions using **FastAPI**, **Scikit-learn**, and **Random Forest**.

---

## 🚀 Project Overview

This project predicts whether a credit card transaction is **fraudulent** or **legitimate** using a trained Machine Learning pipeline.

The system includes:

- Machine Learning model training
- Feature engineering
- Fraud probability prediction
- REST API with FastAPI
- Automated testing with Pytest
- Logging system
- Interactive API documentation (Swagger UI)
- Docker support

---

## 🧠 Model Details

### Algorithm
- Random Forest Classifier

### Features
- Time
- V1 → V28 (PCA-transformed features)
- Amount

### Feature Engineering
- `Amount_log = log1p(Amount)`

### Output Example

```json
{
    "prediction": 0,
    "fraud_probability": 0.0001235
}
````

Where:

* `0` = Legitimate Transaction
* `1` = Fraudulent Transaction

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│── api.py
│── requirements.txt
│── Dockerfile
│── compose.yaml
│── README.md
│── .gitignore
│
├── data/
│   └── creditcard.csv
│
├── logs/
│   └── app.log
│
├── models/
│   ├── final_fraud_pipeline.pkl
│   ├── random_forest_baseline.pkl
│   └── tuned_random_forest.pkl
│
├── notebooks/
│   ├── Credit-Card-Fraud-Detection.ipynb
│   └── Final-Credit-Card-Fraud-Detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│   └── logger.py
│
└── tests/
    └── test_api.py
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

## 📘 API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

#### Home Endpoint

```http
GET /
```

#### Health Check

```http
GET /health
```

Example Response:

```json
{
    "status": "healthy",
    "model_loaded": true
}
```

#### Fraud Prediction

```http
POST /predict
```

Example Request:

```json
{
    "Time": 0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.5516,
    "V12": -0.617801,
    "V13": -0.99139,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}
```

Example Response:

```json
{
    "prediction": 0,
    "fraud_probability": 0.0001235
}
```

---

## 🧪 Running Tests

Run tests using:

```bash
python -m pytest
```

Expected output:

```text
3 passed
```

---

## 🐳 Docker Support

Build and run the application:

```bash
docker compose up --build
```

Application will be available at:

```text
http://localhost:8000
```

---

## 📊 Tech Stack

* Python
* FastAPI
* Scikit-Learn
* Pandas
* NumPy
* Joblib
* Pytest
* Docker
* Uvicorn

---

## 📌 Future Improvements

* CI/CD Pipeline
* Cloud Deployment (AWS / Azure / GCP)
* Model Monitoring
* Real-time Fraud Detection
* Model Versioning
* Advanced Feature Engineering

---

## 👨‍💻 Author

**Mahmoud Morsy**

AI Engineer | Data Science | Machine Learning
