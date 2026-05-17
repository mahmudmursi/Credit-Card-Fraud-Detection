# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using real-world highly imbalanced financial data.

This project demonstrates a complete ML workflow, including:

* data preprocessing
* feature engineering
* model training
* hyperparameter tuning
* evaluation metrics
* ROC-AUC analysis
* Precision-Recall analysis
* threshold optimization
* calibration
* monitoring and drift detection
* explainability
* model serialization
* production-style inference

---

# 🎯 Problem Statement

Credit card fraud causes significant financial losses worldwide.

The goal of this project is to build a machine learning system capable of detecting fraudulent transactions while minimizing false positives.

Since fraud detection datasets are highly imbalanced, traditional accuracy metrics are not sufficient. Instead, this project focuses on:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall analysis

---

# 📂 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

Characteristics:

* Highly imbalanced dataset
* Fraudulent transactions represent a very small percentage of total observations
* Features are anonymized PCA components (`V1–V28`)
* Includes:

  * `Time`
  * `Amount`
  * `Class`

Target variable:

```text
Class
0 → Normal transaction
1 → Fraudulent transaction
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Joblib
* Jupyter Notebook

Machine Learning concepts used:

* Pipelines
* Random Forest
* Logistic Regression
* Gradient Boosting
* GridSearchCV
* Cross Validation
* ROC-AUC
* Precision-Recall Curve
* Threshold Tuning
* Calibration
* Drift Monitoring
* Feature Importance
* Model Serialization

---

# ⚙️ Project Workflow

The project follows an end-to-end machine learning pipeline:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Model Building
6. Hyperparameter Tuning
7. Model Evaluation
8. Threshold Optimization
9. Calibration Analysis
10. Monitoring
11. Explainability
12. Production Inference
13. Final Model Selection

---

# 🧹 Feature Engineering

A log transformation was applied to transaction amounts:

```python
Amount_log = np.log1p(Amount)
```

Purpose:

* reduce skewness
* improve numerical stability
* normalize transaction amount distribution

---

# 🤖 Models Evaluated

Several machine learning models were tested and compared.

| Model                         | Precision | Recall | F1 Score |
| ----------------------------- | --------- | ------ | -------- |
| Logistic Regression           | 0.85      | 0.73   | 0.79     |
| Logistic Regression + Scaling | 0.83      | 0.63   | 0.72     |
| Gradient Boosting             | 0.83      | 0.60   | 0.70     |
| Random Forest                 | 0.94      | 0.83   | 0.88     |
| Tuned Random Forest           | 0.93      | 0.81   | 0.86     |

---

# 🏆 Final Selected Model

## Random Forest Classifier

Random Forest was selected as the final production-ready model because it achieved the strongest balance between:

* precision
* recall
* robustness
* generalization

### Why Random Forest?

The model provided:

* strong fraud detection capability
* low false positive rate
* stable cross-validation performance
* excellent ROC-AUC score
* strong performance on imbalanced data

---

# 📊 Final Model Performance

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.95   |
| Recall    | 0.82   |
| F1 Score  | 0.88   |
| ROC-AUC   | 0.9787 |

### Confusion Matrix

```text
TN = 56860
FP = 4
FN = 18
TP = 80
```

Interpretation:

* Only 4 legitimate transactions were incorrectly flagged as fraud.
* The model successfully detected 80 fraudulent transactions.
* 18 fraud cases were missed.

---

# 📈 ROC-AUC Analysis

ROC-AUC score:

```text
0.9787
```

A high ROC-AUC indicates that the model can effectively separate fraudulent transactions from normal ones.

---

# 📉 Precision-Recall Curve

Because fraud detection is highly imbalanced, Precision-Recall analysis was essential.

The model maintained:

* strong precision
* strong recall
* balanced fraud detection performance

This metric is often more informative than accuracy for fraud detection systems.

---

# 🎯 Threshold Optimization

Different prediction thresholds were evaluated.

### Threshold Tradeoff

Lower threshold:

* higher recall
* more fraud detection
* more false positives

Higher threshold:

* fewer false positives
* lower recall
* higher chance of missing fraud

This analysis simulates real-world business decisions in banking systems.

---

# 📏 Calibration Analysis

Calibration was performed to evaluate whether model probabilities reflected actual fraud likelihood.

This is important in financial systems because probability confidence directly impacts business decisions.

---

# 📡 Monitoring & Drift Detection

Feature drift monitoring was simulated by comparing train and test feature distributions.

This helps detect changes in customer transaction behavior over time.

Why this matters:

Fraud patterns evolve.

A model that performs well today may degrade in production if transaction distributions shift.

---

# 🔍 Feature Importance

Feature importance analysis identified the most influential variables affecting fraud detection.

Top features included:

* V17
* V14
* V12
* V10
* V16

This provides interpretability and insight into model behavior.

---

# 🔮 Production Inference

A reusable inference pipeline was built to simulate real-world prediction systems.

Example:

```python
result = predict_transaction(
    model,
    sample_transaction
)

print(result)
```

Example output:

```python
{
    "prediction": 0,
    "fraud_probability": 0.000047
}
```

Meaning:

* prediction = 0 → legitimate transaction
* fraud probability ≈ 0.0047%

---

# 🔧 Hyperparameter Tuning

GridSearchCV was used to optimize model parameters.

Best parameters:

```python
{
    'model__max_depth': 15,
    'model__min_samples_split': 2,
    'model__n_estimators': 100
}
```

Observation:

Hyperparameter tuning did not outperform the baseline Random Forest.

This demonstrates an important machine learning lesson:

**More tuning does not always guarantee better performance.**

---

# 🧪 Cross Validation

Cross-validation was performed to evaluate model stability.

Results showed:

* consistent performance
* strong generalization
* low performance variance

This increases confidence in model robustness.

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── Credit-Card-Fraud-Detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── inference.py
│
├── models/
│   ├── fraud_detection_model.pkl
│   ├── fraud_detection_pipeline.pkl
│   └── final_fraud_pipeline.pkl
│
├── images/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone <repository-url>
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Open Jupyter Notebook

```bash
jupyter notebook
```

## 4. Run the notebook

Run all notebook cells from top to bottom.

---

# 🚀 Future Improvements

Potential future upgrades include:

* REST API deployment
* Docker containerization
* Real-time fraud detection
* MLflow experiment tracking
* SHAP explainability
* Streamlit dashboard

---

# 🎯 Final Conclusion

In this project, multiple machine learning models were evaluated for credit card fraud detection.

Advanced machine learning concepts were applied, including:

* feature engineering
* pipelines
* hyperparameter tuning
* ROC-AUC analysis
* Precision-Recall analysis
* threshold tuning
* calibration
* monitoring
* explainability
* production-style inference

After experimentation and evaluation, Random Forest was selected as the final production-ready model due to its strong balance between precision, recall, and robustness.

This project demonstrates a complete end-to-end machine learning workflow for solving a real-world fraud detection problem.
