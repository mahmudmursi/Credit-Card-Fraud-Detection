# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

Credit card fraud detection is one of the most important real-world applications of Machine Learning in the financial sector.

This project builds a complete fraud detection pipeline using multiple machine learning models to identify fraudulent credit card transactions from highly imbalanced transactional data.

The project focuses not only on model accuracy, but also on:

* Recall optimization
* Fraud detection capability
* Threshold tuning
* Cross-validation reliability
* Business-oriented evaluation

---

# 🚀 Results Highlights

* ROC-AUC Score: 0.9759
* F1-score: 0.87
* False Positives: 5 only
* Fraud Detection Recall: 81%
* Strong performance on highly imbalanced data

---

# 🎯 Project Objectives

The main goals of this project are:

* Detect fraudulent transactions effectively
* Handle severe class imbalance
* Compare multiple ML algorithms
* Improve model performance using:

  * Feature Engineering
  * Hyperparameter Tuning
  * Threshold Optimization
* Analyze business trade-offs between:

  * Fraud detection
  * False alarms

---

# 📂 Dataset Information

Dataset: Credit Card Fraud Detection Dataset

The dataset contains anonymized credit card transactions made by European cardholders.

## Dataset Characteristics

| Property           | Value   |
| ------------------ | ------- |
| Total Transactions | 284,807 |
| Fraud Cases        | 492     |
| Fraud Ratio        | 0.172%  |
| Features           | 31      |
| Target Column      | `Class` |

---

# ⚠️ Problem Type

## Imbalanced Binary Classification

Where:

* `0` → Normal Transaction
* `1` → Fraudulent Transaction

Due to the extreme imbalance:

* Accuracy alone is misleading
* Recall and F1-score are critical metrics

---

# 🧠 Machine Learning Workflow

```text
EDA
→ Visualization
→ Feature Engineering
→ Scaling
→ Modeling
→ Cross Validation
→ Hyperparameter Tuning
→ Threshold Optimization
→ Model Evaluation
→ Business Interpretation
```

---

# 🔎 Exploratory Data Analysis (EDA)

Performed detailed exploratory analysis including:

* Dataset structure inspection
* Missing value analysis
* Fraud distribution analysis
* Transaction amount analysis
* Time distribution analysis
* Correlation matrix analysis

---

# 📊 Key EDA Insights

## ⚠️ Severe Class Imbalance

Fraudulent transactions represent less than:

```text
0.172%
```

of the entire dataset.

---

## 💰 Transaction Amount Distribution

* Most transactions are low-value transactions
* Fraud cases are concentrated around smaller amounts
* Large outliers exist in normal transactions

---

## 🔥 Correlation Insights

Some PCA-transformed features showed stronger relationships with fraud activity, especially:

* V12
* V14
* V17
* V10

---

# 🛠️ Feature Engineering

Several feature engineering techniques were applied.

---

## 🔄 Log Transformation

The `Amount` feature was highly skewed.

Applied:

```python
df['Amount_log'] = np.log1p(df['Amount'])
```

Benefits:

* Reduced skewness
* Compressed outliers
* Stabilized feature distribution

---

## 🔗 Interaction Features

Created new interaction features:

```python
df['V12_V14'] = df['V12'] * df['V14']
df['V10_V17'] = df['V10'] * df['V17']
```

Purpose:

* Capture non-linear fraud patterns
* Improve representation learning
* Detect combined anomalies

---

# ⚖️ Scaling & Normalization

Applied `StandardScaler` to evaluate its impact on Logistic Regression.

## Observation

Scaling did not improve performance significantly because:

* Features were already PCA-transformed
* Dataset was partially normalized beforehand

## Important Insight

Preprocessing techniques should always be validated experimentally.

---

# 🤖 Models Trained

| Model                        |
| ---------------------------- |
| Logistic Regression          |
| Logistic Regression (Scaled) |
| Decision Tree                |
| Random Forest                |
| Gradient Boosting            |
| Tuned Random Forest          |

---

# 📈 Model Performance Comparison

| Model | Precision | Recall | F1-score |
|--|--|--|--|
| Logistic Regression | 0.85 | 0.73 | 0.79 |
| Logistic Regression (Scaled) | 0.83 | 0.65 | 0.73 |
| Decision Tree | 0.89 | 0.78 | 0.83 |
| Random Forest | 0.95 | 0.82 | 0.88 |
| Tuned Random Forest | 0.94 | 0.81 | 0.87 |
| Gradient Boosting | 0.83 | 0.60 | 0.70 |

---

## 🚀 Gradient Boosting Analysis

Gradient Boosting achieved strong precision but lower recall compared to Random Forest.

### Key Observation

The model became more conservative when predicting fraud transactions.

As a result:
- False positives decreased
- Missed fraud cases increased

### Business Interpretation

In fraud detection systems, Recall is often more important than Precision because:

- Missing fraudulent transactions can lead to direct financial losses
- False alarms are generally less costly

Therefore, Random Forest remained the preferred final model for this project.

---

# 🌲 Final Selected Model

## Tuned Random Forest

Selected because it achieved:

* Strong fraud detection capability
* Stable cross-validation performance
* Excellent Precision/Recall balance
* Reliable generalization

---

# 🔥 Hyperparameter Tuning

Used `GridSearchCV` to optimize:

* `n_estimators`
* `max_depth`
* `min_samples_split`

## Best Parameters

```python
{
    'max_depth': 15,
    'min_samples_split': 5,
    'n_estimators': 100
}
```

---

# 🔁 Cross Validation

Applied:

## Stratified K-Fold Cross Validation

Purpose:

* Reduce evaluation randomness
* Preserve fraud ratio across folds
* Measure generalization stability

## Final Cross Validation Result

```text
Mean F1 Score ≈ 0.847
```

---

# 🎯 Threshold Tuning

Different classification thresholds were evaluated.

| Threshold | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| 0.3       | 0.84      | 0.87   | 0.85     |
| 0.5       | 0.94      | 0.81   | 0.87     |
| 0.7       | 0.97      | 0.71   | 0.82     |

---

# 🧠 Threshold Trade-Off

## Lower Thresholds

* Detect more fraud
* Increase false alarms

## Higher Thresholds

* Reduce false positives
* Miss more fraud cases

---

# 🔍 Confusion Matrix Analysis

| Category             | Count  |
| -------------------- | ------ |
| True Negatives (TN)  | 56,859 |
| False Positives (FP) | 5      |
| False Negatives (FN) | 19     |
| True Positives (TP)  | 79     |

---

## 🧠 Confusion Matrix Insights

* The model correctly classified most normal transactions
* False positives are extremely low
* The model successfully detected most fraud cases
* Some fraud transactions were still missed

### ⚠️ Business Interpretation

False Positives:

* May inconvenience customers
* Trigger unnecessary fraud alerts

False Negatives:

* Represent undetected fraud
* Can lead to financial loss

The current model achieves a strong balance between:

* Fraud detection capability
* Customer experience
* Operational reliability

---

# 📊 ROC Curve & AUC

Final ROC-AUC Score:

```text
AUC = 0.9759
```

## Interpretation

The model demonstrates:

* Excellent discrimination capability
* Strong fraud separation performance
* Robust classification quality

---

## 🧠 ROC Curve Analysis

The ROC Curve evaluates the model’s ability to distinguish between:

* Fraudulent transactions
* Normal transactions

### 🎯 Key Insights

* The model separates fraud and normal transactions very effectively
* High True Positive Rate is achieved with low False Positive Rate
* Performance remains strong despite severe class imbalance

### 💼 Business Interpretation

A high AUC score means the system can:

* Detect fraudulent transactions reliably
* Reduce unnecessary fraud alerts
* Improve trust in automated fraud detection systems

---

# 📌 Feature Importance

The Random Forest model identified the following features as most influential:

* V12
* V17
* V14
* V10
* V11

These features contributed most to distinguishing fraudulent behavior from legitimate transactions.

---

# 🧠 Key Technical Insights

This project demonstrates practical experience with:

* Imbalanced classification
* Feature engineering
* Ensemble learning
* Cross-validation
* Threshold optimization
* Hyperparameter tuning
* Model evaluation
* Business-oriented ML analysis

---

# 💼 Business Impact

The final fraud detection system can help financial institutions:

* Detect suspicious transactions early
* Reduce financial fraud losses
* Minimize unnecessary customer alerts
* Improve operational efficiency
* Support automated fraud prevention systems

---

# 🚀 Future Improvements

Possible future enhancements include:

* XGBoost / LightGBM
* SMOTE oversampling
* Deep Learning approaches
* Real-time fraud detection APIs
* Explainable AI (SHAP values)
* Streamlit deployment
* Model monitoring systems

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mahmudmursi/Credit-Card-Fraud-Detection
```

Move into the project folder:

```bash
cd Credit-Card-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Fraud_Detection_ML_Pipeline.ipynb
```

---

# 🧰 Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Core Programming     |
| Pandas       | Data Analysis        |
| NumPy        | Numerical Operations |
| Matplotlib   | Visualization        |
| Seaborn      | Visualization        |
| Scikit-learn | Machine Learning     |
| Joblib       | Model Serialization  |

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── Fraud_Detection_ML_Pipeline.ipynb
│
├── models/
│   └── fraud_detection_model.pkl
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── README.md
│
└── requirements.txt
```

---

# 📦 requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

---

# 🏁 Final Conclusion

This project successfully built a robust fraud detection pipeline capable of handling severe class imbalance while maintaining strong fraud detection performance.

The final model achieved:

* High Precision
* Strong Recall
* Excellent AUC score
* Reliable generalization

The workflow reflects a realistic production-oriented machine learning solution for financial fraud detection systems.

---

# 📜 License

This project is open-source and available under the MIT License.
