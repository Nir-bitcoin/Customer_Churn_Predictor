 **[Live Demo →](https://customer-churn-predictor-28z3.onrender.com)**
# Customer Churn Predictor

A machine learning project that predicts which telecom customers are about to leave — so the business can keep them before it's too late.

Built with XGBoost, explained with SHAP.

<div>
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/61/Scikit_learn_logo_small.svg" height="60">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png" height="60">
  <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_logo.png" height="60">
</div>

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/model-XGBoost-red)
![SHAP](https://img.shields.io/badge/SHAP-explainable-green)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.842-success)

---

## What's the Problem?

Getting a new customer costs 5–7x more than keeping an existing one. So predicting who's about to leave — before they do — saves real money.

## The Data

- IBM Telco Customer Churn (Kaggle)
- 7,043 customers, 21 features
- Only 26.5% churned — imbalanced data

## What I Did

1. Cleaned and preprocessed the data
2. Encoded categorical features with `ColumnTransformer`
3. Handled class imbalance with `scale_pos_weight`
4. Compared Random Forest vs XGBoost
5. Tuned XGBoost with `RandomizedSearchCV`
6. Explained results with SHAP

## Results

| Model | ROC-AUC |
|-------|---------|
| Random Forest | 0.822 |
| XGBoost (default) | 0.824 |
| **XGBoost (tuned)** ⭐ | **0.842** |

## Top Churn Drivers

1. Month-to-month contract
2. Tenure
3. Monthly charges
4. No online security
5. No tech support

![ROC Curve](images/roc_curve.png)
![SHAP Importance](images/shap_importance.png)


## Author
Niranjan vishe

GitHub: [@Nir-bitcoin](https://github.com/Nir-bitcoin)
LinkedIn: [nirvishe](https://www.linkedin.com/in/nirvishe)

## How to Run

```bash
git clone https://github.com/Nir-bitcoin/Customer_Churn_Predictor.git
cd Customer_Churn_Predictor
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_analysis.ipynb


