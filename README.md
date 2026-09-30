# Industrial Machine Failure Prediction

## Objective
Predict whether an industrial machine will experience failure from operating conditions.

## Dataset
AI4I 2020 Predictive Maintenance Dataset supplied as `data/ai4i2020.csv`.

- Rows: 10,000
- Columns: 14
- Missing values: 0
- Duplicate rows: 0
- Failure cases: 339 (3.39%)

## Features used
Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].

UDI and Product ID are identifiers. TWF, HDF, PWF, OSF and RNF are failure-mode indicators and are excluded from the main predictive model because they can encode outcome-related information.

## Models
Logistic Regression, Decision Tree, Random Forest. Class-balanced training is used because failures are rare.

## Results
- Logistic Regression: Accuracy 0.8245 | Precision 0.1418 | Recall 0.8235 | F1 0.2419 | ROC-AUC 0.9069
- Decision Tree: Accuracy 0.9365 | Precision 0.3314 | Recall 0.8529 | F1 0.4774 | ROC-AUC 0.8762
- Random Forest: Accuracy 0.9765 | Precision 0.6364 | Recall 0.7206 | F1 0.6759 | ROC-AUC 0.9653

## Final model
**Random Forest**

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
