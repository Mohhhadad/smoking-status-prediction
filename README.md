# Smoking Status Prediction

Binary classification on routine health check-up data. The question: can we tell whether someone currently smokes from standard biometric and blood-test values, without asking them?

Student project for **IE500 Data Mining**, University of Mannheim, Summer 2026.

## Dataset

[Body Signal of Smoking](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking) (Kaggle): 55,692 check-up records, 27 raw features, no missing values. About 37% of records are smokers.

The CSV isn't included in this repo. Download `smoking.csv` from Kaggle and place it next to the notebook (or at `MyDrive/smoking.csv` when running in Colab).

## Approach

- **Preprocessing:** dropped `ID` and the constant `oral` column, binary-encoded `gender` → `male` and `tartar`.
- **Feature engineering:** `BMI`, `pulse_pressure`, `avg_eyesight`, `hearing_problem` (originals dropped to avoid multicollinearity).
- **Split & scaling:** stratified 80/20 split, `RobustScaler` fitted on train only (heavy-tailed features such as GTP and triglyceride).
- **Models:** Logistic Regression (baseline), Gaussian Naïve Bayes (GridSearchCV), Random Forest and XGBoost (RandomizedSearchCV, 5-fold CV, scored on ROC-AUC). Class imbalance handled with `class_weight="balanced"` / `scale_pos_weight`.
- **Ablation:** every model retrained without the `male` feature to check how much the models depend on gender.

## Results (held-out test set)

| Model | ROC-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | **0.8854** | **0.7497** | 0.8814 | 0.6522 | 0.7838 |
| XGBoost | 0.8734 | 0.7425 | 0.8213 | 0.6776 | 0.7908 |
| Random Forest (–gender) | 0.8715 | 0.7340 | 0.8106 | 0.6707 | 0.7843 |
| XGBoost (–gender) | 0.8615 | 0.7316 | 0.7991 | 0.6747 | 0.7847 |
| Logistic Regression | 0.8276 | 0.7054 | 0.9147 | 0.5741 | 0.7195 |
| Logistic Regression (–gender) | 0.7959 | 0.6639 | 0.7644 | 0.5868 | 0.7158 |
| Naïve Bayes | 0.7948 | 0.6635 | 0.7744 | 0.5803 | 0.7115 |
| Naïve Bayes (–gender) | 0.7651 | 0.5356 | 0.4811 | 0.6041 | 0.6936 |

The tree ensembles clearly beat the linear and probabilistic baselines. Removing gender costs Random Forest only about 0.014 AUC: biomarkers such as hemoglobin, GTP, waist circumference and triglyceride carry most of the signal on their own.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook smoking_prediction.ipynb
```

Or open the notebook in Google Colab.

## Team

Group project by a team of six students. Mohammad Hadad implemented and tuned the Random Forest model and wrote that part of the report.
