# 🚀 Feature Shipment Delay Prediction

Predict how many days a feature shipment will be delayed using sprint metadata, team characteristics, and historical performance.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006600)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Problem Statement

Given sprint and team-level metadata — such as team size, feature complexity, number of blockers, and historical delay patterns — predict the `delay_days` for a planned feature shipment.

This is a **supervised regression** problem solved end-to-end in a single Jupyter Notebook.

---

## 📁 Project Structure

```
feature-shipment-prediction/
├── dataset.csv                          # Source dataset (~1,300 rows)
├── shipment_delay_prediction.ipynb      # Complete ML pipeline notebook
├── xgb_delay_model.joblib               # Trained XGBoost model artifact
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `planned_shipment_date` | Planned date for the feature shipment |
| `team_size` | Number of team members |
| `feature_complexity` | Complexity score of the feature |
| `num_dependencies` | Number of upstream dependencies |
| `sprint_length_weeks` | Sprint duration in weeks |
| `num_blockers` | Number of active blockers |
| `holidays_in_sprint` | Number of holidays within the sprint |
| `priority_encoded` | Encoded priority level |
| `past_avg_delay_days` | Historical average delay (standardized) |
| `estimated_bug_count` | Estimated number of bugs |
| **`delay_days`** | **Target — actual delay in days** |

- **Rows:** ~1,300
- **Missing values:** None
- **Duplicates:** None

---

## 🔬 Approach

The notebook follows a structured ML pipeline:

1. **Exploratory Data Analysis** — distributions, correlations, outlier detection
2. **Data Cleaning** — type corrections, duplicate/missing value handling
3. **Feature Selection & Engineering** — data-driven approach using mutual information, permutation importance, and CV ablation:
   - **Removed** all 7 date-derived features (each one hurt CV performance)
   - **Added** 3 validated interaction features: `complexity_x_bugs`, `total_risk`, `complexity_x_deps`
4. **Baseline Comparison** — 4 models evaluated on identical train/test split
5. **Hyperparameter Tuning** — RandomizedSearchCV with 5-fold CV on XGBoost
6. **Final Evaluation** — metrics, diagnostic plots, feature importance

---

## 📈 Results

### Baseline Comparison

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.882 | 1.108 | 0.950 |
| XGBoost | 1.050 | 1.357 | 0.925 |
| Random Forest | 1.107 | 1.413 | 0.919 |
| Decision Tree | 1.485 | 1.967 | 0.843 |

### Tuned XGBoost (Final Model)

| Metric | Score |
|--------|-------|
| **MAE** | **0.915** |
| **RMSE** | **1.135** |
| **R²** | **0.948** |
| **MAPE** | **7.71%** |

> After hyperparameter tuning with data-driven interaction features, XGBoost achieves strong performance with R² = 0.948 while providing built-in feature importance for interpretability.

---

## 🛠️ Tech Stack

- **pandas** / **numpy** — data manipulation
- **matplotlib** / **seaborn** — visualization
- **scikit-learn** — preprocessing, model selection, evaluation
- **XGBoost** — gradient boosted regression
- **joblib** — model serialization

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/brijeshkumarmorya/feature-shipment-prediction.git
cd feature-shipment-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 3. Run the notebook

```bash
jupyter notebook shipment_delay_prediction.ipynb
```

### 4. Use the saved model

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("xgb_delay_model.joblib")


def prepare_features(df):
    """
    Generate the engineered features used during model training.
    """
    df = df.copy()

    df["complexity_x_bugs"] = (
        df["feature_complexity"] * df["estimated_bug_count"]
    )

    df["total_risk"] = (
        df["feature_complexity"]
        * df["num_blockers"]
        * df["num_dependencies"]
    )

    df["complexity_x_deps"] = (
        df["feature_complexity"] * df["num_dependencies"]
    )

    df["sprint_pressure"] = (
        df["feature_complexity"]
        * (df["num_dependencies"] + df["num_blockers"])
    ) / df["sprint_length_weeks"]

    return df


# -------------------------
# Example Input
# -------------------------
sample = pd.DataFrame([{
    "team_size": 15,
    "feature_complexity": 2.0,
    "num_dependencies": 0,
    "sprint_length_weeks": 2,
    "num_blockers": 1,
    "holidays_in_sprint": 0,
    "priority_encoded": 1,
    "past_avg_delay_days": 0.5,
    "estimated_bug_count": 5
}])

# Prepare features
sample = prepare_features(sample)

# Arrange columns exactly as expected by the model
sample = sample[model.feature_names_in_]

# Predict
predicted_delay = model.predict(sample)[0]

print(f"Predicted Shipment Delay: {predicted_delay:.2f} days")
```

---

## 🔮 Future Improvements

- **More training data** — the ~1,300 row dataset limits model capacity
- **Ensemble / Stacking** — blend XGBoost with Linear Regression
- **SHAP values** — model-agnostic interpretability beyond feature importance
- **Temporal train/test split** — for production-realistic evaluation
- **sklearn Pipeline** — wrap preprocessing + model for cleaner deployment
- **CI/CD integration** — automated retraining on new data

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
