# House-pricing
# King County House Price Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **From raw chaos to $50k accuracy — preprocessing is 80% of ML.**

An end-to-end ML pipeline that predicts house prices in King County with **89.25% accuracy** (MAE $50,860) using **CatBoost on Clean_All**.

---

## �� Dataset
- **Source**: King County House Sales (kc_house_data.csv)
- **Size**: 21,613 houses | 21 original columns
- **Target**: `price` (log-transformed for training)

---

## �� Objective
- Build a robust model that generalizes to unseen houses
- Compare 8 preprocessing versions
- Test 7 algorithms + tuned hyperparameters
- Achieve **<12% average error** on test set

---

## ��️ Preprocessing Pipeline
- **8 versions** created for ablation study
- Outlier removal (IQR × 3)
- Z-score scaling
- Feature selection (SelectKBest k=12)
- Date extraction → year/month/dow
- Binary handling: `waterfront` → `waterfront_1`

### Versions Overview
| Version          | Steps                          | Features | Rows (Train) |
|------------------|--------------------------------|----------|--------------|
| Raw_All          | None                           | 18       | 17,290       |
| Raw_Selected     | Selection                      | 14       | 17,290       |
| Out_All          | Outliers                       | 18       | ~15,800      |
| Out_Selected     | Outliers + Selection           | 14       | ~15,800      |
| Norm_All         | Scaling                        | 18       | 17,290       |
| Norm_Selected    | Scaling + Selection            | 14       | 17,290       |
| Clean_All        | Outliers + Scaling             | 18       | ~15,800      |
| **Clean_Selected** | **All 3 (BEST)**             | **14**   | ~15,800      |

---

## �� Models Tested
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- AdaBoost

**Hyperparameter tuning via GridSearchCV** on best version per model.

---

## �� Results
### Top Performers (Test Set)
| Rank | Model             | Version        | MAE     | MAE%  | R²     |
|------|-------------------|----------------|---------|-------|--------|
| 1    | **CatBoost**      | Clean_All      | **$50,860** | **10.75%** | **0.8775** |
| 2    | XGBoost           | Clean_Selected | $52,153 | 11.02%| 0.8673 |
| 3    | Gradient Boosting | Clean_Selected | $52,212 | 11.03%| 0.8661 |

**Winner**: CatBoost on `Clean_All` → **89.25% accurate** on unseen houses!

---

## �� Project Structure
```
.
├── data/
│   └── kc_house_data.csv
├── versions/                  # 8 preprocessed CSVs + plots
├── models_all/                # 7 tuned .pkl files
├── gridsearch_results.txt     # Best hyperparameters
├── results_table.py           # Full metrics + % table
├── gridsearch_all.py          # Tuning script
├── best_model_all.pkl         # Production model
└── README.md
```

---

## ⚙️ How to Run
```bash
# 1. Preprocess (creates versions/)
python 1_eda_all_versions.py

# 2. Train all models
python results_table.py

# 3. Tune best params
python gridsearch_all.py

# 4. Load & predict
joblib.load('best_model_all.pkl')
```

---

## �� Key Insights
- **Preprocessing > Model choice**: Clean_Selected cut MAE by $70k+
- **CatBoost wins**: Handles ordinals natively, no one-hot needed
- **Overfitting check**: Small train-test gap = reliable model
- **Feature selection**: Only 14/18 features needed → faster + better

---

## �� Visualizations
- Price vs sqft_living scatter (per version)
- Feature importance plots
- Train vs Test MAE% comparison

---

## �� Next Steps
- Deploy as FastAPI endpoint
- Add SHAP explanations
- Try ensemble stacking (CatBoost + XGBoost)
- Real-time prediction web app

---

## �� License
MIT © 2025 Chantal Silva
