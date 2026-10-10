# Customer Churn Prediction

**Author:** Shoaib ur Rehman | Quaid-i-Azam University (QAU)
**Course:** Introduction to Applied AI - Project 1: ML-Powered Customer Analytics
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,043 customers, 21 features

## Repository contents
- `week1-eda.ipynb` - Week 1: exploratory data analysis
- `week2-ml-models.ipynb` - Week 2: building, evaluating and interpreting ML models
- `week3-optimization.ipynb` - Week 3: cross-validation, tuning, XGBoost, K-means segments, PCA
- `churn_model.joblib` - Week 3: final saved model (about 428 KB)
- Week 4: the deployed app lives in its own repo, `churn-risk-advisor` (see the Week 4 section below)
- `README.md` - this file

## Week 1: Exploratory Data Analysis
- Notebook: `week1-eda.ipynb`
- Overall churn rate is about 26.5%, so the target is imbalanced.
- Month-to-month contracts churn about 43%, versus about 11% (1-year) and about 3% (2-year).
- Churned customers have much lower tenure (median about 10 months vs about 38).
- Fiber optic customers churn about 42% vs about 19% for DSL; electronic check users churn the most (about 45%).

## Week 2: Building ML Models
- Notebook: `week2-ml-models.ipynb`
- Baseline (always "stay"): accuracy 0.735, catches 0 of 374 test churners
- Best model: Logistic Regression, AUC 0.842, recall 0.920 at threshold 0.15
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.15, because a missed churner (PKR 6,000) costs six times a wasted retention offer (PKR 1,000), so t* = 0.14; this cuts total cost from PKR 1,082,000 (at 0.5) to PKR 617,000
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 -> 0.8420 (no improvement)
- Biggest lesson: accuracy is misleading on imbalanced churn data, and a cost-based threshold mattered more than the choice of model (Logistic Regression and Random Forest both reach AUC 0.842).

### Model comparison (test set, 1,409 customers, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| LR balanced | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (d=5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | 0.842 |

## Week 3: Model Optimization and Unsupervised Learning
- Notebook: `week3-optimization.ipynb`
- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828 (std 0.0104, theoretical standard error 0.0107)
- 5-fold CV AUC: LR 0.846 +/- 0.013, RF 0.844 +/- 0.011, XGBoost (tuned) 0.850 +/- 0.012
- Tuning: best RF params (random search) max_depth 15, min_samples_leaf 15, max_features 0.21; grid vs random search time 73 s vs 84 s (both 120 fits, CV AUC 0.8468 vs 0.8464)
- Test AUC of final model (XGBoost, used once): 0.848 (recall 0.521, precision 0.659 at threshold 0.5), saved as `churn_model.joblib`
- Customer segments (k = 4): Fiber month-to-month (43% churn), New low spend (32%), Loyal premium bundle (14%), Loyal basic (5%)
- PCA: 15 of 30 components explain 90% of the variance; PC1 is mostly a "no internet service" axis (several one-hot columns are exact copies)
- Biggest lesson: a single split can move accuracy by about 5 points, and after careful tuning the three model families are statistically tied (CV AUC about 0.85), so the choice of threshold and the use of segments matter more than the choice of model.

## Week 4: From Notebook to Product
- Live app: PASTE-YOUR-STREAMLIT-URL-HERE (code and model card: https://github.com/Shoaibrhman/churn-risk-advisor)
- Notebook cells (end of `week3-optimization.ipynb`): skew demo (P(churn) 0.136 with the Week 3 encoder vs 0.012 with the serving encoder), parity test passed on 7,043 customers (largest difference 0.0), what-if table, `model_meta.json`
- Final model in the app: tuned XGBoost, CV AUC 0.850 +/- 0.012, test AUC 0.848, contact threshold 0.20 (cost analysis, FN = PKR 6,000, FP = PKR 1,000)
- Biggest lesson: a silent encoding bug changed one customer's probability from 0.012 to 0.136 without any error, so a parity test on every row is what proves that a deployed model gives the notebook's answers.

### Setup
Open the Kaggle notebooks, or run locally:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then download the dataset from Kaggle and update the file path in the notebook.
