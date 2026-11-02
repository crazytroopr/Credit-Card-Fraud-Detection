# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions on a highly imbalanced dataset. The project compares a logistic regression baseline against a neural network classifier and evaluates them with metrics that actually matter for rare-event detection (precision, recall, F1, AUC) instead of raw accuracy.

**Tech stack:** Python, scikit-learn, Pandas, NumPy, Matplotlib, imbalanced-learn (SMOTE) <!-- add PyTorch here only if the neural network is written in PyTorch -->

---

## Problem

Fraudulent transactions are a tiny fraction of all card activity, so a model that predicts "not fraud" for everything can still score over 99% accuracy while catching nothing. This project treats fraud detection as an imbalanced classification problem and focuses on:

- handling class imbalance properly (class weighting and SMOTE),
- choosing evaluation metrics that reflect the real cost of missed fraud and false alarms,
- comparing a simple, interpretable baseline with a more flexible model.

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Transactions made by European cardholders over two days in September 2013.
- About 284,807 transactions, of which only 492 (about 0.17%) are fraudulent.
- 30 input features: `V1`–`V28` (PCA-transformed for confidentiality), `Time`, and `Amount`.
- Target: `Class` (1 = fraud, 0 = legitimate).

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the `data/` folder.

## Approach

1. **Preprocessing:** scale `Time` and `Amount`, then split into train and test sets with stratification so both keep the same fraud ratio.
2. **Imbalance handling:** compare class-weighted training and SMOTE oversampling. SMOTE is applied to the training set only, never the test set, to avoid data leakage.
3. **Models:**
   - Logistic regression (baseline)
   - Neural network classifier
4. **Evaluation:** precision, recall, F1-score, ROC-AUC, ROC curves, and confusion matrices.

## Why not accuracy?

With about 0.17% fraud, accuracy is misleading. A model that flags nothing is 99.8% accurate and completely useless. Instead:

- **Recall** shows how much real fraud is caught.
- **Precision** shows how many flagged transactions are truly fraud (false alarms annoy customers).
- **F1** balances the two.
- **AUC / ROC curve** shows the tradeoff across decision thresholds.

## Results

> Fill this in with your own numbers from the final run. Do not leave placeholders in the published version.

| Model | Imbalance handling | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD | TBD |

Key takeaways: *(2–3 sentences on which model did better, what the tradeoff between precision and recall looked like, and what surprised you.)*

Plots (ROC curves and confusion matrices) are saved in the `results/` folder.

## Project structure

```
.
├── data/              # place creditcard.csv here (not tracked)
├── notebooks/         # exploration and experiments
├── src/               # preprocessing, training, evaluation code
├── results/           # plots and saved metrics
├── requirements.txt
└── README.md
```

*(Adjust this to match your actual repository layout.)*

## How to run

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
# download creditcard.csv from Kaggle into data/
python src/train.py        # or open the notebook in notebooks/
```

*(Replace the commands with the ones that actually work for your repo.)*

## Limitations and future work

- The PCA-transformed features can't be interpreted, which limits explainability.
- The data covers only two days in 2013, so the models may not generalize to current fraud patterns.
- Possible next steps: threshold tuning based on the cost of missed fraud, tree-based models such as XGBoost, SHAP explanations, and serving the model through an API.

## Author

Pardhu Thallapudi: B.Tech in Mathematics and Computing, IIIT Raichur
