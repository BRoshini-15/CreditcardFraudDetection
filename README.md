# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using multiple supervised classification algorithms, with a focus on handling severe class imbalance.

## Overview

Credit card fraud detection is a highly imbalanced classification problem — fraudulent transactions make up a tiny fraction of all transactions. This project builds and compares four ML classifiers to identify fraud while minimizing missed fraud cases, evaluated using metrics suited to imbalanced data (precision, recall, F1-score, AUC) rather than accuracy alone.

## Dataset

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Highly imbalanced — fraudulent transactions make up a tiny fraction of all transactions (~0.17–0.35% depending on the sample)
- Features are anonymized/PCA-transformed (`V1`–`V28`), plus `Time` and `Amount`

## Approach

1. **Data Cleaning** — removed rows with missing/corrupted `Class` labels before splitting.
2. **Preprocessing** — scaled `Amount` and `Time` with `StandardScaler` (the `V1`–`V28` features are already PCA-scaled).
3. **Class Imbalance Handling** — applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set only, to avoid leaking synthetic data into evaluation.
4. **Model Training** — trained and compared four classifiers on an identical train/test split:
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Support Vector Machine (SVM)
5. **Evaluation** — compared models using confusion matrices, precision, recall, F1-score, and AUC (not accuracy alone, given the class imbalance).

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 98.54% | 16.87% | 96.55% | 28.72% | **0.9854** |
| **KNN** | 99.80% | 60.87% | 96.55% | 74.67% | 0.9825 |
| **SVM** | 98.85% | 20.00% | 93.10% | 32.93% | 0.9571 |
| **Decision Tree** | 99.85% | 74.19% | 79.31% | **76.67%** | 0.8961 |

All four models were evaluated on the same held-out test set (29 actual fraud cases out of 9,526 transactions).

## Key Takeaways

- **Accuracy is misleading here.** Every model scores 98%+ accuracy simply because legitimate transactions dominate the dataset, yet a model predicting "legit" for everything would score just as high while catching zero fraud.

- **KNN and Logistic Regression tie for the best recall (96.55%)** — each misses only 1 of 29 fraud cases — but KNN does it with far fewer false alarms (60.87% precision vs. 16.87%), giving it the best F1-score (74.67%) among the high-recall models and the second-highest AUC (0.9825).

- **Decision Tree trades recall for precision.** It has the highest precision (74.19%) but the lowest recall and AUC of the four, missing 6 fraud cases.

- **Recommended model: KNN.** It combines top-tier fraud recall with far fewer false alarms than Logistic Regression or SVM, making it the most balanced choice for this use case.

- **SMOTE drives the higher recall** across models by synthetically balancing the training set, trading some precision for better fraud coverage — expected behavior, not a flaw.

## Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, imbalanced-learn (SMOTE), Pandas, NumPy, Matplotlib, Seaborn

## Project Structure

```
CreditcardFraudDetection/
├── credit_card_fraud_detection.py   # Main training & evaluation script
├── creditcard.csv                   # Dataset (not included — download from Kaggle)
├── model_comparison_results.csv     # Output: metrics for all models
├── confusion_matrix_*.png           # Output: confusion matrix per model
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/BRoshini-15/CreditcardFraudDetection.git
cd CreditcardFraudDetection

# Install dependencies
pip install -r requirements.txt

# Download creditcard.csv from Kaggle and place it in this folder:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Run the training/evaluation script
python credit_card_fraud_detection.py
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
