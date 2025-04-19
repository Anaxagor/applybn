# Example: Handling Severe Class Imbalance with BNOverSampler and SMOTE

This example demonstrates how to handle severe class imbalance using two oversampling techniques:  
**BNOverSampler** (a Bayesian Network-based method) and **SMOTE**. We compare their performance on a marketing dataset using F1-score and AUC-ROC metrics.

---

## Dependencies
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from applybn.imbalanced.over_sampling import BNOverSampler  # Custom package
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
```

## Step 1: Load Data and Create Imbalance
We use the bank-marketing dataset and artificially create a 30:1 class ratio to simulate severe imbalance:
```python
data = pd.read_csv('data/bn_oversampling/bank-marketing.csv', index_col=[0])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create 5:1 imbalance
_, X_min, _, y_min = train_test_split(
    X[y == 1], y[y == 1], 
    test_size=0.2, 
    random_state=42
)
X_imb = pd.concat([X[y == 0], X_min])
y_imb = pd.concat([y[y == 0], y_min])
```
## Step 2: Train-Test Split
Split data into 70% training and 30% testing, preserving stratification:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, 
    test_size=0.3, 
    stratify=y_imb,
    random_state=42
)
```
## Step 3: Initialize Resamplers
Configure oversampling methods:
```python
bn_sampler = BNOverSampler(class_column='class', strategy='max_class')  # Balances to majority class size
smote = SMOTE(k_neighbors=5, random_state=42)
```
## Step 4: Resample Training Data
Apply both methods and measure execution time:
```python
# BNOverSampler
start_bn = time.time()
X_bn, y_bn = bn_sampler.fit_resample(X_train, y_train)
time_bn = time.time() - start_bn

# SMOTE
start_smote = time.time()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
time_smote = time.time() - start_smote
```
## Step 5: Train and Evaluate Models
Use a **RandomForestClassifier** for both resampled datasets:
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_bn, y_bn)
bn_pred = clf.predict(X_test)
bn_proba = clf.predict_proba(X_test)[:, 1]

clf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
clf_smote.fit(X_smote, y_smote)
smote_pred = clf_smote.predict(X_test)
smote_proba = clf_smote.predict_proba(X_test)[:, 1]
```
## Step 6: Results
| Method          | F1-Score | AUC-ROC | Time (s) |
|-----------------|----------|---------|----------|
| BNOverSampler   | 0.575    | 0.858   | 35.1     |
| SMOTE           | 0.556    | 0.844   | 0.0      |

Use BNOverSampler when interpretability and distribution preservation are critical, and computational resources are sufficient. 