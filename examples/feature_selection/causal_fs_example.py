# Example: Causal Feature Selection on High-Dimensional Synthetic Data
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector

# Generate synthetic dataset with 1000 features (50 informative, 950 noise)
X, y = make_classification(
    n_samples=2000,
    n_features=50,
    n_informative=5,
    n_redundant=10
)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize causal feature selector
causal_selector = CausalFeatureSelector(n_bins=10)

# Fit on training data and transform features
causal_selector.fit(X_train, y_train)
X_train_selected = causal_selector.transform(X_train)
X_test_selected = causal_selector.transform(X_test)

# Verify feature reduction
print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")

# Train classifier on full features
clf_full = LogisticRegression(max_iter=1000, random_state=42)
clf_full.fit(X_train, y_train)
y_pred_full = clf_full.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)

# Train classifier on causal-selected features
clf_selected = LogisticRegression(max_iter=1000, random_state=42)
clf_selected.fit(X_train_selected, y_train)
y_pred_selected = clf_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)

# Compare performance
print(f"\nAccuracy with all features: {accuracy_full:.4f}")
print(f"Accuracy with causal features: {accuracy_selected:.4f}")

# Show mask of selected features
print("\nSelected feature indices:", np.where(causal_selector.support_)[0])