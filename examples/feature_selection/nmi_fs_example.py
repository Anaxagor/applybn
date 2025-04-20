import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from applybn.feature_selection.bn_nmi_feature_selector import NMIFeatureSelector

# Load dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and apply feature selector
selector = NMIFeatureSelector(threshold=0.5, n_bins=20)
selector.fit(X_train, y_train)  # Fixed version captures feature names before conversion

# Transform datasets
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = selector.feature_names_in_[selector.selected_features_]
print(f"Selected features ({len(selected_features)}):\n{selected_features}")

# Train classifier and compare results
clf_full = RandomForestClassifier(random_state=42)
clf_full.fit(X_train, y_train)
acc_full = clf_full.score(X_test, y_test)

clf_selected = RandomForestClassifier(random_state=42)
clf_selected.fit(X_train_selected, y_train)
acc_selected = clf_selected.score(X_test_selected, y_test)

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")
print(f"Accuracy with all features: {acc_full:.4f}")
print(f"Accuracy with selected features: {acc_selected:.4f}")