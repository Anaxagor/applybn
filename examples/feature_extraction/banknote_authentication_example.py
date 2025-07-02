import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator
import ssl
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context

# Load the banknote-authentication dataset
print("Loading banknote-authentication dataset...")
data = fetch_openml(name="banknote-authentication", version=1, as_frame=True)
X = pd.DataFrame(data.data)
y = pd.Series(data.target, name="target")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Reset indices to ensure proper alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# First approach: Using only original features
print("\n1. Training Decision Tree with original features...")
dt_original = DecisionTreeClassifier(random_state=42)
dt_original.fit(X_train, y_train)
y_pred_original = dt_original.predict(X_test)

# Second approach: Using Bayesian Network features
print("\n2. Generating Bayesian Network features...")
bn_feature_generator = BNFeatureGenerator()
bn_feature_generator.fit(X=X_train, y=y_train)  # Fit the generator with training data

# Transform both training and testing data
X_train_bn = bn_feature_generator.transform(X_train).reset_index(drop=True)
X_test_bn = bn_feature_generator.transform(X_test).reset_index(drop=True)


# Train with combined features
print("\nTraining Decision Tree with combined features...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_bn, y_train)
y_pred = dt.predict(X_test_bn)
print("\nClassification Report with Original Features:")
print(classification_report(y_test, y_pred_original))
print("\nClassification Report with Combined Features:")
print(classification_report(y_test, y_pred))
