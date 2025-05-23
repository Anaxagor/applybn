import logging

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
)

from applybn.explainable.causal_analysis import InterventionCausalExplainer


# Example data loading function (replace with actual data)
def load_data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Load data
X_train, X_test, y_train, y_test = load_data()

# Initialize and run ModelInterpreter
interpreter = InterventionCausalExplainer()
interpreter.interpret(RandomForestClassifier(), X_train, y_train, X_test, y_test)
