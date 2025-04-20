import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.exceptions import NotFittedError
from applybn.feature_selection.bn_nmi_feature_selector import NMIFeatureSelector

# Fixtures
@pytest.fixture
def classification_data():
    return make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        random_state=42
    )

@pytest.fixture
def regression_data():
    return make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        random_state=42
    )

@pytest.fixture
def synthetic_data():
    X = np.zeros((100, 4))
    y = np.random.randint(0, 2, 100)
    X[:, 0] = y
    X[:, 1] = y + np.random.normal(0, 0.1, 100)
    X[:, 2] = X[:, 0] * 2 + np.random.normal(0, 0.1, 100)
    X[:, 3] = np.random.normal(0, 1, 100)
    return pd.DataFrame(X, columns=['f0', 'f1', 'f2', 'f3']), y

# Basic functionality tests
def test_basic_functionality_classification(classification_data):
    X, y = classification_data
    selector = NMIFeatureSelector(threshold=0.01, n_bins=5)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    
    # Test DataFrame input
    selector.fit(X_df, y)
    X_transformed = selector.transform(X_df)
    assert X_transformed.shape[1] <= X.shape[1]
    assert len(selector.selected_features_) == X_transformed.shape[1]
    assert hasattr(selector, 'feature_names_in_')

    # Test array input
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert X_transformed.shape[1] <= X.shape[1]

def test_feature_selection_stages(synthetic_data):
    X, y = synthetic_data
    selector = NMIFeatureSelector(threshold=0.1, n_bins=2)
    selector.fit(X, y)
    
    first_stage_features = np.where(selector.nmi_features_target_ > 0.1)[0]
    assert len(first_stage_features) >= 3
    assert 2 in selector.selected_features_
    assert 0 in selector.selected_features_
    assert 3 not in selector.selected_features_

# Edge case tests
def test_all_features_below_threshold():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    selector = NMIFeatureSelector(threshold=0.5)
    selector.fit(X, y)
    assert len(selector.selected_features_) == 0

def test_redundant_features_removal():
    X = np.zeros((100, 3))
    X[:, 0] = y = np.random.randint(0, 2, 100)
    X[:, 1] = X[:, 0]
    X[:, 2] = X[:, 0]
    selector = NMIFeatureSelector(threshold=0.01)
    selector.fit(X, y)
    assert len(selector.selected_features_) == 3

# Attribute tests
def test_attributes_exist(classification_data):
    X, y = classification_data
    selector = NMIFeatureSelector().fit(X, y)
    assert hasattr(selector, 'nmi_features_target_')
    assert hasattr(selector, 'selected_features_')
    assert hasattr(selector, 'selected_mask_')
    assert selector.selected_mask_.sum() == len(selector.selected_features_)

# Reproducibility test
def test_reproducibility(classification_data):
    X, y = classification_data
    selector1 = NMIFeatureSelector().fit(X, y)
    selector2 = NMIFeatureSelector().fit(X, y)
    np.testing.assert_array_equal(selector1.selected_features_,
                                 selector2.selected_features_)


# Regression test
def test_regression_support(regression_data):
    X, y = regression_data
    selector = NMIFeatureSelector(n_bins=5)
    selector.fit(X, y)
    assert len(selector.selected_features_) > 0
    assert X.shape[1] >= selector.transform(X).shape[1]

