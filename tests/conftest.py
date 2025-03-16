import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def imbalanced_data():
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.randint(0, 5, 100),
        'feature3': 2*np.random.normal(0, 1, 100),
        'class': [0]*90 + [1]*10  # 90:10 imbalance
    })
    return data

@pytest.fixture
def mixed_type_data():
    data = pd.DataFrame({
        'numeric': np.random.rand(50),
        'categorical': ['A', 'B'] * 25,
        'class': [0]*30 + [1]*20
    })
    return data