import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def imbalanced_data():
    """
    Generates a Pandas DataFrame with an imbalanced class distribution.

        This method creates a DataFrame containing three features and a binary
        classification target variable ('class') where the negative class (0) is
        significantly more represented than the positive class (1).

        Returns:
            pd.DataFrame: A DataFrame with 100 samples, three features ("feature1",
                          "feature2", "feature3"), and a binary target variable
                          ("class") exhibiting a 90:10 imbalance.
    """
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.randint(0, 5, 100),
            "feature3": 2 * np.random.normal(0, 1, 100),
            "class": [0] * 90 + [1] * 10,  # 90:10 imbalance
        }
    )
    return data


@pytest.fixture
def mixed_type_data():
    """
    Creates a Pandas DataFrame with mixed data types.

        This method generates a DataFrame containing numeric, categorical, and
        integer columns using randomly generated data.

        Parameters:
            None

        Returns:
            pd.DataFrame: A DataFrame with 50 rows and three columns: 'numeric',
                          'categorical', and 'class'.  The 'numeric' column contains
                          random floating-point numbers, 'categorical' contains
                          repeated "A" and "B" values, and 'class' contains 30 zeros
                          followed by 20 ones.
    """
    data = pd.DataFrame(
        {
            "numeric": np.random.rand(50),
            "categorical": ["A", "B"] * 25,
            "class": [0] * 30 + [1] * 20,
        }
    )
    return data
