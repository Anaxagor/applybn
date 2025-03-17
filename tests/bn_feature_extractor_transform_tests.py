import pytest
import pandas as pd
import numpy as np
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator


@pytest.fixture
def sample_data():
    """
Creates a Pandas DataFrame with sample data.

    Args:
        None

    Returns:
        pd.DataFrame: A DataFrame containing three columns ('A', 'B', 'C') 
                      with binary values (0 or 1).
    """
    return pd.DataFrame(
        {
            "A": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            "B": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "C": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def sample_target():
    """
Returns a Pandas Series representing a sample target variable.

    This method creates and returns a Pandas Series containing a sequence of 0s and 1s,
    intended to serve as a simple example target variable for demonstration or testing purposes.

    Returns:
        pd.Series: A Pandas Series with the values [0, 1, 0, 1, 1, 0, 1, 0, 1, 1].
    """
    return pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])


def test_transform(sample_data, sample_target):
    """
Tests the transform method of BNFeatureGenerator.

    Args:
        sample_data: The input data for transformation.
        sample_target: The target variable used during fitting.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)
    features = generator.transform(sample_data)

    assert features.shape == (10, 3)
    assert list(features.columns) == ["lambda_A", "lambda_B", "lambda_C"]
    assert (features >= 0).all().all() and (features <= 1).all().all()


def test_transform_without_target(sample_data):
    """
Tests the transform method of BNFeatureGenerator without a target variable.

    Args:
        sample_data: The input data for transformation.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    generator.fit(sample_data)
    features = generator.transform(sample_data)

    assert features.shape == (10, 3)
    assert list(features.columns) == ["lambda_A", "lambda_B", "lambda_C"]
    assert (features >= 0).all().all() and (features <= 1).all().all()


def test_transform_with_missing_feature(sample_data, sample_target):
    """
Tests the transform method with a missing feature.

    Args:
        sample_data: The input data for fitting and transforming.
        sample_target: The target variable used during fitting.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)

    new_sample = sample_data.copy()
    new_sample.loc[0, "A"] = np.nan

    features = generator.transform(new_sample)
    assert features.shape == (10, 3)
    assert not np.isnan(features.iloc[0, 0])
