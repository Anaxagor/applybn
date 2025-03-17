import pytest
import pandas as pd
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
Generates a Pandas Series representing a sample target variable.

    Args:
        None

    Returns:
        pd.Series: A Pandas Series containing the values [0, 1, 0, 1, 1, 0, 1, 0, 1, 1].
    """
    return pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])


@pytest.fixture
def known_structure():
    """
Returns a predefined structure of paired elements.

    This method provides a hardcoded list of tuples, where each tuple 
    represents a relationship between two elements.

    Returns:
        list: A list of tuples representing the known structure.  Each tuple contains two string elements.
    """
    return [("A", "B"), ("B", "C")]


def test_fit_without_target(sample_data):
    """
Tests the fit method without a target variable.

    Args:
        sample_data: The input data for fitting.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    generator.fit(sample_data)
    assert generator.variables == ["A", "B", "C"]
    assert generator.num_classes is None
    assert generator.bn is not None


def test_fit_with_target(sample_data, sample_target):
    """
Tests the fit method with a target variable.

    Args:
        sample_data: The input data for fitting.
        sample_target: The target variable for fitting.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    generator.fit(sample_data, y=sample_target)
    assert generator.num_classes == 2
    assert generator.bn is not None


def test_fit_with_known_structure(sample_data, known_structure):
    """
Tests fitting the feature generator with a known structure.

    Args:
        sample_data: The data used to fit the generator.
        known_structure: The expected Bayesian network structure.

    Returns:
        None
    """
    generator = BNFeatureGenerator(known_structure)
    generator.fit(sample_data)
    assert set(generator.bn.edges()) == set(known_structure)


def test_fit_with_black_list(sample_data):
    """
Tests that the fit method respects the black list of edges.

    Args:
        sample_data: The input data for fitting.

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    black_list = [("A", "C")]
    generator.fit(sample_data, black_list=black_list)
    assert ("A", "C") not in generator.bn.edges()
