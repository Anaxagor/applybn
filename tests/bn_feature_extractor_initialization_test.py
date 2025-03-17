import pytest
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator


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


def test_initialization(known_structure):
    """
Tests the initialization of the BNFeatureGenerator class.

    Args:
        known_structure: The known structure to initialize with (or None).

    Returns:
        None
    """
    generator = BNFeatureGenerator()
    assert generator.known_structure is None
    assert generator.bn is None
    assert generator.variables is None
    assert generator.num_classes is None

    generator_with_structure = BNFeatureGenerator(known_structure)
    assert generator_with_structure.known_structure == known_structure
