import pytest
import pandas as pd
from applybn.feature_extraction.bn_feature_extractor import BNFeatureGenerator
import numpy as np

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'A': np.array([1, 2, 3, 4, 5]),
        'B': np.array(['x', 'y', 'x', 'y', 'x']),
        'C': np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    })

    return data

@pytest.fixture
def setup_generator():
    generator = BNFeatureGenerator()
    return generator

def test_initialization(setup_generator):
    assert setup_generator.bn is None
    assert setup_generator.variables is None
    assert setup_generator.nodes_dict == {}
    assert setup_generator.target_name == ''

def test_fit_without_target(setup_generator, sample_data):
    setup_generator.fit(sample_data)

    assert setup_generator.bn is not None
    assert set(setup_generator.variables) == set(sample_data.columns)
    assert setup_generator.nodes_dict is not None


def test_fit_with_target(setup_generator, sample_data):
    target = sample_data['B']
    target.name = 'B'

    setup_generator.fit(sample_data.drop('B', axis=1), target)

    assert setup_generator.bn is not None
    assert setup_generator.target_name == 'B'

    expected_columns = set(sample_data.columns)
    actual_columns = set(setup_generator.variables)
    assert actual_columns == expected_columns



def test_transform(setup_generator, sample_data):
    setup_generator.fit(sample_data)

    transformed_data = setup_generator.transform(sample_data)

    assert isinstance(transformed_data, pd.DataFrame)

    assert len(transformed_data.columns) == len(sample_data.columns)
    assert all(['lambda_' in col for col in transformed_data.columns])

    assert transformed_data.applymap(np.isreal).all().all()
