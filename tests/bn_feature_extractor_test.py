import pytest
import pandas as pd
import numpy as np
from applybn.feature_extraction import BNFeatureGenerator

class TestBNFeatureGenerator:

    @pytest.fixture
    def sample_data(self):
        data = pd.DataFrame({
            'A': np.array([1, 2, 3, 4, 5]),
            'B': np.array(['x', 'y', 'x', 'y', 'x']),
            'C': np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        })

        return data

    @pytest.fixture
    def setup_generator(self):
        generator = BNFeatureGenerator()
        return generator

    def test_initialization(self, setup_generator):
        assert setup_generator.bn is None

    def test_fit_without_target(self, setup_generator, sample_data):
        setup_generator.fit(sample_data)

        assert setup_generator.bn is not None
        assert set(list(map(str, setup_generator.bn.nodes))) == set(sample_data.columns)
        assert setup_generator.bn.nodes is not None

    def test_fit_with_target(self, setup_generator, sample_data):
        target = sample_data['B']
        target.name = 'B'

        setup_generator.fit(sample_data.drop('B', axis=1), target)

        assert setup_generator.bn is not None

        expected_columns = set(sample_data.columns)
        actual_columns = set(list(map(str, setup_generator.bn.nodes)))
        assert actual_columns == expected_columns

    def test_transform(self, setup_generator, sample_data):
        setup_generator.fit(sample_data)

        transformed_data = setup_generator.transform(sample_data)

        assert isinstance(transformed_data, pd.DataFrame)

        assert len(transformed_data.columns) == len(sample_data.columns)
        assert all(['lambda_' in col for col in transformed_data.columns])

        assert transformed_data.map(np.isreal).all().all()
