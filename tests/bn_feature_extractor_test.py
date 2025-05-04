import pytest
import pandas as pd
import numpy as np
from applybn.feature_extraction import BNFeatureGenerator


class TestBNFeatureGenerator:
    """
    Tests for the BNFeatureGenerator class.

    This class contains unit tests to verify the functionality of the
    BNFeatureGenerator, including its initialization, fitting process with and
    without a target variable, and data transformation capabilities."""

    @pytest.fixture
    def sample_data(self):
        """
        Generates a sample Pandas DataFrame.

            Args:
                None

            Returns:
                pd.DataFrame: A DataFrame with columns 'A', 'B', and 'C' containing
                              sample data.
        """
        data = pd.DataFrame(
            {
                "A": np.array([1, 2, 3, 4, 5]),
                "B": np.array(["x", "y", "x", "y", "x"]),
                "C": np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            }
        )

        return data

    @pytest.fixture
    def setup_generator(self):
        """
        Sets up and returns a BNFeatureGenerator instance.

          This method instantiates a BNFeatureGenerator object, which is used for
          generating features. It does not take any input parameters.

          Returns:
            BNFeatureGenerator: A configured instance of the BNFeatureGenerator class.
        """
        generator = BNFeatureGenerator()
        return generator

    def test_initialization(self, setup_generator):
        """
        Asserts that the batch normalization layer is initially None.

          Args:
            setup_generator: A fixture providing a generator object for testing.

          Returns:
            None
        """
        assert setup_generator.bn is None

    def test_fit_without_target(self, setup_generator, sample_data):
        """
        Fits the generator without a target variable and asserts properties of the resulting Bayesian network.

            Args:
                setup_generator: The generator object to fit.
                sample_data: The data used for fitting.

            Returns:
                None
        """
        setup_generator.fit(sample_data)

        assert setup_generator.bn is not None
        assert set(list(map(str, setup_generator.bn.nodes))) == set(sample_data.columns)
        assert setup_generator.bn.nodes is not None

    def test_fit_with_target(self, setup_generator, sample_data):
        """
        Tests the fit method with a target variable.

            Args:
                setup_generator: The generator object to be fitted.
                sample_data: A DataFrame containing the data for fitting.

            Returns:
                None
        """
        target = sample_data["B"]
        target.name = "B"

        setup_generator.fit(sample_data.drop("B", axis=1), target)

        assert setup_generator.bn is not None

        expected_columns = set(sample_data.columns)
        actual_columns = set(list(map(str, setup_generator.bn.nodes)))
        assert actual_columns == expected_columns

    def test_transform(self, setup_generator, sample_data):
        """
        Tests the transform method of a generator.

            This method fits the generator to sample data and then transforms it,
            asserting that the transformed data is a Pandas DataFrame with the same
            number of columns as the original data, that all column names contain "lambda_",
            and that all values in the transformed dataframe are real numbers.

            Args:
                setup_generator: The generator object being tested.
                sample_data: The sample data used for fitting and transforming.

            Returns:
                None
        """
        setup_generator.fit(sample_data)

        transformed_data = setup_generator.transform(sample_data)

        assert isinstance(transformed_data, pd.DataFrame)

        assert len(transformed_data.columns) == len(sample_data.columns)
        assert all(["lambda_" in col for col in transformed_data.columns])

        assert transformed_data.map(np.isreal).all().all()
