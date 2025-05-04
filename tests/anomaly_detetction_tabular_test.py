import pytest

from sklearn.exceptions import NotFittedError
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import (
    TabularDetector,
)
from bamt.log import logger_preprocessor

import pandas as pd
import numpy as np

logger_preprocessor.disabled = True


@pytest.fixture
def dummy_data():
    """
    Creates a sample Pandas DataFrame for testing.

        This method generates a small DataFrame with three columns: 'col1', 'col2', and 'anomaly'.
        The 'anomaly' column contains binary values indicating potential anomalies.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing dummy data.
    """
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "anomaly": [0, 1, 0]})


@pytest.fixture
def real_data():
    """
    Loads healthcare data from a CSV, adds an anomaly column.

        This method reads the 'healthcare.csv' file located in the 'tests/test_data/' directory
        into a Pandas DataFrame, and then adds a new column named "anomaly" containing randomly
        assigned binary values (0 or 1) with probabilities of 0.6 and 0.4 respectively.

        Args:
            None

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the healthcare data with an added 'anomaly' column.
    """
    df = pd.read_csv("tests/test_data/healthcare.csv", index_col=0)
    df["anomaly"] = np.random.choice([0, 1], size=df.shape[0], p=[0.6, 0.4])
    return df


def test_fit_with_missing_target_column_raises_key_error(dummy_data):
    """
    Tests that fitting the detector with a missing target column raises a KeyError.

        Args:
            dummy_data: A dataframe to use for testing.

        Returns:
            None
    """
    detector = TabularDetector(target_name="missing_column")
    with pytest.raises(KeyError):
        detector.fit(dummy_data)


def test_decision_function_without_fitting_raises_not_fitted_error(dummy_data):
    """
    Tests that calling decision_function before fitting raises a NotFittedError.

        Args:
            dummy_data: Data used for testing the decision function.

        Returns:
            None
    """
    detector = TabularDetector(target_name="anomaly")
    with pytest.raises(NotFittedError):
        detector.decision_function(dummy_data)


def test_predict_with_supervised_thresholding(dummy_data):
    """
    Tests the predict method with supervised thresholding.

        Args:
            dummy_data: The input data for testing.

        Returns:
            None: This function asserts a condition and does not return a value.
    """
    detector = TabularDetector(target_name="anomaly", verbose=0)
    detector.fit(dummy_data)
    predictions = detector.predict(dummy_data)
    assert predictions.shape[0] == dummy_data.shape[0]


def test_predict_without_supervised_thresholding_raises_not_implemented_error(
    dummy_data,
):
    """
    Tests that predict raises NotImplementedError when supervised thresholding is not implemented.

        Args:
            dummy_data: The input data for testing.

        Returns:
            None
    """
    detector = TabularDetector(verbose=0)
    detector.fit(dummy_data.drop("anomaly", axis=1))
    with pytest.raises(NotImplementedError):
        detector.predict(dummy_data)


def test_construct_score_with_invalid_score_type_raises_key_error():
    """
    Tests that constructing a score with an invalid type raises a KeyError.

        Args:
            None

        Returns:
            None
    """
    detector = TabularDetector(target_name="anomaly", score="invalid_score")
    with pytest.raises(KeyError):
        detector.construct_score()


def test_tabular_detector_mixed_data(real_data):
    """
    Tests the TabularDetector with mixed data.

        Args:
            real_data: The input dataframe for testing.

        Returns:
            None: This function asserts a condition and does not return a value.
    """
    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="original_modified",
    )
    detector.fit(real_data)

    predictions = detector.predict(real_data)
    assert predictions.shape[0] == real_data.shape[0]


def test_tabular_detector_wrong_method(real_data):
    """
    Tests that TabularDetector raises TypeError when predict is called.

        This test verifies the expected behavior of the TabularDetector class
        when an incorrect method (predict) is called after fitting with a method
        that doesn't support prediction.

        Args:
            real_data: The input data for testing.

        Returns:
            None: This function does not return a value; it asserts that a TypeError is raised.
    """
    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="iqr",
    )
    detector.fit(real_data)

    with pytest.raises(TypeError):
        detector.predict(real_data)


def test_tabular_detector_on_cont_data(real_data):
    """
    Tests the TabularDetector on continuous data.

        Args:
            real_data: The input DataFrame containing both numerical features and an anomaly column.

        Returns:
            None: This function asserts that the prediction shape matches the original data shape,
                  but does not explicitly return a value.
    """
    data = real_data.select_dtypes(include=[np.number])
    data["anomaly"] = real_data["anomaly"].values

    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="iqr",
    )

    detector.fit(data)
    preds = detector.predict(data)
    assert preds.shape[0] == real_data.shape[0]


def test_tabular_detector_on_cont_data_wrong(real_data):
    """
    Tests TabularDetector on continuous data when it expects categorical.

        This test checks that a TypeError is raised when attempting to predict
        with continuous features after fitting the detector, as it's designed for
        categorical anomaly detection.

        Args:
            real_data: The input DataFrame containing both numerical and anomaly columns.

        Returns:
            None
    """
    data = real_data.select_dtypes(include=[np.number])
    data["anomaly"] = real_data["anomaly"].values

    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="cond_ratio",
    )

    detector.fit(data)
    with pytest.raises(TypeError):
        detector.predict(data)


def test_tabular_detector_on_disc_data(real_data):
    """
    Tests the TabularDetector on discrete data.

        Args:
            real_data: The input DataFrame containing both categorical features and an 'anomaly' column.

        Returns:
            None: This function asserts that the prediction shape matches the input data shape,
                  but does not explicitly return a value.
    """
    data = real_data.select_dtypes(exclude=[np.number])
    data["anomaly"] = real_data["anomaly"].values

    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="cond_ratio",
        additional_score=None,
    )

    detector.fit(data)
    preds = detector.predict(data)
    assert preds.shape[0] == real_data.shape[0]


def test_tabular_detector_on_disc_data_wrong(real_data):
    """
    Tests TabularDetector raises TypeError when predicting on non-numeric data.

        This test checks that the TabularDetector correctly raises a TypeError
        when attempting to predict anomalies on a dataset containing only
        non-numeric features, even if the target variable is included.

        Args:
            real_data: The input DataFrame containing both numeric and non-numeric data,
                including an 'anomaly' column.

        Returns:
            None: This function does not return a value; it asserts that a TypeError
                is raised during prediction.
    """
    data = real_data.select_dtypes(exclude=[np.number])
    data["anomaly"] = real_data["anomaly"].values

    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="iqr",
        additional_score=None,
    )

    detector.fit(data)
    with pytest.raises(TypeError):
        detector.predict(data)
