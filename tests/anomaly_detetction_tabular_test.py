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
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "anomaly": [0, 1, 0]})


@pytest.fixture
def real_data():
    df = pd.read_csv("test_data/healthcare.csv", index_col=0)
    df["anomaly"] = np.random.choice([0, 1], size=df.shape[0], p=[0.6, 0.4])
    return df


def test_fit_with_missing_target_column_raises_key_error(dummy_data):
    detector = TabularDetector(target_name="missing_column")
    with pytest.raises(KeyError):
        detector.fit(dummy_data)


def test_decision_function_without_fitting_raises_not_fitted_error(dummy_data):
    detector = TabularDetector(target_name="anomaly")
    with pytest.raises(NotFittedError):
        detector.decision_function(dummy_data)


def test_predict_with_supervised_thresholding(dummy_data):
    detector = TabularDetector(target_name="anomaly", verbose=0)
    detector.fit(dummy_data)
    predictions = detector.predict(dummy_data)
    assert predictions.shape[0] == dummy_data.shape[0]


def test_predict_without_supervised_thresholding_raises_not_implemented_error(
    dummy_data,
):
    detector = TabularDetector(verbose=0)
    detector.fit(dummy_data.drop("anomaly", axis=1))
    with pytest.raises(NotImplementedError):
        detector.predict(dummy_data)


def test_construct_score_with_invalid_score_type_raises_key_error():
    detector = TabularDetector(target_name="anomaly", score="invalid_score")
    with pytest.raises(KeyError):
        detector.construct_score()


def test_tabular_detector_mixed_data(real_data):
    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="original_modified",
    )
    detector.fit(real_data)

    predictions = detector.predict(real_data)
    assert predictions.shape[0] == real_data.shape[0]


def test_tabular_detector_wrong_method(real_data):
    detector = TabularDetector(
        verbose=0,
        target_name="anomaly",
        model_estimation_method="iqr",
    )
    detector.fit(real_data)

    with pytest.raises(TypeError):
        detector.predict(real_data)


def test_tabular_detector_on_cont_data(real_data):
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
