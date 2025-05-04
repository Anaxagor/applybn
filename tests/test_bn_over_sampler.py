import pytest
import pandas as pd
import numpy as np
from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler


def test_sampler_initialization():
    """
    Tests the initialization of the BNOverSampler class.

        This test verifies that the class attributes are correctly set
        during object creation with specified parameters.

        Parameters:
            None

        Returns:
            None
    """
    sampler = BNOverSampler(class_column="target", strategy=100, shuffle=False)
    assert sampler.class_column == "target"
    assert sampler.strategy == 100
    assert sampler.shuffle is False


def test_basic_oversampling(imbalanced_data):
    """
    Tests basic oversampling functionality.

        This function tests that the BNOverSampler correctly resamples an imbalanced dataset,
        resulting in a balanced dataset with the expected shape and class distribution.

        Args:
            imbalanced_data: The imbalanced DataFrame to be resampled.  Must contain a
                             'class' column indicating the target variable and 'feature1',
                             'feature2', and 'feature3' columns for features.

        Returns:
            None: This function asserts conditions based on the resampling result, but does not return a value.
    """
    sampler = BNOverSampler(class_column="class", shuffle=True)
    X = imbalanced_data.drop(columns="class")
    y = imbalanced_data["class"]
    X_res, y_res = sampler.fit_resample(X, y)

    # Check basic output shape
    assert len(X_res) == 180  # 90*2 classes
    assert len(y_res) == 180
    assert X_res.columns.tolist() == ["feature1", "feature2", "feature3"]

    # Check class balance
    class_counts = y_res.value_counts()
    assert class_counts[0] == 90
    assert class_counts[1] == 90


def test_custom_strategy(imbalanced_data):
    """
    Tests the BNOverSampler with a custom strategy.

        Args:
            imbalanced_data: The imbalanced dataset to be resampled.

        Returns:
            None: This function asserts conditions based on the resampling result and does not return a value.
    """
    sampler = BNOverSampler(strategy=150, class_column="class")
    X = imbalanced_data.drop(columns="class")
    y = imbalanced_data["class"]

    X_res, y_res = sampler.fit_resample(X, y)

    # Check total samples (original 100 + synthetic 200; 150 for each class)
    assert len(X_res) == 300  # 300 total
    assert y_res.value_counts()[1] == 150
