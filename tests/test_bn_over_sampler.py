import pytest
import pandas as pd
import numpy as np
from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler


def test_sampler_initialization():
    sampler = BNOverSampler(class_column='target', strategy=100, shuffle=False)
    assert sampler.class_column == 'target'
    assert sampler.strategy == 100
    assert sampler.shuffle is False


def test_basic_oversampling(imbalanced_data):
    sampler = BNOverSampler(class_column='class', shuffle=True)
    X = imbalanced_data.drop(columns='class')
    y = imbalanced_data['class']
    X_res, y_res = sampler._fit_resample(X, y)
    
    # Check basic output shape
    assert len(X_res) == 180  # 90*2 classes
    assert len(y_res) == 180
    assert X_res.columns.tolist() == ['feature1', 'feature2', 'feature3']
    
    # Check class balance
    class_counts = y_res.value_counts()
    assert class_counts[0] == 90
    assert class_counts[1] == 90

def test_custom_strategy(imbalanced_data):
    sampler = BNOverSampler(strategy=150, class_column='class')
    X = imbalanced_data.drop(columns='class')
    y = imbalanced_data['class']
    
    X_res, y_res = sampler._fit_resample(X, y)
    
    # Check total samples (original 100 + synthetic 140)
    assert len(X_res) == 300  # 300 total
    assert y_res.value_counts()[1] == 150

def test_mixed_data_types(mixed_type_data):
    sampler = BNOverSampler(class_column='class')
    X = mixed_type_data.drop(columns='class')
    y = mixed_type_data['class']
    
    X_res, y_res = sampler._fit_resample(X, y)
    
    # Check data types preservation
    assert pd.api.types.is_numeric_dtype(X_res['numeric'])
    assert pd.api.types.is_object_dtype(X_res['categorical'])
    
    