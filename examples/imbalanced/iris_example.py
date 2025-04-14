"""Bayesian Network Oversampling Example with Iris Dataset.

This example demonstrates using BNOverSampler to address class imbalance while preserving feature
relationships through Bayesian network modeling. Includes artificial imbalance creation, resampling,
and logging integration.

Attributes:
    logger: Configured logger instance.
    oversampler: Bayesian network-based resampler.
    X_res: Resampled feature matrix.
    y_res: Balanced target vector.

Notes:
    - Requires applybn package components
    - Uses sklearn's Iris dataset as base data
    - Implements custom logging configuration
    - Preserves original data structure and types
"""

from sklearn.datasets import load_iris
import pandas as pd
import logging
from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler
from applybn.core.logger import Logger

def create_imbalanced_data(X: pd.DataFrame, y: pd.DataFrame, ratios: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create artificial class imbalance by subsampling majority classes
    
    Args:
        X: Original feature matrix
        y: Original target vector
        ratios: Dictionary mapping class labels to desired sample counts
        
    Returns:
        tuple with specified class distribution
 
    """
    imbalanced_X = pd.DataFrame()
    imbalanced_y = pd.Series(dtype=y.dtype)
    
    for cls, n_samples in ratios.items():
        cls_X = X[y == cls].sample(n=n_samples, random_state=42)
        cls_y = y[y == cls].sample(n=n_samples, random_state=42)
        
        imbalanced_X = pd.concat([imbalanced_X, cls_X], ignore_index=True)
        imbalanced_y = pd.concat([imbalanced_y, cls_y], ignore_index=True)

    return imbalanced_X, imbalanced_y

if __name__ == "__main__":
    # Configure logging using updated Logger class
    logger = Logger("bn_oversample_demo", level=logging.INFO)
    logger.info("Initializing BN oversampling demonstration")

    # Load and prepare data
    iris = load_iris()
    X = pd.DataFrame(
        iris.data,
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )
    y = pd.Series(iris.target, name='species')
    logger.debug("Loaded base Iris dataset with %d samples", len(X))

    # Create artificial imbalance
    imbalance_ratios = {0: 5, 1: 10, 2: 30}  # Class: sample_count
    imbalanced_X, imbalanced_y = create_imbalanced_data(X, y, imbalance_ratios)
    logger.info("Created imbalanced dataset:\n%s", 
               imbalanced_y.value_counts().sort_index().to_string())

    # Initialize and apply oversampler
    oversampler = BNOverSampler(
        class_column='species',
        strategy='max_class',
        shuffle=True
    )
    logger.debug("Initialized BNOverSampler with parameters: %s", vars(oversampler))

    # Perform resampling
    try:
        X_res, y_res = oversampler.fit_resample(imbalanced_X, imbalanced_y)
        logger.info("Resampling completed successfully")
    except Exception as e:
        logger.error("Resampling failed: %s", str(e))
        raise

    # Log final results
    logger.info("Resampled class distribution:\n%s", 
               y_res.value_counts().sort_index().to_string())
    logger.info("Total samples after resampling: %d", len(X_res))
    logger.debug("Final feature matrix shape: %s", str(X_res.shape))