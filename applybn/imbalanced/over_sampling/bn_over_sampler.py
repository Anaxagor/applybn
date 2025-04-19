from imblearn.over_sampling.base import BaseOverSampler
from bamt.preprocessors import Preprocessor
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
import pandas as pd
from sklearn.exceptions import NotFittedError
from applybn.core.estimators.base_estimator import BNEstimator
import numpy as np
from typing import Any


class BNOverSampler(BaseOverSampler):
    """A Bayesian Network-based oversampler for handling imbalanced datasets.

    This class uses Bayesian Networks to learn the joint probability distribution of features
    and generates synthetic samples for minority classes to balance class distribution.
    Inherits from BaseOverSampler to ensure compatibility with scikit-learn pipelines.

    Args:
        class_column: Name of the target class column. If None, will attempt to infer from y's
            name attribute.
        strategy: Oversampling strategy. Either 'max_class' to match the largest class size or
            an integer specifying target sample count per class.
        shuffle: Whether to shuffle the dataset after resampling.

    Attributes:
        data_generator_: Fitted Bayesian Network synthetic data generator instance.

    Example:
        >>> from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler
        >>> oversampler = BNOverSampler(class_column='target', strategy='max_class')
        >>> X_res, y_res = oversampler.fit_resample(X, y)
    """

    def __init__(self, class_column=None, strategy='max_class', shuffle=True):
        """Initialize the BNOverSampler."""
        super().__init__()
        self.class_column = class_column
        self.strategy = strategy
        self.shuffle = shuffle
        self.data_generator_ = BNEstimator()


    def _generate_samples_for_class(self, cls: str|int, needed: int, data_columns: list, types_dict: dict) -> pd.DataFrame:
        """Generate synthetic samples for a specific minority class.

        Args:
            cls: Target class value to generate samples for.
            needed: Number of synthetic samples needed for this class.
            data_columns: List of column names in the original dataset.
            types_dict: Dictionary mapping columns to their data types 
                (e.g., 'disc_num' for discrete numeric).

        Returns:
            samples: Generated samples with proper data types.
        """
        samples = self.data_generator_.sample(
            needed, 
            evidence={self.class_column: cls}, 
            filter_neg=False
        )[data_columns]
        if samples.shape[0] < needed:
            additional = self.data_generator_.sample(needed, evidence={self.class_column: cls}, filter_neg=False)[data_columns]
            samples = pd.concat([samples, additional.sample(needed - samples.shape[0])])
        return self._adjust_sample_types(samples, types_dict)
    
    def _adjust_sample_types(self, samples: pd.DataFrame, types_dict: dict) -> pd.DataFrame:
        """Adjust data types of generated samples to match original data.
        
        Args:
            samples: Generated synthetic samples.
            types_dict: Dictionary mapping columns to their data types.

        Returns:
            samples: Samples with corrected data types.
        """
        disc_num_cols = {col for col, dtype in types_dict.items() if dtype == 'disc_num'}
        samples = samples.apply(
            lambda col: col.astype(int) 
            if col.name in disc_num_cols 
            else col
        )
        return samples
        
    def _balance_classes(self, data: pd.DataFrame, class_counts: pd.Series, target_size: int) -> pd.DataFrame:
        """Generate synthetic samples to balance class distribution.

        Args:
            data: Original dataset with target class column.
            class_counts: Count of samples per class.
            target_size: Target number of samples per class.

        Returns:
            balanced_data: Balanced dataset containing original and synthetic samples.
        """
        samples = []
        types_dict = self.data_generator_.bn_.descriptor['types']
        
        # Calculate needed samples for each class
        needed_samples = (target_size - class_counts).clip(lower=0)
        
        # Generate samples for classes requiring augmentation
        for cls, needed in needed_samples.items():
            if needed > 0:
                samples.append(self._generate_samples_for_class(cls, needed, data.columns, types_dict))
        
        # Combine original data with all generated samples at once
        return pd.concat([data] + samples, ignore_index=True) if samples else data.copy()

    def _fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **params: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample the dataset using Bayesian Network synthetic generation.
        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            X_res: Resampled feature matrix.
            y_res: Corresponding resampled target vector.

        Raises:
            NotFittedError: If synthetic generator fails to fit Bayesian Network.

        Note:
            1. Combines X and y into single DataFrame for Bayesian Network learning
            2. Determines target sample sizes based on strategy
            3. Generates synthetic samples for minority classes using conditional sampling
            4. Preserves original data types and column names
        """

        # Combine X and y into a DataFrame with class column
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        if self.class_column is None:
            self.class_column = y.name if hasattr(y, 'name') else 'class'
        data = X_df.assign(**{self.class_column: y_series})
        
        # Preprocess data
        n_bins=5
        feature_types = pp = Preprocessor([]).get_nodes_types(data)
        for k in feature_types:
            if (feature_types[k] != 'cont') & (data[k].nunique()>n_bins):
                n_bins = data[k].nunique()
        encoder = LabelEncoder()
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        pp = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
        preprocessed_data, _ = pp.apply(data)
        
        
        # Fit Bayesian Network
        self.data_generator_.use_mixture = True
        fit_package = (preprocessed_data, pp.info, data)
        self.data_generator_.fit(X = fit_package)

        if self.data_generator_.bn_ is None:
            raise NotFittedError('Generator model must be fitted first.')

        # Determine target class size
        class_counts = data[self.class_column].value_counts().sort_values(ascending=False)
        target_size = class_counts.iloc[0] if self.strategy == 'max_class' else self.strategy

        # Generate synthetic samples for minority classes
        balanced_data = self._balance_classes(data, class_counts, target_size)
        # shuffle data
        if self.shuffle:
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
        # Split back into features and target
        X_res = balanced_data.drop(columns=[self.class_column]).to_numpy()
        y_res = balanced_data[self.class_column].to_numpy()

        return X_res, y_res
