from imblearn.over_sampling.base import BaseOverSampler
import pandas as pd
from sklearn.exceptions import NotFittedError
from applybn.synthetics.bn_synthetic_generator import BNSyntheticGenerator

class BNOverSampler(BaseOverSampler):
    """
    A Bayesian Network-based oversampler for handling imbalanced datasets.
    
    This class uses Bayesian Networks to learn the joint probability distribution of features
    and generates synthetic samples for minority classes to balance class distribution.
    Inherits from BaseOverSampler to ensure compatibility with scikit-learn pipelines.

    Parameters
    ----------
    class_column : str, default=None
        Name of the target class column. If None, will attempt to infer from y's name attribute.

    strategy : str or int, default='max_class'
        Oversampling strategy:
        - 'max_class': match minority classes to the size of the largest class
        - integer: directly specify target number of samples per class

    shuffle : bool, default=True
        Whether to shuffle the dataset after resampling.

    Attributes
    ----------
    data_generator_ : BNSyntheticGenerator
        Fitted Bayesian Network synthetic data generator instance.

    Example
    -------
    >>> from applybn.imbalanced.over_sampling.bn_over_sampler import BNOverSampler
    >>> oversampler = BNOverSampler(class_column='target', strategy='max_class')
    >>> X_res, y_res = oversampler.fit_resample(X, y)
    """

    def __init__(self, class_column=None, strategy='max_class', shuffle=True):
        """
        Initialize the BNOverSampler with class column, balancing strategy, and shuffle option.

        Parameters
        ----------
        class_column : str, optional
            Name of the target variable column in the dataset.
            
        strategy : str or int, optional
            Determines target sample count for minority classes:
            - 'max_class' (default): match largest class size
            - Integer value: explicit target sample count
            
        shuffle : bool, optional
            Whether to shuffle data after resampling (default=True).
        """
        super().__init__()
        self.class_column = class_column
        self.strategy = strategy
        self.shuffle = shuffle
        self.data_generator_ = BNSyntheticGenerator()

    def _fit_resample(self, X, y, **params):
        """
        Resample the dataset using Bayesian Network synthetic generation.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Feature matrix
        
        y : pandas.Series or array-like
            Target vector

        Returns
        -------
        X_res : pandas.DataFrame
            Resampled feature matrix
            
        y_res : pandas.Series
            Corresponding resampled target vector

        Raises
        ------
        NotFittedError
            If synthetic generator fails to fit Bayesian Network

        Notes
        -----
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

        # Fit Bayesian Network using DataGenerator
        self.data_generator_.fit(data)
        if self.data_generator_.bn is None:
            raise NotFittedError('Generator model must be fitted first.')

        # Determine target class size
        class_counts = data[self.class_column].value_counts().sort_values(ascending=False)
        target_size = class_counts.iloc[0] if self.strategy == 'max_class' else self.strategy

        # Generate synthetic samples for minority classes
        balanced_data = data.copy()
        types_dict = self.data_generator_.bn.descriptor['types']
        for cls in class_counts.index:
            current_count = class_counts[cls]
            needed = max(0, target_size - current_count)
            if needed > 0:
                additional_samples = pd.DataFrame()
                samples = self.data_generator_.bn.sample(
                    needed, 
                    evidence={self.class_column: cls}, filter_neg=False
                )[data.columns]
                if samples.shape[0] < needed:
                    additional_samples = self.data_generator_.bn.sample(
                    needed, 
                    evidence={self.class_column: cls}, filter_neg=False
                )[data.columns]
                samples = pd.concat([samples, additional_samples.sample(needed - samples.shape[0])])
                for c in samples.columns:
                    if types_dict[c] == 'disc_num':
                        samples[c] = samples[c].astype(int)
                balanced_data = pd.concat([balanced_data, samples], ignore_index=True)
        # shuffle data
        if self.shuffle:
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
        # Split back into features and target
        X_res = balanced_data.drop(columns=[self.class_column])
        y_res = balanced_data[self.class_column]

        return X_res, y_res
