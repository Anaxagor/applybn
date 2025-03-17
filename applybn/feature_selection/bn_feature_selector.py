from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np


class BNFeatureSelector(BaseEstimator, SelectorMixin):
    """Bayesian Network feature selector."""

    # TODO: Implement Bayesian Network feature selector, use SelectorMixin as reference
    def fit(self, X, y=None):
        """
Fit the model to the given data.

    Args:
        X: The input samples.
        y: Optional target values. Not used in this implementation.

    Returns:
        self: Returns the instance itself.
    """
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        """
Returns a boolean mask indicating supported features.

  The first two features are always considered supported, while the rest are not.

  Args:
    self: The instance of the class.

  Returns:
    numpy.ndarray: A boolean array where True indicates a supported feature and 
                   False indicates an unsupported feature.
  """
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[:2] = True
        return mask
