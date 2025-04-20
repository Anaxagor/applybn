import numpy as np
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_X_y
from bamt.external.pyitlib.DiscreteRandomVariableUtils import entropy, information_mutual


class NMIFeatureSelector(BaseEstimator, SelectorMixin):
    """Feature selection based on Normalized Mutual Information (NMI).

    This selector performs a two-stage feature selection process:
    1. Select features with NMI to the target above a threshold.
    2. Remove redundant features based on pairwise NMI between features.

    Parameters
    ----------
    threshold : float, optional (default=0.0)
        The threshold value for the first stage selection. Features with NMI
        greater than this value are retained after the first stage.
    n_bins : int, optional (default=10)
        The number of bins to use for discretizing continuous features.

    Attributes
    ----------
    nmi_features_target_ : ndarray of shape (n_features,)
        The NMI between each feature and the target.
    selected_features_ : ndarray of shape (n_selected_features,)
        The indices of the selected features after both stages.
    selected_mask_ : ndarray of shape (n_features,)
        Boolean mask indicating selected features.
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit.
    """

    def __init__(self, threshold=0.0, n_bins=10):
        self.threshold = threshold
        self.n_bins = n_bins

    def fit(self, X, y):
        """Fit the feature selector to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        # Capture feature names BEFORE data conversion
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])
        
        # Convert to numpy arrays after capturing names
        X, y = check_X_y(X, y, dtype=None, force_all_finite=True)

        # Discretize continuous features and target
        X_disc = self._discretize_features(X)
        y_disc = self._discretize_target(y.reshape(-1, 1)).flatten()

        # Compute NMI between each feature and target
        mi = np.array([information_mutual(X_disc[:, i], y_disc) for i in range(X_disc.shape[1])])
        h_y = entropy(y_disc)
        h_features = np.array([entropy(X_disc[:, i]) for i in range(X_disc.shape[1])])
        self.nmi_features_target_ = np.zeros_like(mi)
        for i in range(len(mi)):
            min_h = min(h_features[i], h_y)
            self.nmi_features_target_[i] = mi[i] / min_h if min_h > 0 else 0.0

        # First stage: select features above threshold
        first_stage_mask = self.nmi_features_target_ > self.threshold
        selected_indices = np.where(first_stage_mask)[0]

        # Second stage: remove redundant features
        keep = np.ones(len(selected_indices), dtype=bool)
        nmi_selected = self.nmi_features_target_[selected_indices]

        for j in range(len(selected_indices)):
            fj_idx = selected_indices[j]
            fj = X_disc[:, fj_idx]
            nmi_j = nmi_selected[j]
            h_fj = entropy(fj)

            for i in range(len(selected_indices)):
                if i == j or not keep[i]:
                    continue
                fi_idx = selected_indices[i]
                fi = X_disc[:, fi_idx]
                nmi_i = nmi_selected[i]

                if nmi_i > nmi_j:
                    mi_pair = information_mutual(fi, fj)
                    h_fi = entropy(fi)
                    min_h_pair = min(h_fi, h_fj)
                    nmi_pair = mi_pair / min_h_pair if min_h_pair > 0 else 0.0

                    if nmi_pair > nmi_j:
                        keep[j] = False
                        break  # No need to check other i's

        self.selected_features_ = selected_indices[keep]
        self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.selected_mask_[self.selected_features_] = True

        return self

    def _get_support_mask(self):
        return self.selected_mask_

    def _discretize_features(self, X):
        """Discretize continuous features using KBinsDiscretizer."""
        X_disc = np.empty_like(X, dtype=np.int32)
        for i in range(X.shape[1]):
            col = X[:, i]
            unique_vals = np.unique(col)
            if len(unique_vals) > self.n_bins:
                # Discretize continuous feature
                discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
                discretized_col = discretizer.fit_transform(col.reshape(-1, 1)).flatten().astype(np.int32)
                X_disc[:, i] = discretized_col
            else:
                # Treat as discrete (convert to integers for pyitlib)
                X_disc[:, i] = col.astype(np.int32)
        return X_disc

    def _discretize_target(self, y):
        """Discretize target variable if continuous."""
        unique_vals = np.unique(y)
        if len(unique_vals) > self.n_bins:
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
            y_disc = discretizer.fit_transform(y).flatten().astype(np.int32)
        else:
            y_disc = y.astype(np.int32).flatten()
        return y_disc