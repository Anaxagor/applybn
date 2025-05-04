from applybn.anomaly_detection.scores.score import Score
from sklearn.neighbors import LocalOutlierFactor

from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class ProximityBasedScore(Score):
    """
    Calculates proximity-based outlier scores for data."""

    def __init__(self, verbose: int, proximity_steps: int = 5):
        """
        Initializes the class with verbosity level and proximity steps.

            Args:
                verbose:  Controls the amount of debugging information printed to the console.
                proximity_steps: The number of steps to consider when calculating proximity. Defaults to 5.

            Returns:
                None
        """
        super().__init__(verbose)
        self.proximity_steps = proximity_steps

    def local_score(self, X: pd.DataFrame):
        """
        Calculates a local score based on the input DataFrame.

          Args:
            X: The input DataFrame used for calculating the score.

          Returns:
            float: The calculated local score.
        """
        pass

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes the proximity-based outlier scores for the given data.

        This method performs multiple iterations, each time selecting a random subset of columns
        and computing the outlier scores for that subset.

        Args:
            X: The input data.

        Returns:
            np.ndarray: A 2D array of outlier scores, where each column corresponds to an iteration.

        Raises:
            RuntimeError: If no valid proximity scores could be computed.
        """
        proximity_factors = []

        proximity_iterator = (
            tqdm(range(self.proximity_steps), desc="Proximity")
            if self.verbose >= 1
            else range(self.proximity_steps)
        )

        for _ in proximity_iterator:
            try:
                # Randomly select a subset of columns
                t = np.random.randint(X.shape[1] // 2, X.shape[1])
                columns = np.random.choice(X.columns, t, replace=False)

                # Select only numeric columns from the subset
                subset = X[columns].select_dtypes(include=["number"])

                # Compute outlier factors for the subset
                outlier_factors = self.local_score(subset)
                proximity_factors.append(outlier_factors)

            except ValueError:
                # Skip iterations with invalid subsets
                continue

        if not proximity_factors:
            raise RuntimeError(
                "No valid proximity scores could be computed. Do you have any cont columns?"
            )

        return np.vstack(proximity_factors).T


class LocalOutlierScore(ProximityBasedScore):
    """
    A class for computing outlier scores using the Local Outlier Factor (LOF) algorithm.
    """

    def __init__(self, proximity_steps: int = 5, verbose: int = 1, **kwargs):
        """
        Initializes the LocalOutlierScore object.

        Args:
            proximity_steps: The number of proximity steps to perform. Default is 5.
            verbose: The verbosity level for logging. Default is 1.
            **kwargs: Additional parameters for the Local Outlier Factor algorithm.
        """
        super().__init__(proximity_steps=proximity_steps, verbose=verbose)
        self.params = kwargs

    def local_score(self, X: pd.DataFrame):
        """
        Computes the local outlier scores for the given data using the LOF algorithm.

        Args:
            X: The input data.

        Returns:
            np.ndarray: An array of negative outlier factors, where higher values indicate more abnormal data points.
        """
        clf = LocalOutlierFactor(**self.params)
        clf.fit(X)
        # The higher the value, the more abnormal the data point
        return np.negative(clf.negative_outlier_factor_)


class IsolationForestScore(ProximityBasedScore):
    """
    A class for computing outlier scores using the Isolation Forest algorithm.
    """

    def __init__(self, proximity_steps: int = 5, verbose: int = 1, **kwargs):
        """
        Initializes the IsolationForestScore object.

        Args:
            **kwargs: Additional parameters for the Isolation Forest algorithm.
        """
        super().__init__(verbose=verbose, proximity_steps=proximity_steps)
        self.params = kwargs

    def local_score(self, X: pd.DataFrame):
        """
        Computes the outlier scores for the given data using the Isolation Forest algorithm.

        Args:
            X: The input data.

        Returns:
            np.ndarray: An array of negative decision function values, where higher values indicate more abnormal data points.
        """
        clf = IsolationForest(**self.params)
        clf.fit(X)
        return np.negative(clf.decision_function(X))
