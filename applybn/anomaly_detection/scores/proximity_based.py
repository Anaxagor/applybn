from applybn.anomaly_detection.scores.score import Score
from sklearn.neighbors import LocalOutlierFactor

from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import numpy as np


class LocalOutlierScore(Score):
    def __init__(self, proximity_steps=5, verbose=1, **kwargs):
        super().__init__(verbose)
        self.params = kwargs
        self.proximity_steps = proximity_steps

    def local_score(self, X):
        clf = LocalOutlierFactor(**self.params)
        clf.fit(X)
        # the higher, the more abnormal
        return np.negative(clf.negative_outlier_factor_)

    def score(self, X):
        proximity_factors = []

        proximity_iterator = (
            tqdm(range(self.proximity_steps), desc="Proximity")
            if self.verbose >= 1
            else range(self.proximity_steps)
        )

        for _ in proximity_iterator:
            try:
                t = np.random.randint(X.shape[1] // 2, X.shape[1])
                columns = np.random.choice(X.columns, t, replace=False)

                subset = X[columns].select_dtypes(include=["number"])

                # The higher, the more abnormal
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


class IsolationForestScore(Score):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

    def score(self, X):
        clf = IsolationForest(**self.params)
        clf.fit(X)
        return np.negative(clf.decision_function(X))
