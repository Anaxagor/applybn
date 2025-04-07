from applybn.anomaly_detection.scores.score import Score
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import IsolationForest
import numpy as np

class LocalOutlierScore(Score):
    def __init__(self,  proximity_steps, verbosity=1, **kwargs):
        super().__init__(verbosity)
        self.params = kwargs
        self.proximity_steps = proximity_steps

    def local_score(self, X):
        clf = LocalOutlierFactor(**self.params)
        clf.fit(X)
        # the higher, the more abnormal
        return np.negative(clf.negative_outlier_factor_)

    def score(self, X):
        proximity_factors = []
        # if self.verbose >= 1:
        #     proximity_iterator = tqdm(range(self.proximity_steps), desc="Proximity")
        # else:
        #     proximity_iterator = range(self.proximity_steps)

        while len(proximity_factors) < self.proximity_steps:
            try:
                t = np.random.randint(X.shape[1] // 2, X.shape[1] - 1)
                columns = np.random.choice(X.columns, t, replace=False)

                subset = X[columns]
                subset_cont = subset.select_dtypes(include=["number"])

                # The higher, the more abnormal
                outlier_factors = self.local_score(subset_cont)
            except ValueError:
                continue
            proximity_factors.append(outlier_factors)

        return np.vstack(proximity_factors).T


class IsolationForestScore(Score):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

    def score(self, X):
        clf = IsolationForest(**self.params)
        clf.fit(X)
        return np.negative(clf.decision_function(X))
