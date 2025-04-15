from applybn.anomaly_detection.scores.score import Score
from applybn.core.schema import bamt_network

import pandas as pd
import numpy as np
from tqdm import tqdm


class ModelBasedScore(Score):
    """Generic Score"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def score(self, X):
        probas = self.model.predict_proba(X)

        if isinstance(probas, pd.Series):
            return probas.values
        if isinstance(probas, np.ndarray):
            return probas


class BNBasedScore(Score):
    def __init__(
        self,
        bn: bamt_network,
        encoding,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.encoding = encoding
        self.bn = bn

        child_nodes = []

        for column in bn.nodes_names:
            if self.bn[column].disc_parents + self.bn[column].cont_parents:
                child_nodes.append(column)

        self.child_nodes = child_nodes

    def local_score(self, X: pd.DataFrame, node_name):
        node = self.bn[node_name]
        diff = []
        parents = node.cont_parents + node.disc_parents
        parent_dtypes = X[parents].dtypes.to_dict()

        for i in X.index:
            node_value = X.loc[i, node_name]
            # for some reason pd.DataFrame.__getitem__ does not preserve data types
            row_df = X.loc[[i], parents].astype(parent_dtypes)

            pvalues = row_df.to_dict("records")[0]

            cond_dist = self.bn.get_dist(node_name, pvals=pvalues)

            if "gaussian" in cond_dist.node_type:
                cond_mean, std = cond_dist.get()
            else:
                probs, classes = cond_dist.get()

                match self.bn.descriptor["types"][node_name]:
                    case "disc_num":
                        classes_ = [int(class_name) for class_name in classes]
                    case "disc":
                        classes_ = np.asarray(
                            [
                                self.encoding[node_name][class_name]
                                for class_name in classes
                            ]
                        )

                cond_mean = classes_ @ np.asarray(probs).T

            match self.bn.descriptor["types"][node_name]:
                case "disc_num":
                    diff.append((node_value - cond_mean))
                case "disc":
                    diff.append(self.encoding[node_name][node_value] - cond_mean)
                case "cont":
                    diff.append((node_value - cond_mean) / std)

        return np.asarray(diff).reshape(-1, 1)

    def score(self, X: pd.DataFrame):
        if self.verbose >= 1:
            model_iterator = tqdm(self.child_nodes, desc="Model")
        else:
            model_iterator = self.child_nodes

        model_factors = []

        for child_node in model_iterator:
            model_factors.append(self.local_score(X, child_node))

        return np.hstack(model_factors)


class IQRBasedScore(BNBasedScore):
    def __init__(
        self,
        bn: bamt_network,
        encoding,
        iqr_sensivity=1.0,
        verbose=1,
    ):
        super().__init__(bn=bn, encoding=encoding, verbose=verbose)
        self.encoding = encoding
        self.bn = bn
        self.iqr_sensivity = iqr_sensivity

    @staticmethod
    def score_iqr(upper, lower, y, max_distance, min_distance):
        if lower < y <= upper:
            return 0

        closest_value = min([upper, lower], key=lambda x: abs(x - y))

        current_distance = abs(closest_value - y)

        if closest_value == upper:
            ref_distance = max_distance
        elif closest_value == lower:
            ref_distance = min_distance
        else:
            raise Exception
        result = min(1, current_distance / abs(ref_distance))

        return result

    def local_score(self, X, node_name):
        node = self.bn[node_name]

        parents = node.cont_parents + node.disc_parents
        parent_dtypes = X[parents].dtypes.to_dict()

        scores = []
        for i in X.index:
            row_df = X.loc[[i], parents].astype(parent_dtypes)
            pvalues = row_df.to_dict("records")[0]

            dist = self.bn.get_dist(node_name, pvals=pvalues).get(with_gaussian=True)

            X_value = X.loc[i, node_name]

            q25 = dist.ppf(0.25)
            q75 = dist.ppf(0.75)
            iqr = q75 - q25

            lower_bound = q25 - iqr * self.iqr_sensivity
            upper_bound = q75 + iqr * self.iqr_sensivity

            scores.append(
                self.score_iqr(
                    upper_bound,
                    lower_bound,
                    X_value,
                    max_distance=1 * X[node_name].max(),
                    min_distance=1 * X[node_name].min(),
                )
            )

        return np.asarray(scores).reshape(-1, 1)


class CondRatioScore(BNBasedScore):
    def __init__(
        self,
        bn: bamt_network,
        encoding,
        verbose=1,
    ):
        super(CondRatioScore, self).__init__(bn=bn, encoding=encoding, verbose=verbose)

    def local_score(self, X: pd.DataFrame, node_name):
        node = self.bn[node_name]
        diff = []
        parents = node.cont_parents + node.disc_parents
        parent_dtypes = X[parents].dtypes.to_dict()

        for i in X.index:
            row_df = X.loc[[i], parents].astype(parent_dtypes)
            pvalues = row_df.to_dict("records")[0]

            node_value = X.loc[i, node_name]
            cond_dist = self.bn.get_dist(node_name, pvals=pvalues).get()

            diff.append(self.score_proba_ratio(X[node_name], node_value, cond_dist))

        return np.asarray(diff).reshape(-1, 1)

    @staticmethod
    def score_proba_ratio(sample: pd.Series, X_value, cond_dist):
        cond_probs, values = cond_dist
        marginal_prob = sample.value_counts(normalize=True)[X_value]

        index = values.index(str(X_value))

        cond_prob = cond_probs[index]

        if not np.isfinite(marginal_prob / cond_prob):
            # it is impossible to estimate if cond dataframe doesn't contain X_value
            return np.nan

        # the greater, the more abnormal
        return min(1, marginal_prob / cond_prob)


class CombinedIQRandProbRatioScore(BNBasedScore):
    def __init__(
        self,
        bn: bamt_network,
        encoding,
        scores,
        verbose=1,
    ):
        super(CombinedIQRandProbRatioScore, self).__init__(
            bn=bn, encoding=encoding, verbose=verbose
        )

        self.scores = scores

    def local_score(self, X: pd.DataFrame, node_name):
        node = self.bn[node_name]
        iqr_sensivity = self.scores["cont"].iqr_sensivity

        parents = node.cont_parents + node.disc_parents

        parent_dtypes = X[parents].dtypes.to_dict()

        scores = []
        for i in X.index:
            row_df = X.loc[[i], parents].astype(parent_dtypes)
            pvalues = row_df.to_dict("records")[0]

            X_value = X.loc[i, node_name]
            dist = self.bn.get_dist(node_name, pvals=pvalues)

            if "gaussian" in dist.node_type:
                dist = dist.get(with_gaussian=True)
                if dist.kwds["scale"] == 0:
                    scores.append(0)
                    continue

                q25 = dist.ppf(0.25)
                q75 = dist.ppf(0.75)
                iqr = q75 - q25

                lower_bound = q25 - iqr * iqr_sensivity
                upper_bound = q75 + iqr * iqr_sensivity

                scores.append(
                    self.scores["cont"].score_iqr(
                        upper_bound,
                        lower_bound,
                        X_value,
                        max_distance=1 * X[node_name].max(),
                        min_distance=1 * X[node_name].min(),
                    )
                )
            else:
                dist = dist.get()
                scores.append(
                    self.scores["disc"].score_proba_ratio(X[node_name], X_value, dist)
                )

        return np.asarray(scores).reshape(-1, 1)
