from applybn.anomaly_detection.scores.score import Score
from applybn.core.schema import bamt_network

import pandas as pd
import numpy as np
from tqdm import tqdm


class ModelBasedScore(Score):
    """
    A generic score class that computes scores based on a provided model.
    Model must implement the predict_proba method.
    """

    def __init__(self, model):
        """
        Initializes the ModelBasedScore object.

        Args:
            model: The model used to compute probabilities for scoring.
        """
        super().__init__()
        self.model = model

    def score(self, X) -> np.ndarray:
        """
        Computes the score for the input data using the model's predicted probabilities.

        Args:
            X: The input data to be scored.

        Returns:
            np.ndarray: The predicted probabilities for the input data.
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The model does not have a predict_proba method.")
        probas = self.model.predict_proba(X)

        if isinstance(probas, pd.Series):
            return probas.values
        if isinstance(probas, np.ndarray):
            return probas


class BNBasedScore(Score):
    """
    A score class based on a Bayesian network (BN).

    Attributes:
        bn: The Bayesian network used for scoring.
        encoding: The encoding for discrete variables.
        child_nodes: The child nodes in the Bayesian network.
        verbose: The verbosity level for logging.
    """

    def __init__(self, bn: bamt_network, encoding: dict, verbose: int = 1):
        """
        Initializes the BNBasedScore object.

        Args:
            bn: The Bayesian network used for scoring.
            encoding: The encoding for discrete variables.
            verbose: The verbosity level for logging.
        """
        super().__init__(verbose=verbose)
        self.encoding = encoding
        self.bn = bn

        child_nodes = []
        for column in bn.nodes_names:
            if self.bn[column].disc_parents + self.bn[column].cont_parents:
                child_nodes.append(column)

        self.child_nodes = child_nodes

    def local_score(self, X: pd.DataFrame, node_name: str):
        """
        Computes the local score for a specific node in the Bayesian network.

        Args:
            X: The input data.
            node_name : The name of the node to compute the score for.

        Returns:
            np.ndarray: An array of local scores for the specified node.
        """
        node = self.bn[node_name]
        diff = []
        parents = node.cont_parents + node.disc_parents
        parent_dtypes = X[parents].dtypes.to_dict()

        for i in X.index:
            node_value = X.loc[i, node_name]
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
                            [self.encoding[node_name][class_name] for class_name in classes]
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
        """
        Computes the scores for all child nodes in the Bayesian network.

        Args:
            X: The input data.

        Returns:
            np.ndarray: A 2D array of scores for all child nodes.
        """
        if self.verbose >= 1:
            model_iterator = tqdm(self.child_nodes, desc="Model")
        else:
            model_iterator = self.child_nodes

        model_factors = []
        for child_node in model_iterator:
            model_factors.append(self.local_score(X, child_node))

        return np.hstack(model_factors)


class IQRBasedScore(BNBasedScore):
    """
    A score class that uses the Interquartile Range (IQR) for anomaly detection.
    """

    def __init__(self,
                 bn: bamt_network,
                 encoding: dict,
                 iqr_sensivity: float =1.0,
                 verbose:int=1):
        """
        Initializes the IQRBasedScore object.

        Args:
            bn: The Bayesian network used for scoring.
            encoding: The encoding for discrete variables.
            iqr_sensivity: The sensitivity factor for IQR-based scoring.
            verbose: The verbosity level for logging.
        """
        super().__init__(bn=bn, encoding=encoding, verbose=verbose)
        self.iqr_sensivity = iqr_sensivity

    @staticmethod
    def score_iqr(upper: float,
                  lower: float,
                  y: float,
                  max_distance: float,
                  min_distance: float):
        """
        Computes the IQR-based score for a given value.

        Args:
            upper: The upper bound of the IQR.
            lower: The lower bound of the IQR.
            y: The value to score.
            max_distance: The maximum distance for scaling.
            min_distance: The minimum distance for scaling.

        Raises:
            ValueError: If the closest value does not match either upper or lower bound.

        Returns:
            float: The IQR-based score.
        """
        if lower < y <= upper:
            return 0

        closest_value = min([upper, lower], key=lambda x: abs(x - y))
        current_distance = abs(closest_value - y)

        if closest_value == upper:
            ref_distance = max_distance
        elif closest_value == lower:
            ref_distance = min_distance
        else:
            raise ValueError("Unexpected state: closest_value does not match either upper or lower bound.")

        return min(1, current_distance / abs(ref_distance))

    def local_score(self, X: pd.DataFrame, node_name: str):
        """
        Computes the local IQR-based score for a specific node.

        Args:
            X: The input data.
            node_name : The name of the node to compute the score for.

        Returns:
            np.ndarray: An array of local scores for the specified node.
        """
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
    """
    A score class that uses conditional probability ratios for anomaly detection.
    """

    def __init__(self, bn: bamt_network, encoding:dict, verbose:int=1):
        """
        Initializes the CondRatioScore object.

        Args:
            bn: The Bayesian network used for scoring.
            encoding: The encoding for discrete variables.
            verbose: The verbosity level for logging.
        """
        super(CondRatioScore, self).__init__(bn=bn, encoding=encoding, verbose=verbose)

    def local_score(self, X: pd.DataFrame, node_name: str):
        """
        Computes the local conditional ratio score for a specific node.

        Args:
            X: The input data.
            node_name: The name of the node to compute the score for.

        Returns:
            np.ndarray: An array of local scores for the specified node.
        """
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
    def score_proba_ratio(sample: pd.Series, X_value: str, cond_dist: tuple):
        """
        Computes the conditional probability ratio score.

        Args:
            sample: The sample data.
            X_value: The value to score.
            cond_dist: The conditional distribution.

        Returns:
            float: The conditional probability ratio score.
        """
        cond_probs, values = cond_dist
        marginal_prob = sample.value_counts(normalize=True)[X_value]

        index = values.index(str(X_value))
        cond_prob = cond_probs[index]

        if not np.isfinite(marginal_prob / cond_prob):
            return np.nan

        return min(1, marginal_prob / cond_prob)


class CombinedIQRandProbRatioScore(BNBasedScore):
    """
    A score class that combines IQR-based scoring and probability ratio scoring for anomaly detection.
    """

    def __init__(self,
                 bn: bamt_network,
                 encoding: dict,
                 scores: dict,
                 verbose:int=1):
        """
        Initializes the CombinedIQRandProbRatioScore object.

        Args:
            bn: The Bayesian network used for scoring.
            encoding: The encoding for discrete variables.
            scores: A dictionary containing scoring objects for continuous and discrete variables.
            verbose: The verbosity level for logging.
        """
        super(CombinedIQRandProbRatioScore, self).__init__(
            bn=bn, encoding=encoding, verbose=verbose
        )
        self.scores = scores

    def local_score(self, X: pd.DataFrame, node_name: str):
        """
        Computes the local score for a specific node by combining IQR-based and probability ratio scoring.

        Args:
            X: The input data.
            node_name: The name of the node to compute the score for.

        Returns:
            np.ndarray: An array of local scores for the specified node.
        """
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
