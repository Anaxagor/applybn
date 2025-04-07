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
    def __init__(self, bn: bamt_network, encoding, verbose=1,):
        super().__init__(verbose=verbose)
        self.encoding = encoding
        self.bn = bn

    def local_score(self, X: pd.DataFrame, node_name):
        node = self.bn[node_name]
        diff = []
        dist = self.bn.distributions[node_name]
        parents = node.cont_parents + node.disc_parents

        for _, row in X.iterrows():
            # todo: disgusting
            pvalues = row[parents].to_dict()

            pvals_bamt_style = [pvalues[parent] for parent in parents]

            cond_dist = self.bn.get_dist(node_name, pvals=pvalues)
            # todo: super disgusting
            if isinstance(cond_dist, tuple):
                if len(cond_dist) == 2:
                    cond_mean, var = cond_dist
                else:
                    cond_mean = node.predict(dist, pvals=pvals_bamt_style)
                    # todo: may be use singular vals of cov matrix as norm constants?
                    diff.append(row[node_name] - cond_mean)
                    continue
            else:
                dispvals = []
                for pval in pvals_bamt_style:
                    if isinstance(pval, str):
                        dispvals.append(pval)

                if "vals" in dist.keys():
                    classes = dist["vals"]
                elif "classes" in dist.keys():
                    classes = dist["classes"]
                elif "hybcprob" in dist.keys():
                    if "classes" in dist["hybcprob"][str(dispvals)]:
                        classes = dist["hybcprob"][str(dispvals)]["classes"]
                        if pd.isna(classes[0]):
                            # if subspace of a combination is empty
                            diff.append(np.nan)
                            continue
                    else:
                        raise Exception()
                else:
                    raise Exception()

                match self.bn.descriptor["types"][node_name]:
                    case "disc_num":
                        classes_ = [int(class_name) for class_name in classes]
                    case "disc":
                        classes_ = np.asarray([self.encoding[node_name][class_name] for class_name in classes])
                    case "cont":
                        classes_ = classes

                cond_mean = classes_ @ np.asarray(cond_dist).T

            match self.bn.descriptor["types"][node_name]:
                case "disc_num":
                    diff.append(
                        (row[node_name] - cond_mean)
                    )
                case "disc":
                    diff.append(
                        self.encoding[node_name][row[node_name]] - cond_mean
                    )
                case "cont":
                    diff.append(
                        (row[node_name] - cond_mean) / var
                    )

        return np.asarray(diff).reshape(-1, 1)

    def score(self, X: pd.DataFrame):
        child_nodes = []

        for column in X.columns:
            if self.bn[column].disc_parents + self.bn[column].cont_parents:
                child_nodes.append(column)

        if self.verbose >= 1:
            model_iterator = tqdm(child_nodes, desc="Model")
        else:
            model_iterator = child_nodes

        model_factors = []

        for child_node in model_iterator:
            model_factors.append(self.local_score(X, child_node))

        return np.hstack(model_factors)
