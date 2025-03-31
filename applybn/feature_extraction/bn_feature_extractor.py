import pandas as pd
import bamt.preprocessors as pp
from typing import Optional, List, Tuple
import logging
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from bamt.preprocess.discretization import code_categories
from scipy.stats import norm
from applybn.core.estimators.base_estimator import BNEstimator
class BNFeatureGenerator(BaseEstimator, TransformerMixin):
    
    """
    Generates features based on a Bayesian Network (BN).
    """

    def __init__(self):
        self.bn = None
        self.target_name = ''

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the BNFeatureGenerator to the data.

        This involves:
        1.  Adding the target variable (if provided) to the input data.
        2.  Encoding categorical columns.
        3.  Discretizing continuous columns.
        4.  Creating a Bayesian Network based on the data types.
        5.  Learning the structure of the Bayesian Network (if known_structure is not provided).
        6.  Fitting the parameters of the Bayesian Network.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target variable. If provided, it will be added to the input data and
                treated as a node in the Bayesian Network.

        Returns:
            self: The fitted BNFeatureGenerator object.
        """

        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='kmeans',
            subsample=None
        )
        preprocessor = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

        if y is not (None):
            if y.name == '':
                y = y.rename('target')
            self.target_name = y.name
            X = pd.concat([X, y], axis=1).reset_index(drop=True)
        clean_data = X

        cat_columns = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
        if cat_columns:
            X, cat_encoder = code_categories(X, method="label", columns=cat_columns)
        discretized_data, est = preprocessor.apply(X)
        discretized_data = pd.DataFrame(discretized_data, columns=X.columns)

        # Initializing BNEstimator for BN selection & fitting
        info = preprocessor.info
        info =  {"types": info["types"], "signs": info['signs']}
        params = {}
        if self.target_name:
            bl = self.create_black_list(X, self.target_name)  # Edges to avoid
            params = {'bl_add': bl}
        learning_params = {'params': params}
        bn_estimator = BNEstimator(use_mixture=False, has_logit=True, learning_params=learning_params)
        L = (discretized_data, info, clean_data)
        self.bn = bn_estimator.fit(L)
        self.bn = bn_estimator.bn_
        logging.info(self.bn.get_info())

        return self

    def _process_target(self, X: pd.DataFrame) -> pd.Series:
        """
        Processes the target variable by making predictions using the Bayesian Network.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.Series: Predictions for the target variable.
        """
        if not self.target_name:
            return None

        predictions = self.bn.predict(test=X, parall_count=-1, progress_bar=False)
        return pd.Series(predictions[self.target_name])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame `X` into a new DataFrame where each column
        represents the calculated feature based on the fitted BN.

        Args:
            X (pd.DataFrame) is the input DataFrame to transform.

        Returns:
            pd.DataFrame is a new DataFrame with lambda-features.
        """
        if not self.bn:
            logging.error(AttributeError,
                          "Parameter learning wasn't done. Call fit method"
                          )
            return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

        results = []
        # Process each feature (column) in the row using the BN
        for _, row in X.iterrows():
            row_probs = [self.process_feature(feat, row, X) for feat in list(map(str, self.bn.nodes))]
            results.append(row_probs)

        result = pd.DataFrame(results, columns=['lambda_' + c for c in list(map(str, self.bn.nodes))])

        # Process target
        target_predictions = self._process_target(X)
        if target_predictions is not None:
            result['lambda_' + self.target_name] = target_predictions

        return result

    def create_black_list(self, X: pd.DataFrame, y: Optional[str]):
        if not y:
            return []
        target_node = y
        black_list = [(target_node, (col)) for col in X.columns.to_list() if col != target_node]

        return black_list

    def process_feature(self, feature: str, row: pd.Series, X):

        """
        Processes a single feature (node) in the Bayesian network for a given row of data.

        Args:
            feature (str): The name of the feature (node) being processed.
            row (pd.Series): A row from X.

        Returns:
            float: The probability or observed value depending on the node type.
        """
        if str(feature) != self.target_name:

            try:
                node = next((n for n in self.bn.nodes if n.name == feature), None)
                pvals = {}
                pvals_disc = []

                # Iterate through the continuous parents
                for p in node.cont_parents:
                    pvals[p] = row[p]

                # Iterate through the discrete parents
                for p in node.disc_parents:
                    norm_val = str(row[p])
                    pvals[p] = norm_val
                    pvals_disc.append(norm_val)

                # Process discrete nodes
                if node.type == 'Discrete':
                    vals = X[node.name].value_counts(normalize=True).sort_index()
                    vals = [str(i) for i in vals.index.to_list()]
                    return self._process_discrete_node(feature, row, pvals, vals)

                # Process non-discrete nodes
                else:
                    vals = X[node.name].value_counts(normalize=True).sort_index()
                    vals = [(i) for i in vals.index.to_list()]
                    return self._process_non_discrete_node(feature, row, pvals, vals, pvals_disc)

            except Exception as e:
                logging.error(f"Error processing node {feature}: {e}")
                return 0.0001

    def _process_discrete_node(self, feature, row, pvals, vals):
        """
        Processes a discrete node.

        Args:
            node - the discrete node object.
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.
            vals - possible values of the 'feature'.

        Returns:
            float - value of a new feature.
        """

        obs_value = str((row[feature]))
        try:
            dist = self.bn.get_dist(str(feature), pvals=pvals)

        except:
            return 0.0001

        try:
            # Try to find the index of the observed value in the node's values
            idx = dist["vals"].index(obs_value)
            if not dist.get("cprob"):
                return 0.0001
            return dist["cprob"][idx]
        except:
            idx = vals.index((obs_value))
            return dist[idx]

    def _process_non_discrete_node(self, feature, row, pvals, vals, pvals_disc):
        """
        Processes a non-discrete node.

        Args:
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.
            vals - possible values of the 'feature'.

        Returns:
            float - value of a new feature.
        """
        obs_value = (row[feature])
        try:
            dist = self.bn.get_dist(str(feature), pvals=pvals)
        except:
            return 0.0001

        match dist:
            case dict() if len(dist) > 2:
                try:
                    mean, variance = dist['mean'], dist['variance']
                    sigma = np.sqrt(variance)
                    if sigma <= 0 or np.isnan(sigma):
                        return 0.0001
                    prob = norm.cdf(obs_value, loc=mean, scale=sigma)
                    if np.isnan(prob):
                        return 0.0001
                    return prob
                except:
                    logging.warning("dist not found for node %s:", feature)
                    return 0.0001

            case dict():
                try:
                    idx = dist["vals"].index(obs_value)
                    return dist["cprob"][idx]
                except:
                    idx = vals.index((obs_value))
                    return dist[idx]

            case tuple() if len(dist) == 2: # to be deleted
                try:
                    mean, variance = dist
                    sigma = variance
                    prob = norm.cdf(obs_value, loc=mean, scale=sigma)
                    if np.isnan(prob):
                        return 0.0001
                    return prob
                except:
                    logging.warning("dist not found for node %s:", feature)
                    return 0.0001

            case tuple():
                logging.warning("unknown dist for node %s: %s", feature, dist)
                return 0.0001

            case np.ndarray(): # to be deleted
                try:
                    if "classes" in self.bn.distributions[feature]["hybcprob"][str(pvals_disc)]:
                        classes = self.bn.distributions[feature]["hybcprob"][str(pvals_disc)]["classes"]
                        if obs_value in classes:
                            idx = classes.index(obs_value)
                            prob = dist[idx]
                            return prob
                        else:
                            return 0.0001
                except KeyError:
                    logging.warning("dist not found for node %s:", feature)
                    return 0.0001

            case _:
                logging.warning("dist not found for node %s:", feature)
                return 0.0001

        return 0.0001
