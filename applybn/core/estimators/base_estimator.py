import logging

import pandas as pd
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import Options
from sklearn.exceptions import NotFittedError

from bamt.networks import DiscreteBN, HybridBN, ContinuousBN
from bamt.utils.GraphUtils import nodes_types

from typing import Unpack, Literal
from applybn.core.schema import ParamDict
from applybn.core.logger import Logger
from applybn.core.exceptions.estimator_exc import NodesAutoTypingError

logger = Logger("estimators", level=logging.DEBUG)


class BNEstimator(BaseEstimator):
    """
    A Bayesian Network Estimator class that extends scikit-learn's BaseEstimator.

    Use it only in the backend of the library and proceed with caution when out of CorePipeline context.

    Attributes:
        has_logit (bool): Indicates if logit nodes are used.
        use_mixture (bool): Indicates if mixture model is used.
        bn_type (Optional[Literal["hybrid", "disc", "cont"]]): Type of Bayesian Network.
        partial (Options): Indicates if partial fitting is used.
        learning_params (Optional[Unpack[ParamDict]]): Parameters for learning.
    """

    _parameter_constraints = {
        "has_logit": [bool],
        "use_mixture": [bool],
        "bn_type": [str, None],
        "partial": [Options(object, {False, "parameters", "structure"})],
        "learning_params": [None, dict],
    }

    def __init__(
        self,
        has_logit: bool = False,
        use_mixture: bool = False,
        partial: False | Literal["parameters", "structure"] = False,
        bn_type: Literal["hybrid", "disc", "cont"] | None = None,
        learning_params: Unpack[ParamDict] | None = None,
    ):
        """
        Initializes the BNEstimator with the given parameters.

        Args:
            has_logit (bool): Indicates if logit transformation is used.
            use_mixture (bool): Indicates if mixture model is used.
            partial (Options): Indicates if partial fitting is used.
            bn_type (Optional[Literal["hybrid", "disc", "cont"]]): Type of Bayesian Network.
            learning_params (Optional[Unpack[ParamDict]]): Parameters for learning.
        """
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.bn_type = bn_type
        self.partial = partial
        self.learning_params = {} if learning_params is None else learning_params

    def _is_fitted(self):
        """
        Checks whether the estimator is fitted or not by checking "bn_" key if __dict__.
        This has to be done because check_is_fitted(self) does not imply correct and goes into recursion because of
        delegating strategy in getattr method.
        """
        return True if "bn_" in self.__dict__ else False

    def __getattr__(self, attr: str):
        """If attribute is not found in the pipeline, look in the last step of the pipeline."""
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            if self._is_fitted():
                return getattr(self.bn_, attr)
            else:
                raise NotFittedError("BN Estimator has not been fitted.")

    @staticmethod
    def detect_bn(data: pd.DataFrame) -> str:
        """
        Detects the type of Bayesian Network based on the data.
        Bamt typing is used.

        Args:
            data (pd.DataFrame): The input data to analyze.

        Returns:
            str: The detected type of Bayesian Network.

        Raises:
            None: an error translates into bamt logger.
                Possible errors:
                    "Unsupported data type. Dtype: {dtypes}"
        """

        node_types = nodes_types(data)

        if len(node_types.keys()) != len(data.columns):
            diff = set(data.columns) - set(node_types.keys())
            raise NodesAutoTypingError(diff)

        nodes_types_unique = set(node_types.values())

        net_types2unqiue = {
            "hybrid": [
                {"cont", "disc", "disc_num"},
                {"cont", "disc_num"},
                {"cont", "disc"},
            ],
            "disc": [{"disc"}, {"disc_num"}, {"disc", "disc_num"}],
            "cont": [{"cont"}],
        }
        find_matching_key = (
            {frozenset(s): k for k, v in net_types2unqiue.items() for s in v}
        ).get
        return find_matching_key(frozenset(nodes_types_unique))

    def init_bn(self, bn_type: Literal["hybrid", "disc", "cont"]):
        """
        Initializes the Bayesian Network based on the type.

        Args:
            bn_type: The type of Bayesian Network to initialize.

        Returns:
            An instance of the corresponding Bayesian Network class.

        Raises:
            TypeError: Invalid bn_type.
        """
        str2net = {"hybrid": HybridBN, "disc": DiscreteBN, "cont": ContinuousBN}

        params = dict()
        match bn_type:
            case "hybrid":
                params = dict(use_mixture=self.use_mixture, has_logit=self.has_logit)
            case "cont":
                params = dict(use_mixture=self.use_mixture)
            case "disc":
                ...
            case _:
                raise TypeError(f"Invalid bn_type, obtained bn_type: {bn_type}")
        return str2net[bn_type](**params)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fits the Bayesian Network to the data.

        Args:
            X (tuple): a tuple with (X, descriptor, clean_data).
                If partial is "structure", clean_data can be None (not used).
            y (None): not used.

        Returns:
            self (BNEstimator): The fitted estimator.
        """

        # this has to be done because scikit learn unpacking problem
        # inside pipeline there is unpacking.
        X, descriptor, clean_data = X
        if not self.partial == "parameters":
            if not self.bn_type in ["hybrid", "disc", "cont"]:
                bn_type_ = self.detect_bn(clean_data)
            else:
                bn_type_ = self.bn_type

            bn = self.init_bn(bn_type_)

            self.bn_ = bn
            self.bn_type = bn_type_

        match self.partial:
            case "parameters":
                if not self.bn_.edges:
                    raise NotFittedError(
                        "Trying to learn parameters on unfitted estimator. Call fit method first."
                    )
                self.bn_.fit_parameters(clean_data)
            case "structure":
                self.bn_.add_nodes(descriptor)
                self.bn_.add_edges(X, progress_bar=False, **self.learning_params)
            case False:
                self.bn_.add_nodes(descriptor)
                self.bn_.add_edges(X, progress_bar=False, **self.learning_params)
                self.bn_.fit_parameters(clean_data)

        return self
