import logging

from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import Options

from bamt.networks import DiscreteBN, HybridBN, ContinuousBN
from bamt.utils.GraphUtils import nodes_types

from typing import Unpack, Literal, Optional
from applybn.core.schema import ParamDict
from applybn.core.logger import Logger

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
        "learning_params": [None, dict]
    }

    def __init__(self,
                 has_logit: bool = False,
                 use_mixture: bool = False,
                 partial=False,
                 bn_type: Optional[Literal["hybrid", "disc", "cont"]] = None,
                 learning_params: Optional[Unpack[ParamDict]] = None,
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
        self.learning_params = learning_params

    @staticmethod
    def detect_bn(data):
        """
        Detects the type of Bayesian Network based on the data.

        Args:
            data: The input data to analyze.

        Returns:
            str: The detected type of Bayesian Network.
        """
        node_types = nodes_types(data)
        nodes_types_unique = set(node_types.values())

        net_types2unqiue = {
            "hybrid": [{"cont", "disc", "disc_num"},
                       {"cont", "disc_num"},
                       {"cont", "disc"}],
            "disc": [{"disc"}, {"disc_num"}, {"disc", "disc_num"}],
            "cont": [{"cont"}],
        }
        find_matching_key = ({frozenset(s): k for k, v in net_types2unqiue.items() for s in v}).get
        return find_matching_key(frozenset(nodes_types_unique))

    def init_bn(self, bn_type):
        """
        Initializes the Bayesian Network based on the type.

        Args:
            bn_type (str): The type of Bayesian Network to initialize.

        Returns:
            An instance of the corresponding Bayesian Network class.
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
                logger.error(f"Unknown bn_type, obtained {bn_type}")

        return str2net[bn_type](**params)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self,
            X,
            y=None):
        """
        Fits the Bayesian Network to the data.

        Args:
            X: The input data.
            y: The target values (default is None).

        Returns:
            self: The fitted estimator.
        """
        # this has to be done because scikit learn unpacking problem
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
                # todo: check for structure
                if not self.bn_.edges:
                    logger.error("Trying to learn parameters on unfitted estimator. Call fit method first.")
                self.bn_.fit_parameters(clean_data)
            case "structure":
                self.bn_.add_nodes(descriptor)
                self.bn_.add_edges(X)
            case False:
                self.bn_.add_nodes(descriptor)
                self.bn_.add_edges(X)
                self.bn_.fit_parameters(X)

        return self

# def predict_proba(self, X: pd.DataFrame):
#     # check_is_fitted(self)
#     # X = check_array(X)
#     # todo: turn into vectors? very slow
#     probas = []
#     for indx, row in X.iterrows():
#         anom_index = self.bn.distributions["y"]["classes"].index('1')
#         try:
#             result = self.bn.get_dist("y", pvals=row.to_dict())[int(anom_index)]
#         except KeyError:
#             result = np.nan
#         probas.append(result)
#
#     return pd.Series(probas)
#
# def inject_target(self,
#                   y,
#                   data,
#                   node: Type[BaseNode] = DiscreteNode):
#     if not self.bn.edges:
#         # todo
#         raise Exception("bn.edges is empty")
#     if not isinstance(y, pd.Series):
#         # todo
#         raise Exception("y not a pd.Series")
#     if not issubclass(node, BaseNode):
#         # todo
#         raise Exception("node is not a subclass of BaseNode")
#
#     normal_structure = self.bn.edges
#     info = self.bn.descriptor
#     nodes = self.bn.nodes
#     target_name = str(y.name)
#
#     bl_add = [(target_name, node_name) for node_name in self.bn.nodes_names]
#     nodes += [node(target_name)]
#
#     info["types"] |= {target_name: "disc"}
#     info["signs"] = {".0": "mimic value to bypass broken check in bamt"}
#
#     self.bn.add_nodes(descriptor=info)
#
#     data[target_name] = y.to_numpy()
#
#     # noinspection PyTypeChecker
#     self.bn.add_edges(data=data,
#                       params={"init_edges": list(map(tuple, normal_structure)),
#                               "bl_add": bl_add,
#                               "remove_init_edges": False}
#                       )
#
#     return self
