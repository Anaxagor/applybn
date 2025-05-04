from typing import TypedDict, Sequence, Tuple, List, Union, Callable, TypeVar, Literal
from applybn.anomaly_detection.scores.score import Score
from bamt.networks import BaseNetwork


# bamt inner parameters
class StructureLearnerParams(TypedDict, total=False):
    """
    Parameters for structure learning algorithms.

        This class encapsulates parameters used in various structure learning
        algorithms, allowing for customization of the learning process.
    """

    init_edges: None | Sequence[str]
    init_nodes: None | List[str]
    remove_init_edges: bool
    white_list: None | Tuple[str, str]
    bl_add: None | List[str]


# parameters for bamt
class ParamDict(TypedDict, total=False):
    """
    A dictionary-like object to manage and store parameters for model training."""

    scoring_function: Union[Tuple[str, Callable], Tuple[str]]
    progress_bar: bool
    classifier: None | object
    regressor: None | object
    params: None | StructureLearnerParams
    optimizer: str


# parameters for BNEstimator
class BNEstimatorParams(TypedDict, total=False):
    """
    Holds parameters for the Bayesian Network Estimator."""

    has_logit: bool
    use_mixture: bool
    bn_type: None | str
    partial: Union[False, Literal["parameters", "structure"]]
    learning_params: None | ParamDict


# scores for anomaly detection module
scores = TypeVar("scores", bound=Score)
bamt_network = TypeVar("bamt_network", bound=BaseNetwork)
