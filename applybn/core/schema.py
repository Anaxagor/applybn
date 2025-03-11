from typing import TypedDict, Optional, Sequence, Tuple, List, Union, Callable, TypeVar, Literal
from applybn.anomaly_detection.scores.score import Score


# bamt inner parameters
class StructureLearnerParams(TypedDict, total=False):
    init_edges: Optional[Sequence[str]]
    init_nodes: Optional[List[str]]
    remove_init_edges: bool
    white_list: Optional[Tuple[str, str]]
    bl_add: Optional[List[str]]

# parameters for bamt
class ParamDict(TypedDict, total=False):
    scoring_function: Union[Tuple[str, Callable], Tuple[str]]
    progress_bar: bool
    classifier: Optional[object]
    regressor: Optional[object]
    params: Optional[StructureLearnerParams]
    optimizer: str

# parameters for BNEstimator
class BNEstimatorParams(TypedDict, total=False):
    has_logit: bool
    use_mixture: bool
    bn_type: Optional[str]
    partial: Union[False, Literal["parameters", "structure"]]
    learning_params: Optional[ParamDict]


scores = TypeVar("scores", bound=Score)
