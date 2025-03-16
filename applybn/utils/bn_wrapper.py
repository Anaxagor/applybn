from bamt.preprocessors import Preprocessor
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.discrete_bn import DiscreteBN
from copy import copy
import pandas as pd

def get_bn(
    data: pd.DataFrame,
    n_bins: int = 5,
    has_mixture: bool = True
) -> HybridBN | DiscreteBN:
    """Builds and returns a Bayesian Network (BN) after preprocessing the input data.
    This function preprocesses the data using discretization and encoding, determines the
    appropriate BN type (hybrid or discrete) based on data types, structures the BN with nodes
    and edges, and fits parameters to the data.
    Args:
        data: Input dataset containing continuous and/or discrete features.
        n_bins: Number of bins to use for discretizing continuous features.
        has_mixture: Whether to use mixture models for continuous variables in HybridBN. Ignored
            if the BN is discrete.
    Returns:
        Configured Bayesian Network. HybridBN is returned if continuous features are detected,
        otherwise DiscreteBN is used.
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"A": [1, 2, 3], "B": ["X", "Y", "X"]})
        >>> bn = get_bn(data, n_bins=5, has_mixture=False)
        >>> bn.type  # Returns 'Discrete' based on data types
    Note:
        - Preprocessing: Combines LabelEncoder (for categoricals) and KBinsDiscretizer (for
          continuous features) via a Preprocessor
        - Structure Learning: Uses the K2 scoring function to determine edges
        - HybridBN: Used when continuous features are detected. DiscreteBN is a fallback
    """
    cur_data = copy(data)
    encoder = LabelEncoder()
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    pp = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
    discrete_data, _ = pp.apply(cur_data)
    nodes_data = pp.info
    nodes_types = nodes_data['types']
    has_continuous = any(value not in ['disc', 'disc_num'] for value in nodes_types)
    bn = DiscreteBN()
    if has_continuous:
        bn = HybridBN(use_mixture=has_mixture, has_logit=False)
    bn.add_nodes(nodes_data)
    bn.add_edges(data=discrete_data, scoring_function=("K2",))
    bn.fit_parameters(cur_data)
    return bn