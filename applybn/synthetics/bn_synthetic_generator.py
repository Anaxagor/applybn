from pandas import DataFrame
from applybn.utils.bn_wrapper import get_bn




class BNSyntheticGenerator:
    """Bayesian Network Synthetic Data Generator."""

    def __init__(self):
        self.bn = None

    def fit(self, data: DataFrame):
        self.bn = get_bn(data)
