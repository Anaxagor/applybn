from abc import ABC, abstractmethod
import pandas as pd


class Score(ABC):
    def __init__(self, verbose=1):
        self.verbose = verbose

    @abstractmethod
    def score(self, X: pd.DataFrame):
        pass
