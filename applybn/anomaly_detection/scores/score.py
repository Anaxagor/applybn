from abc import ABC, abstractmethod
import pandas as pd


class Score(ABC):
    """
    An abstract base class for implementing scoring mechanisms.
    """

    def __init__(self, verbose:int=1):
        """
        Initializes the Score object.

        Args:
            verbose: The verbosity level for logging. Default is 1.
        """
        self.verbose = verbose

    @abstractmethod
    def score(self, X: pd.DataFrame):
        """
        Abstract method to compute scores for the given input data.

        Args:
            X: The input data to be scored.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
