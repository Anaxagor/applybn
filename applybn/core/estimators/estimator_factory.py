from typing import Unpack
from applybn.core.estimators.base_estimator import BNEstimator
from applybn.core.bamt_wrappers import BamtPreprocessorWrapper
from applybn.core.schema import BNEstimatorParams
from applybn.core.pipelines import CorePipeline

from sklearn.base import ClassifierMixin, RegressorMixin

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp


class EstimatorPipelineFactory:
    """
    Factory class to create an estimator pipeline for classification or regression tasks.

    Attributes:
        interfaces (dict): Mapping of task types to their corresponding scikit-learn mixin classes.
        task_type (str): The type of task ('classification' or 'regression').
        estimator_ (None | BaseEstimator): The estimator instance.
    """

    interfaces = {"classification": ClassifierMixin, "regression": RegressorMixin}

    def __init__(self, task_type: str = "classification"):
        """
        Initializes the EstimatorPipelineFactory with the given task type.

        Args:
            task_type: The type of task ('classification' or 'regression').
        """
        self.task_type = task_type
        self.estimator_ = None

    @staticmethod
    def convert_bamt_preprocessor(preprocessor: list):
        """
        Converts a BAMT preprocessor to a BamtPreprocessorWrapper.

        Args:
            preprocessor: The BAMT preprocessor to convert.

        Returns:
            BamtPreprocessorWrapper: The wrapped preprocessor.
        """
        return BamtPreprocessorWrapper(preprocessor)

    def __call__(
        self, preprocessor: None | list = None, **params: Unpack[BNEstimatorParams]
    ):
        """
        Creates a pipeline with the given preprocessor and parameters.

        Args:
            preprocessor: The preprocessor to use (default is None).
            **params: Parameters for the BNEstimator.

        Returns:
            CorePipeline: The constructed pipeline.
        """
        if preprocessor is None:
            preprocessor = self.default_preprocessor

        self.estimator.set_params(**params)

        wrapped_preprocessor = self.convert_bamt_preprocessor(preprocessor)

        pipeline = CorePipeline(
            [("preprocessor", wrapped_preprocessor), ("bn_estimator", self.estimator)]
        )
        return pipeline

    def _adjust_interface(self) -> None:
        """
        Adjusts the interface of the estimator based on the task type.
        """
        interface = self.interfaces[self.task_type]
        names = {
            "regression": "RegressorMixin",
            "classification": "ClassifierMixin",
        }
        new_class = type(
            f"BNEstimatorWith{names[self.task_type]}", (BNEstimator, interface), {}
        )
        self.estimator_ = new_class()

    @property
    def estimator(self):
        """
        Returns the estimator instance, creating it if necessary.

        Returns:
            BNEstimatorMixin: The estimator instance with classifier or regressor interface.
        """
        if not self.estimator_:
            self._adjust_interface()
        return self.estimator_

    @property
    def default_preprocessor(self):
        """
        Returns a default preprocessor pipeline.

            This method creates and returns a Preprocessor object configured with a
            LabelEncoder for categorical features and a KBinsDiscretizer for numerical
            features, using uniform width binning with 5 bins.

            Args:
                None

            Returns:
                Preprocessor: A preprocessor pipeline ready to be fit and transformed on data.
        """
        encoder = pp.LabelEncoder()
        discretizer = pp.KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="uniform"
        )
        preprocessor = Preprocessor(
            [("encoder", encoder), ("discretizer", discretizer)]
        )
        return preprocessor
