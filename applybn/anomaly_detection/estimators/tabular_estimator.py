from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
from applybn.core.schema import scores

factory = EstimatorPipelineFactory(task_type="classification")

estimator_with_default_interface = factory.estimator.__class__

class TabularEstimator(estimator_with_default_interface):
    """
    A custom tabular estimator that extends the default estimator interface.

    Attributes:
        scorer: An optional scoring object used to calculate scores for the input data.
    """

    def __init__(self, scorer: scores =None):
        """
        Initializes the TabularEstimator.

        Args:
            scorer: An optional scoring object used to calculate scores for the input data.
        """
        self.scorer = scorer
        super().__init__()

    def score(self, X, y=None, sample_weight=None, **params):
        """
        Computes the score for the input data using the provided scorer.

        Args:
            X: The input data to be scored.
            y: (Optional) The target values. Not used in this implementation.
            sample_weight: (Optional) Sample weights. Not used in this implementation.
            **params: Additional parameters to be passed to the scorer.

        Returns:
            The score computed by the scorer.
        """
        return self.scorer.score(X, **params)