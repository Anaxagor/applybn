from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory


factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__

class TabularEstimator(estimator_with_default_interface):
    def __init__(self, scorer=None):
        self.scorer = scorer
        super().__init__()

    def score(self, X, y=None, sample_weight=None, **params):
        return self.scorer.score(X, **params)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return self