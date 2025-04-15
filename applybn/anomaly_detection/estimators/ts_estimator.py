from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
import inspect

factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__


class StaticEstimator(estimator_with_default_interface):
    def __init__(self):
        pass


my_estimator = StaticEstimator()

print(*inspect.getmembers(my_estimator), sep="\n")  # check that all methods are in
