from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
import inspect

factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__


class StaticEstimator(estimator_with_default_interface):
    """
    A class that provides a static estimate without learning from data.

        Currently, this class serves as a placeholder and does not perform any
        estimation or learning. It is designed to be extended with more
        sophisticated estimation logic in the future.
    """

    def __init__(self):
        """
        Initializes the class.

          This method serves as the constructor for the class,
          currently performing no specific initialization steps.

          Args:
            None

          Returns:
            None
        """
        pass


my_estimator = StaticEstimator()

print(*inspect.getmembers(my_estimator), sep="\n")  # check that all methods are in
