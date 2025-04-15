from applybn.core.pipelines import CorePipeline
from functools import wraps
from copy import deepcopy


class AnomalyDetectionPipeline(CorePipeline):
    @staticmethod
    def _score_context(method):
        """A decorator to temporarily disable the preprocessor step in the pipeline during score."""

        @wraps(method)
        def wrapper(pipeline, *args, **kwargs):
            original_first_step = deepcopy(pipeline.steps[0])
            try:
                # Temporarily disable the first step
                pipeline.steps[0] = ["preprocessor", "passthrough"]
                return method(pipeline, *args, **kwargs)
            finally:
                # Restore the original step
                pipeline.steps[0] = original_first_step

        return wrapper

    @_score_context
    def score(self, X, y=None, sample_weight=None, **params):
        return super(AnomalyDetectionPipeline, self).score(
            X, y, sample_weight, **params
        )

    @classmethod
    def from_core_pipeline(cls, core_pipeline: CorePipeline):
        return cls(core_pipeline.steps)

    @property
    def encoding(self):
        return self.steps[0][1].coder
