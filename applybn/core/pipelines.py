from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class CorePipeline(Pipeline):
    """
    Pipeline modification to better experience. Send all getattr to last element if none is found in self.
    """
    def __getattr__(self, attr):
        """If attribute is not found in the pipeline, look in the last step of the pipeline."""
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            last_step = self.steps[-1][1]
            check_is_fitted(self) # actually calls check_is_fitted(self.steps[-1][1])
            return getattr(last_step, attr)


