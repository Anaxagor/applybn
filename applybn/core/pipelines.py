from sklearn.pipeline import Pipeline

class CorePipeline(Pipeline):
    """
    Pipeline modification to better experience. Send all getattr to last element if none is found in self.
    """
    def __getattr__(self, attr):
        """If attribute is not found in the pipeline, look in the last step of the pipeline."""
        try:
            return object.__getattribute__(self, attr)  # Try getting from self first
        except AttributeError:
            if self.steps:  # Ensure pipeline is not empty
                last_step = self.steps[-1][1]  # Get the last estimator
                if hasattr(last_step, attr):
                    return getattr(last_step, attr)  # Delegate lookup
            raise  # Raise if the attribute is not found anywhere

