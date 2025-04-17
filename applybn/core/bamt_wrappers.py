from sklearn.base import BaseEstimator, TransformerMixin


class BamtPreprocessorWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper class for BAMT preprocessors to make them compatible with scikit-learn pipelines.

    Attributes:
        preprocessor: The BAMT preprocessor instance.
        coder_: The coder used in the transformation process.
    """

    def __init__(self, preprocessor):
        """
        Initializes the BamtPreprocessorWrapper with the given preprocessor.

        Args:
            preprocessor: The BAMT preprocessor to wrap.
        """
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the data.

        Args:
            X: The input data.
            y: The target values (default is None).
                Ignored, used only for conventional scikit-learn compatibility.

        Returns:
            self: The fitted preprocessor wrapper.
        """
        return self

    def transform(self, X):
        """
        Transforms the input data using the wrapped preprocessor.

        Args:
            X: The input data to transform.

        Returns:
            tuple: A tuple containing the transformed data, the coder, and the original data.
        """
        df, coder = self.preprocessor.apply(X)
        self.coder_ = coder

        return df, self.info, X

    def __getattr__(self, attr):
        """
        Forwards attribute access to the wrapped preprocessor.

        Args:
            attr: The attribute to access.

        Returns:
            The value of the requested attribute from the wrapped preprocessor.
        """
        return getattr(self.preprocessor, attr)
