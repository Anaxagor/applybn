from applybn.core.exceptions.exceptions import LibraryError


class EstimatorExc(LibraryError):
    """
    Base class for all exceptions raised by estimators."""

    pass


class NodesAutoTypingError(EstimatorExc):
    """
    Raises when auto-typing of nodes fails."""

    def __init__(self, nodes):
        """
        Initializes a BAMTAutoTypeError exception.

          Args:
            nodes: The nodes that caused the auto-typing error.

          Returns:
            None
        """
        message = f"BAMT nodes auto-typing error on {nodes}. Please check BAMT logs."
        super().__init__(message)
