from applybn.core.exceptions.exceptions import LibraryError


class EstimatorExc(LibraryError):
    pass


class NodesAutoTypingError(EstimatorExc):
    def __init__(self, nodes):
        message = f"BAMT nodes auto-typing error on {nodes}. Please check BAMT logs."
        super().__init__(message)
