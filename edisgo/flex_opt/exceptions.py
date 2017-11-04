class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class MaximumIterationError(Error):
    """
    Exception raised when maximum number of iterations in grid reinforcement
    is exceeded.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class ImpossibleVoltageReduction(Error):
    """
    Exception raised when voltage issue cannot be solved.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message