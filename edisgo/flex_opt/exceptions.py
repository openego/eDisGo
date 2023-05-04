class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class MaximumIterationError(Error):
    """
    Exception raised when maximum number of iterations in network reinforcement
    is exceeded.

    Attributes
    -----------
    message : str
        Explanation of the error

    """

    def __init__(self, message):
        self.message = message


class ImpossibleVoltageReduction(Error):
    """
    Exception raised when voltage issue cannot be solved.

    Attributes
    -----------
    message : str
        Explanation of the error

    """

    def __init__(self, message):
        self.message = message


class InfeasibleModelError(Error):
    """
    Exception raised when OPF can not be solved.

    Attributes
    -----------
    message : str
        Explanation of the error

    """

    def __init__(self, message):
        self.message = message
