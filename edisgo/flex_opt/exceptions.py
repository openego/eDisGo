# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later


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
    Exception raised when the solver of the OPF finds no solution.

    The Julia process ran and the solver ended with a status, for example
    ``INFEASIBLE``, ``NUMERICAL_ERROR`` or ``TIME_LIMIT``. A failure of the Julia
    process itself raises :class:`OPFProcessError` instead.

    Attributes
    -----------
    message : str
        Explanation of the error
    status : str or None
        Termination status reported by the solver, or None if it is unknown.

    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        self.status = status


class OPFProcessError(Error):
    """
    Exception raised when the Julia process of the OPF fails before the solver
    reports a status.

    Examples are an error while building the model, a crash, or the process being
    killed, e.g. by the out-of-memory killer.

    Attributes
    -----------
    message : str
        Explanation of the error
    returncode : int or None
        Return code of the Julia process as given by
        :attr:`subprocess.Popen.returncode`. A negative value ``-N`` means that the
        process was killed by signal ``N``.

    """

    def __init__(self, message, returncode=None):
        super().__init__(message)
        self.message = message
        self.returncode = returncode
