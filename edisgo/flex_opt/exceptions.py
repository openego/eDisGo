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
    Exception raised when OPF can not be solved.

    Attributes
    -----------
    message : str
        Explanation of the error

    """

    def __init__(self, message):
        self.message = message
