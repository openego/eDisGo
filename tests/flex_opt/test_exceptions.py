"""Tests for edisgo.flex_opt.exceptions — custom exception classes."""

import pytest

from edisgo.flex_opt.exceptions import (
    InfeasibleModelError,
    ImpossibleVoltageReduction,
    MaximumIterationError,
)


class TestExceptions:
    @pytest.mark.parametrize(
        "exc_class",
        [MaximumIterationError, ImpossibleVoltageReduction, InfeasibleModelError],
    )
    def test_exception_stores_message(self, exc_class):
        e = exc_class("test error message")
        assert e.message == "test error message"

    @pytest.mark.parametrize(
        "exc_class",
        [MaximumIterationError, ImpossibleVoltageReduction, InfeasibleModelError],
    )
    def test_exception_is_raisable(self, exc_class):
        with pytest.raises(exc_class):
            raise exc_class("boom")
