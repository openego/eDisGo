import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from edisgo.tools import tools


class TestTools:

    def test_calculate_line_reactance(self):
        data = tools.calculate_line_reactance(2, 3)
        assert np.isclose(data, 1.88496)
        data = tools.calculate_line_reactance(np.array([2, 3]), 3)
        assert_allclose(data, np.array([1.88496, 2.82743]), rtol=1e-5)

    def test_calculate_line_resistance(self):
        data = tools.calculate_line_resistance(2, 3)
        assert data == 6
        data = tools.calculate_line_resistance(np.array([2, 3]), 3)
        assert_array_equal(data, np.array([6, 9]))

    def test_calculate_apparent_power(self):
        data = tools.calculate_apparent_power(20, 30)
        assert np.isclose(data, 1.03923)
        data = tools.calculate_apparent_power(30, np.array([20, 30]))
        assert_allclose(data, np.array([1.03923, 1.55884]), rtol=1e-5)
        data = tools.calculate_apparent_power(np.array([30, 30]),
                                              np.array([20, 30]))
        assert_allclose(data, np.array([1.03923, 1.55884]), rtol=1e-5)
