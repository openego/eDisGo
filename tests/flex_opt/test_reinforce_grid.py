import copy

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt.reinforce_grid import reinforce_grid


class TestReinforceGrid:
    """
    Here, currently only reinforce_grid function is tested.
    Other functions in reinforce_grid module are currently tested in test_edisgo module.
    """

    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

        cls.edisgo.set_time_series_worst_case_analysis()

    def test_reinforce_grid(self):
        modes = [None, "mv", "mvlv", "lv"]

        results_dict = {
            mode: reinforce_grid(edisgo=copy.deepcopy(self.edisgo), mode=mode)
            for mode in modes
        }

        for mode, result in results_dict.items():
            if mode is None:
                target = ["mv/lv", "mv", "lv"]
            elif mode == "mv":
                target = ["mv"]
            elif mode == "mvlv":
                target = ["mv", "mv/lv"]
            elif mode == "lv":
                target = ["mv/lv", "lv"]
            else:
                raise ValueError("Non existing mode")

            assert_array_equal(
                np.sort(target),
                np.sort(result.grid_expansion_costs.voltage_level.unique()),
            )

            for comparison_mode, comparison_result in results_dict.items():
                if mode != comparison_mode:
                    with pytest.raises(AssertionError):
                        assert_frame_equal(
                            result.equipment_changes,
                            comparison_result.equipment_changes,
                        )
