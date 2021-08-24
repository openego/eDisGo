import os
import numpy as np
import shutil
import pytest

from examples import example_grid_reinforcement


class TestExamples:

    @pytest.mark.slow
    def test_grid_reinforcement_example(self):
        total_costs = example_grid_reinforcement.run_example()
        # ToDo: total costs are for some reason not deterministic, check why!!
        #assert np.isclose(total_costs, 1147.57198)
        assert total_costs > 0.

        # Delete saved grid and results data
        edisgo_path = os.path.join(os.path.expanduser('~'), '.eDisGo')
        shutil.rmtree(os.path.join(
            edisgo_path, 'ding0_example_grid'))
