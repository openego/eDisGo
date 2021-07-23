import os
import numpy as np
import shutil
import pytest

from examples import example_grid_reinforcement


class TestExamples:

    @pytest.mark.slow
    def test_grid_reinforcement_example(self):
        total_costs = example_grid_reinforcement.run_example()
        assert np.isclose(total_costs, 1147.57198)

        # Delete saved grid and results data
        parent_dir = os.path.dirname(os.getcwd())
        shutil.rmtree(os.path.join(
            parent_dir, 'examples', 'ding0_example_grid'))
