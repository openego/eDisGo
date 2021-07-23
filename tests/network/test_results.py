import os
import shutil
import pandas as pd

from edisgo.network.results import Results
from edisgo.tools.config import Config


class TestResults:
    """
    Tests Results class.

    """
    @classmethod
    def setup_class(self):
        self.config = Config()
        self.results = Results(self)

    def test_to_csv(self):
        # create dummy results
        timeindex = pd.date_range(
            '2011-04-16 00:00:00', '2011-04-16 02:00:00', freq='1H')
        self.results.pfa_v_mag_pu_seed = pd.DataFrame(
            data=1.1094890328530983,
            columns=["bus1", "bus2"],
            index=timeindex,
            dtype="float64"
        )
        self.results.v_res = self.results.pfa_v_mag_pu_seed.copy()
        self.results.grid_expansion_costs = pd.DataFrame(
            data={"total_costs": [2, 3],
                  "quantity": [1, 1]
                  },
            index=["line1", "transformer2"]
        )

        cur_dir = os.getcwd()

        # test with default values
        self.results.to_csv(cur_dir)

        files_in_powerflow_results = os.listdir(
            os.path.join(cur_dir, "powerflow_results"))
        assert len(files_in_powerflow_results) == 1
        assert files_in_powerflow_results[0] == "voltages_pu.csv"
        files_in_grid_expansion_results = os.listdir(
            os.path.join(cur_dir, "grid_expansion_results"))
        assert len(files_in_grid_expansion_results) == 1
        assert files_in_grid_expansion_results[0] == "grid_expansion_costs.csv"

        shutil.rmtree(os.path.join(cur_dir, 'powerflow_results'))

        # test with save_seed=True
        self.results.to_csv(cur_dir, save_seed=True)

        files_in_powerflow_results = os.listdir(
            os.path.join(cur_dir, "powerflow_results"))
        assert len(files_in_powerflow_results) == 2
        assert "pfa_v_mag_pu_seed.csv" in files_in_powerflow_results

        shutil.rmtree(os.path.join(cur_dir, 'powerflow_results'))
        shutil.rmtree(os.path.join(cur_dir, 'grid_expansion_results'))

        # test with provided parameters and reduce memory True
        self.results.pfa_p = self.results.v_res
        self.results.to_csv(
            cur_dir,
            parameters={'powerflow_results': ['v_res']},
            reduce_memory=True
        )

        files_in_powerflow_results = os.listdir(
            os.path.join(cur_dir, "powerflow_results"))
        assert len(files_in_powerflow_results) == 1
        assert "voltages_pu.csv" in files_in_powerflow_results
        assert not os.path.isfile(
            os.path.join(
                cur_dir, "grid_expansion_results", "grid_expansion_costs.csv"
            )
        )
        assert self.results.v_res.bus1.dtype == "float32"

        shutil.rmtree(os.path.join(cur_dir, 'powerflow_results'))

        os.remove(os.path.join(cur_dir, 'configs.csv'))
        os.remove(os.path.join(cur_dir, 'measures.csv'))

    def test_from_csv(self):
        # create dummy results and save to csv
        timeindex = pd.date_range(
            '2011-04-16 00:00:00', '2011-04-16 02:00:00', freq='1H')
        pfa_v_mag_pu_seed = pd.DataFrame(
            data=1.1094890328530983,
            columns=["bus1", "bus2"],
            index=timeindex,
            dtype="float64"
        )
        self.results.pfa_v_mag_pu_seed = pfa_v_mag_pu_seed
        self.results.v_res = self.results.pfa_v_mag_pu_seed.copy()
        grid_expansion_costs = pd.DataFrame(
            data={"total_costs": [2, 3],
                  "quantity": [1, 1]
                  },
            index=["line1", "transformer2"]
        )
        self.results.grid_expansion_costs = grid_expansion_costs
        self.results.measures = "test"

        cur_dir = os.getcwd()
        self.results.to_csv(cur_dir, save_seed=True)

        # reset self.results
        self.results = Results(self)

        # test with default values
        self.results.from_csv(cur_dir)

        pd.testing.assert_frame_equal(
            self.results.pfa_v_mag_pu_seed, pfa_v_mag_pu_seed,
            check_freq=False
        )
        pd.testing.assert_frame_equal(
            self.results.v_res, pfa_v_mag_pu_seed,
            check_freq=False
        )
        pd.testing.assert_frame_equal(
            self.results.grid_expansion_costs, grid_expansion_costs
        )
        assert self.results.measures == ["original", "test"]

        # reset self.results
        self.results = Results(self)

        # test with given parameters
        self.results.from_csv(
            cur_dir,
            parameters={'powerflow_results': ['v_res']}
        )

        pd.testing.assert_frame_equal(
            self.results.v_res, pfa_v_mag_pu_seed,
            check_freq=False
        )
        assert self.results.pfa_v_mag_pu_seed.empty
        assert self.results.grid_expansion_costs.empty

        shutil.rmtree(os.path.join(cur_dir, 'powerflow_results'))
        shutil.rmtree(os.path.join(cur_dir, 'grid_expansion_results'))

        os.remove(os.path.join(cur_dir, 'configs.csv'))
        os.remove(os.path.join(cur_dir, 'measures.csv'))
