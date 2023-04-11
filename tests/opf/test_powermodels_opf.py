import pytest

from edisgo import EDisGo
from edisgo.opf.powermodels_opf import pm_optimize


class TestPowerModelsOPF:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex

    def test_pm_optimize(self):

        pm_optimize(self.edisgo, opf_version=2, silence_moi=True)

        print(" ")
        # assert
