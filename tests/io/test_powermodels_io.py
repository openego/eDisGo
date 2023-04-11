import pytest

from edisgo import EDisGo
from edisgo.io import powermodels_io


class TestPowermodelsIO:
    def test_to_powermodels(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)

        assert len(powermodels_network["gen"].keys()) == 1 + 1
        assert len(powermodels_network["gen_slack"].keys()) == 1
        assert len(powermodels_network["gen_nd"].keys()) == 27
        assert len(powermodels_network["bus"].keys()) == 142
        assert len(powermodels_network["branch"].keys()) == 131
        assert len(powermodels_network["load"].keys()) == 50 + 1
        assert len(powermodels_network["storage"]).keys() == 0

        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(
            self.edisgo,
            flexible_storage_units=self.edisgo.topology.storage_units_df.index.values,
        )
        assert len(powermodels_network["gen"].keys()) == 1
        assert len(powermodels_network["gen_slack"].keys()) == 1
        assert len(powermodels_network["gen_nd"].keys()) == 27
        assert len(powermodels_network["bus"].keys()) == 142
        assert len(powermodels_network["branch"].keys()) == 131
        assert len(powermodels_network["load"].keys()) == 50
        assert len(powermodels_network["storage"]).keys() == 1

        # ToDo: test more options with test network including all flexibilities

    def test__get_pf(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

        # test mode None
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)
        for component in ["gen", "storage"]:
            pf, sign = powermodels_io._get_pf(
                self.edisgo, powermodels_network, 1, component
            )
            assert pf == 0.9
            assert sign == -1
            pf, sign = powermodels_io._get_pf(
                self.edisgo, powermodels_network, 29, component
            )
            assert pf == 0.95
            assert sign == -1

        for component in ["hp", "cp"]:
            for bus in [1, 29]:
                pf, sign = powermodels_io._get_pf(
                    self.edisgo, powermodels_network, 1, component
                )
                assert pf == 1
                assert sign == 1
