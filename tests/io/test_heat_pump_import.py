import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.io import heat_pump_import


class TestHeatPumpImport:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

    @pytest.mark.local
    def test_oedb(self, caplog):
        heat_pump_import.oedb(self.edisgo, scenario="eGon2035", engine=pytest.engine)
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert "Capacity of individual heat pumps" not in caplog.text
        assert len(hp_df) == 177
        assert len(hp_df[hp_df.sector == "individual_heating"]) == 176
        assert np.isclose(
            hp_df[hp_df.sector == "individual_heating"].p_set.sum(), 2.97388
        )
        assert len(hp_df[hp_df.sector == "district_heating"]) == 1
        assert np.isclose(
            hp_df[hp_df.sector == "district_heating"].p_set.sum(), 0.095348
        )
