import logging

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.io import storage_import


class TestStorageImport:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

    def setup_home_batteries_data(self):
        df = pd.DataFrame(
            data={
                "p_nom": [0.005, 0.15, 2.0],
                "capacity": [1.0, 1.0, 1.0],
                "building_id": [446651, 445710, 446933],
            },
            index=[1, 2, 3],
        )
        return df

    @pytest.mark.local
    def test_oedb(self, caplog):
        # test without new PV rooftop plants
        with caplog.at_level(logging.DEBUG):
            integrated_storages = storage_import.home_batteries_oedb(
                self.edisgo, scenario="eGon2035", engine=pytest.engine
            )
        storage_df = self.edisgo.topology.storage_units_df
        assert len(integrated_storages) == 666
        assert len(storage_df) == 666
        assert np.isclose(storage_df.p_nom.sum(), 2.02723, atol=1e-3)
        assert "2.03 MW of home batteries integrated." in caplog.text
        assert (
            "Of this 2.03 MW do not have a generator with the same building ID."
            in caplog.text
        )
        caplog.clear()

        # test with new PV rooftop plants
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        self.edisgo.import_generators(
            generator_scenario="eGon2035", engine=pytest.engine
        )
        with caplog.at_level(logging.DEBUG):
            integrated_storages = storage_import.home_batteries_oedb(
                self.edisgo, scenario="eGon2035", engine=pytest.engine
            )
        storage_df = self.edisgo.topology.storage_units_df
        assert len(integrated_storages) == 666
        assert len(storage_df) == 666
        assert np.isclose(storage_df.p_nom.sum(), 2.02723, atol=1e-3)
        assert "2.03 MW of home batteries integrated." in caplog.text
        assert "do not have a generator with the same building ID." not in caplog.text

    def test__grid_integration(self, caplog):

        # ############### test without PV rooftop ###############

        # manipulate bus of the largest storage to be an MV bus
        loads_df = self.edisgo.topology.loads_df
        bus_bat_voltage_level_5_building = loads_df[loads_df.building_id == 446933].bus[
            0
        ]
        self.edisgo.topology.buses_df.at[
            bus_bat_voltage_level_5_building, "v_nom"
        ] = 20.0

        with caplog.at_level(logging.DEBUG):
            integrated_bat_1 = storage_import._home_batteries_grid_integration(
                self.edisgo, self.setup_home_batteries_data()
            )

        storage_df = self.edisgo.topology.storage_units_df
        assert len(storage_df) == 3
        # check that smallest storage is integrated at same bus as building
        bus_bat_voltage_level_7 = storage_df[storage_df.p_nom == 0.005].bus[0]
        assert (
            loads_df[loads_df.building_id == 446651].bus.values
            == bus_bat_voltage_level_7
        ).all()
        # check that medium storage cannot be integrated at same bus as building
        bus_bat_voltage_level_6 = storage_df[storage_df.p_nom == 0.15].bus[0]
        line_bat_voltage_level_6 = self.edisgo.topology.lines_df[
            self.edisgo.topology.lines_df.bus1 == bus_bat_voltage_level_6
        ]
        assert (
            line_bat_voltage_level_6.bus0[0]
            in self.edisgo.topology.transformers_df.bus1.values
        )
        # check that largest storage can be connected to building because the building
        # is already connected to the MV
        bus_bat_voltage_level_5 = storage_df[storage_df.p_nom == 2.0].bus[0]
        assert bus_bat_voltage_level_5 == bus_bat_voltage_level_5_building

        assert "2.15 MW of home batteries integrated." in caplog.text
        assert (
            "Of this 2.15 MW do not have a generator with the same building ID."
            in caplog.text
        )
        assert (
            "0.15 MW of home battery capacity was integrated at a new bus."
            in caplog.text
        )

        # check of duplicated names
        integrated_bat_2 = storage_import._home_batteries_grid_integration(
            self.edisgo, self.setup_home_batteries_data()
        )
        storage_df = self.edisgo.topology.storage_units_df
        assert len(storage_df) == 6
        assert len([_ for _ in integrated_bat_2 if _ not in integrated_bat_1]) == 3

        caplog.clear()

        # ############### test with PV rooftop ###############

        # set up PV data - first one is at same bus as building, second one at higher
        # voltage level
        pv_df = pd.DataFrame(
            data={
                "bus": [
                    "BranchTee_mvgd_33535_lvgd_1164120011_building_442002",
                    "BusBar_mvgd_33535_lvgd_1164120011_LV",
                ],
                "p_nom": [0.005, 0.15],
                "type": ["solar", "solar"],
                "building_id": [442002, 445710],
            },
            index=[1, 2],
        )
        self.edisgo.topology.generators_df = pd.concat(
            [self.edisgo.topology.generators_df, pv_df]
        )
        with caplog.at_level(logging.DEBUG):
            integrated_bat_3 = storage_import._home_batteries_grid_integration(
                self.edisgo, self.setup_home_batteries_data()
            )
        storage_df = self.edisgo.topology.storage_units_df.loc[integrated_bat_3, :]
        assert len(self.edisgo.topology.storage_units_df) == 9

        # check that smallest storage is integrated at same bus as PV system
        bus_bat_voltage_level_7 = storage_df[storage_df.p_nom == 0.005].bus[0]
        assert (
            loads_df[loads_df.building_id == 446651].bus.values
            == bus_bat_voltage_level_7
        ).all()
        # check that medium storage is integrated at same bus as PV system
        bus_bat_voltage_level_6 = storage_df[storage_df.p_nom == 0.15].bus[0]
        assert "BusBar_mvgd_33535_lvgd_1164120011_LV" == bus_bat_voltage_level_6
        # check that largest storage can be connected to building because the building
        # is already connected to the MV
        bus_bat_voltage_level_5 = storage_df[storage_df.p_nom == 2.0].bus[0]
        assert bus_bat_voltage_level_5 == bus_bat_voltage_level_5_building

        assert "2.15 MW of home batteries integrated." in caplog.text
        assert (
            "Of this 2.00 MW do not have a generator with the same building ID."
            in caplog.text
        )
        assert (
            "of home battery capacity was integrated at a new bus." not in caplog.text
        )
