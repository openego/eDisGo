import logging

import numpy as np
import pandas as pd
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.io import heat_pump_import


class TestHeatPumpImport:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

    def setup_heat_pump_data_individual_heating(self):
        hp_df = pd.DataFrame(
            data={
                "p_set": [0.005, 0.15, 2.0],
                "weather_cell_id": [11051, 11051, 11052],
                "building_id": [446963, 445710, 446933],
            },
            index=[1, 2, 3],
        )
        return hp_df

    def setup_heat_pump_data_dh(self):
        geom = Point((10.02178787570608, 47.55650888787377))
        hp_df = pd.DataFrame(
            data={
                "p_set": [0.05, 0.17, 1.0],
                "weather_cell_id": [11051, 11051, 11052],
                "district_heating_id": [5, 5, 5],
                "area_id": [4, 4, 4],
                "geom": [geom, geom, geom],
            },
            index=[1, 2, 3],
        )
        return hp_df

    @pytest.mark.local
    def test_oedb(self, caplog):
        with caplog.at_level(logging.DEBUG):
            heat_pump_import.oedb(
                self.edisgo, scenario="eGon2035", engine=pytest.engine
            )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert "Capacity of individual heat pumps" not in caplog.text
        assert len(hp_df) == 151
        assert len(hp_df[hp_df.sector == "individual_heating"]) == 150
        assert np.isclose(
            hp_df[hp_df.sector == "individual_heating"].p_set.sum(), 2.97316
        )
        assert len(hp_df[hp_df.sector == "district_heating"]) == 1
        assert np.isclose(
            hp_df[hp_df.sector == "district_heating"].p_set.sum(), 0.095202
        )

    def test__grid_integration(self, caplog):

        # ############# test integration of central heat pumps ####################
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=pd.DataFrame(),
            hp_central=self.setup_heat_pump_data_dh(),
        )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == 3
        # check that smallest heat pump is connected to LV
        bus_hp_voltage_level_7 = hp_df[hp_df.p_set == 0.05].bus[0]
        assert self.edisgo.topology.buses_df.at[bus_hp_voltage_level_7, "v_nom"] == 0.4
        # check that medium heat pump is connected to MV/LV station
        bus_hp_voltage_level_6 = hp_df[hp_df.p_set == 0.17].bus[0]
        line_hp_voltage_level_6 = self.edisgo.topology.lines_df[
            self.edisgo.topology.lines_df.bus1 == bus_hp_voltage_level_6
        ]
        assert (
            line_hp_voltage_level_6.bus0[0]
            in self.edisgo.topology.transformers_df.bus1.values
        )
        # check that largest heat pump is connected to MV
        bus_hp_voltage_level_5 = hp_df[hp_df.p_set == 1.0].bus[0]
        assert self.edisgo.topology.buses_df.at[bus_hp_voltage_level_5, "v_nom"] == 20.0

        # ############# test integration of individual heat pumps ####################

        # manipulate bus of the largest individual heat pump to be an MV bus
        loads_df = self.edisgo.topology.loads_df
        bus_hp_voltage_level_5_building = loads_df[loads_df.building_id == 446933].bus[
            0
        ]
        self.edisgo.topology.buses_df.at[
            bus_hp_voltage_level_5_building, "v_nom"
        ] = 20.0
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=self.setup_heat_pump_data_individual_heating(),
            hp_central=pd.DataFrame(),
        )

        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == 6
        # check that smallest heat pump is integrated at same bus as building
        bus_hp_voltage_level_7 = hp_df[hp_df.p_set == 0.005].bus[0]
        assert (
            loads_df[loads_df.building_id == 446963].bus.values
            == bus_hp_voltage_level_7
        ).all()
        # check that medium heat pump cannot be integrated at same bus as building
        bus_hp_voltage_level_6 = hp_df[hp_df.p_set == 0.15].bus[0]
        line_hp_voltage_level_6 = self.edisgo.topology.lines_df[
            self.edisgo.topology.lines_df.bus1 == bus_hp_voltage_level_6
        ]
        assert (
            line_hp_voltage_level_6.bus0[0]
            in self.edisgo.topology.transformers_df.bus1.values
        )
        assert len(loads_df[loads_df.building_id == 445710].bus.unique()) == 2
        # check that largest heat pump can be connected to building because the building
        # is already connected to the MV
        bus_hp_voltage_level_5 = hp_df[hp_df.p_set == 2.0].bus[0]
        assert bus_hp_voltage_level_5 == bus_hp_voltage_level_5_building

        # ######## test check of duplicated names ###########
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=self.setup_heat_pump_data_individual_heating(),
            hp_central=pd.DataFrame(),
        )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == 9
