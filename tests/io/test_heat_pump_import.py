import logging

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.io import heat_pump_import
from edisgo.tools.tools import determine_bus_voltage_level


@pytest.fixture
def mock_oep_heat_pump_query(monkeypatch):
    """Replace one heat-pump OEP query with synthetic return data."""

    def install_mock(function_name, result):
        query_mock = (
            Mock(side_effect=result) if callable(result) else Mock(return_value=result)
        )
        monkeypatch.setattr(heat_pump_import, function_name, query_mock)
        return query_mock

    return install_mock


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

    def setup_resistive_heater_data_dh(self):
        geom = Point((10.02178787570608, 47.55650888787377))
        hp_df = pd.DataFrame(
            data={
                "p_set": [21.0, 0.17],
                "weather_cell_id": [11051, 11051],
                "district_heating_id": [5, 6],
                "area_id": [4, 5],
                "geom": [None, geom],
            },
            index=[1, 2],
        )
        return hp_df

    def test_oedb_offline(self, caplog, mock_oep_heat_pump_query):
        first_edisgo = self.edisgo
        hp_individual = self.setup_heat_pump_data_individual_heating()
        hp_central = self.setup_heat_pump_data_dh()
        resistive_heaters_central = self.setup_resistive_heater_data_dh()
        resistive_heaters_central.at[1, "p_set"] = 0.1
        hp_individual_cap = hp_individual.p_set.sum()

        def query_data(*args, **kwargs):
            import_types = kwargs["import_types"]
            if import_types is None:
                return (
                    hp_individual.copy(deep=True),
                    hp_central.copy(deep=True),
                    resistive_heaters_central.copy(deep=True),
                    hp_individual_cap,
                )
            if import_types == ["central_heat_pumps"]:
                return (
                    pd.DataFrame(columns=["p_set"]),
                    hp_central.iloc[[0]].copy(deep=True),
                    pd.DataFrame(columns=["p_set"]),
                    hp_individual_cap,
                )
            raise ValueError(f"Unexpected import types: {import_types}")

        query_mock = mock_oep_heat_pump_query(
            "_query_heat_pump_data_oedb",
            query_data,
        )

        with caplog.at_level(logging.DEBUG):
            heat_pump_import.oedb(
                self.edisgo,
                scenario="eGon2035",
                engine=None,
            )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert "Capacity of individual heat pumps" not in caplog.text
        assert len(hp_df) == 8
        assert len(hp_df[hp_df.sector == "individual_heating"]) == 3
        assert np.isclose(
            hp_df[hp_df.sector == "individual_heating"].p_set.sum(),
            hp_individual.p_set.sum(),
        )
        dh_hp = hp_df[hp_df.sector == "district_heating"]
        assert len(dh_hp) == 3
        assert np.isclose(dh_hp.p_set.sum(), hp_central.p_set.sum())
        dh_rh = hp_df[hp_df.sector == "district_heating_resistive_heater"]
        assert len(dh_rh) == 2
        assert np.isclose(dh_rh.p_set.sum(), resistive_heaters_central.p_set.sum())
        # central heat pumps and the resistive heater in the same district-heating
        # network are integrated at the same bus
        dh_hp_id_5 = dh_hp[dh_hp.district_heating_id == 5]
        dh_rh_id_5 = dh_rh[dh_rh.district_heating_id == 5]
        assert dh_rh_id_5.bus.iat[0] in dh_hp_id_5.bus.values

        # test without resistive heaters and individual heat pumps
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        heat_pump_import.oedb(
            self.edisgo,
            scenario="eGon2035",
            engine=None,
            import_types=["central_heat_pumps"],
        )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == 1
        assert len(hp_df[hp_df.sector == "district_heating"]) == 1
        # assert central heat pump in voltage level 7
        assert determine_bus_voltage_level(self.edisgo, hp_df.bus.iat[0]) == 7

        assert query_mock.call_count == 2
        assert query_mock.call_args_list[0].kwargs == {
            "edisgo_object": first_edisgo,
            "scenario": "eGon2035",
            "engine": None,
            "import_types": None,
        }
        assert query_mock.call_args_list[1].kwargs == {
            "edisgo_object": self.edisgo,
            "scenario": "eGon2035",
            "engine": None,
            "import_types": ["central_heat_pumps"],
        }

    @pytest.mark.oep
    def test_query_heat_pump_data_oedb_live(self, oep_engine):
        (
            hp_individual,
            hp_central,
            resistive_heaters_central,
            hp_individual_cap,
        ) = heat_pump_import._query_heat_pump_data_oedb(
            edisgo_object=self.edisgo,
            scenario="eGon2035",
            engine=oep_engine,
            query_limit=2,
        )

        assert 0 < len(hp_individual) <= 2
        assert len(hp_central) <= 2
        assert len(resistive_heaters_central) <= 2
        assert {"building_id", "p_set", "weather_cell_id"}.issubset(
            hp_individual.columns
        )
        for central_data in [hp_central, resistive_heaters_central]:
            if not central_data.empty:
                assert {
                    "p_set",
                    "weather_cell_id",
                    "district_heating_id",
                    "geom",
                    "area_id",
                }.issubset(central_data.columns)
        assert hp_individual_cap > 0

    def test__grid_integration(self, caplog):
        # ############# test integration of central heat pumps ####################
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=pd.DataFrame(),
            hp_central=self.setup_heat_pump_data_dh(),
            resistive_heaters_central=pd.DataFrame(),
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
        self.edisgo.topology.buses_df.at[bus_hp_voltage_level_5_building, "v_nom"] = (
            20.0
        )
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=self.setup_heat_pump_data_individual_heating(),
            hp_central=pd.DataFrame(),
            resistive_heaters_central=pd.DataFrame(),
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
            resistive_heaters_central=pd.DataFrame(),
        )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == 9

        # ############# test integration of central resistive heaters #################
        heat_pump_import._grid_integration(
            self.edisgo,
            hp_individual=pd.DataFrame(),
            hp_central=pd.DataFrame(),
            resistive_heaters_central=self.setup_resistive_heater_data_dh(),
        )
        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.sector == "district_heating_resistive_heater"]
        assert len(hp_df) == 2
        # check that resistive heater in same district heating network as heat pumps
        # is integrated at same bus
        bus_rh = hp_df[hp_df.p_set == 21.0].bus[0]
        assert bus_rh in loads_df[loads_df.sector == "district_heating"].bus.values
        # check that resistive heater in other district heating network is integrated
        # in voltage level 6
        bus_rh = hp_df[hp_df.p_set == 0.17].bus[0]
        assert determine_bus_voltage_level(self.edisgo, bus_rh) == 6

    def test_efficiency_resistive_heaters_oedb_offline(
        self,
        mock_oep_heat_pump_query,
    ):
        query_mock = mock_oep_heat_pump_query(
            "_query_resistive_heater_efficiency_oedb",
            {
                "efficiency": {
                    "central_resistive_heater": 0.99,
                    "rural_resistive_heater": 0.9,
                }
            },
        )

        eta_dict = heat_pump_import.efficiency_resistive_heaters_oedb(
            scenario="eGon2035",
            engine=None,
        )

        assert eta_dict["central_resistive_heater"] == 0.99
        assert eta_dict["rural_resistive_heater"] == 0.9
        query_mock.assert_called_once_with(scenario="eGon2035", engine=None)

    @pytest.mark.oep
    def test_query_resistive_heater_efficiency_oedb_live(self, oep_engine):
        heat_parameters = heat_pump_import._query_resistive_heater_efficiency_oedb(
            scenario="eGon2035",
            engine=oep_engine,
        )

        assert "efficiency" in heat_parameters
        assert {
            "central_resistive_heater",
            "rural_resistive_heater",
        }.issubset(heat_parameters["efficiency"])
        assert all(
            0 < heat_parameters["efficiency"][key] <= 1
            for key in ["central_resistive_heater", "rural_resistive_heater"]
        )
