import logging

from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from shapely.geometry import Point, box

from edisgo import EDisGo
from edisgo.io import heat_pump_import
from edisgo.tools.tools import determine_bus_voltage_level


@pytest.fixture
def mock_oep_heat_pump_query(monkeypatch):
    """Replace one heat-pump OEP query with synthetic return data."""

    def install_mock(function_name, result):
        query_mock = Mock(return_value=result)
        monkeypatch.setattr(heat_pump_import, function_name, query_mock)
        return query_mock

    return install_mock


def _central_query_data(plants, crs):
    """Small raw spatial tables; do not precompute the production joins."""
    locations = {
        5: Point(10.02178787570608, 47.55650888787377),
        6: Point(10.04178787570608, 47.55650888787377),
    }
    buses = [locations[district] for district in plants.district_heating_id]
    return {
        "plants": pd.DataFrame(
            {"bus0": 33535, "bus1": range(len(plants)), "p_set": plants.p_set.values}
        ),
        "heat_buses": gpd.GeoDataFrame(
            {"bus1": range(len(plants)), "geom": buses}, geometry="geom", crs=4326
        ).to_crs(crs),
        "weather_cells": gpd.GeoDataFrame(
            {"weather_cell_id": [11051], "geom": [box(9, 47, 11, 48)]},
            geometry="geom",
            crs=4326,
        ).to_crs(crs),
        "district_heating_areas": gpd.GeoDataFrame(
            {
                "district_heating_id": [5, 6],
                "area_id": [4, 5],
                "geom_polygon": [point.buffer(0.005) for point in locations.values()],
            },
            geometry="geom_polygon",
            crs=4326,
        ).to_crs(crs),
        "chp_plants": gpd.GeoDataFrame(
            {"district_heating_id": [], "geom": []}, geometry="geom", crs=crs
        ),
    }


@pytest.mark.parametrize("source_crs", [3857, 4326])
def test_prepare_central_heat_sources_offline(source_crs):
    """Check reprojection, spatial assignments and all three location choices."""
    grid = SimpleNamespace(
        topology=SimpleNamespace(
            grid_district={"srid": 3857, "geom": box(0, 0, 10000, 10000)},
            mv_grid=SimpleNamespace(station=pd.DataFrame({"x": [100], "y": [100]})),
        )
    )
    data = {
        "plants": pd.DataFrame(
            {"bus0": [1] * 4, "bus1": [1, 2, 3, 4], "p_set": [0.1, 0.2, 0.3, 0.4]}
        ),
        "heat_buses": gpd.GeoDataFrame(
            {
                "bus1": [1, 2, 3, 4],
                "geom": [
                    Point(1000, 1000),
                    Point(5100, 4000),
                    Point(11000, 7000),
                    Point(7000, 7000),
                ],
            },
            geometry="geom",
            crs=3857,
        ),
        "weather_cells": gpd.GeoDataFrame(
            {
                "weather_cell_id": [11, 22],
                "geom": [box(0, 0, 6000, 10000), box(10000, 6000, 12000, 8000)],
            },
            geometry="geom",
            crs=3857,
        ),
        "district_heating_areas": gpd.GeoDataFrame(
            {
                "district_heating_id": [5, 6, 7],
                "area_id": [50, 60, 70],
                "geom_polygon": [
                    box(500, 500, 2500, 2500),
                    box(3500, 3500, 5000, 5000),
                    box(10500, 6500, 11500, 7500),
                ],
            },
            geometry="geom_polygon",
            crs=3857,
        ),
        "chp_plants": gpd.GeoDataFrame(
            {
                "district_heating_id": [5, 6, 99],
                "geom": [Point(1500, 1200), Point(-1000, 1000), Point(8000, 8000)],
            },
            geometry="geom",
            crs=3857,
        ),
    }
    for name, frame in data.items():
        if isinstance(frame, gpd.GeoDataFrame):
            data[name] = frame.to_crs(source_crs)

    result = heat_pump_import._prepare_central_heat_sources(grid, data)

    np.testing.assert_allclose(result.p_set, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(result.weather_cell_id, [11, 11, 22, np.nan])
    # The second bus is 100 m outside its nearest area; the fourth has none nearby.
    np.testing.assert_allclose(result.district_heating_id, [5, 6, 7, np.nan])
    np.testing.assert_allclose(result.area_id, [50, 60, 70, np.nan])
    # Valid CHP; reject out-of-grid CHP and use bus; station; unmatched inside bus.
    expected = [
        Point(1500, 1200),
        Point(5100, 4000),
        Point(100, 100),
        Point(7000, 7000),
    ]
    # The nearest-area join runs in EPSG:3035 and reprojects back, which shifts the
    # locations by a few micrometres.
    for actual, location in zip(result.geom, expected):
        assert actual.distance(location) < 1e-3
    assert result.crs.to_epsg() == 3857
    assert "centroid" not in data["district_heating_areas"]


def test_prepare_central_heat_sources_empty_offline():
    plants = pd.DataFrame(columns=["bus0", "bus1", "p_set"])
    result = heat_pump_import._prepare_central_heat_sources(None, {"plants": plants})
    pd.testing.assert_frame_equal(result, plants)


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

    @pytest.mark.parametrize(
        "import_types, expected_counts",
        [
            (None, (3, 3, 2)),
            ([], (0, 0, 0)),
            (["individual_heat_pumps"], (3, 0, 0)),
            (["central_heat_pumps"], (0, 3, 0)),
            (["central_resistive_heaters"], (0, 0, 2)),
            (["individual_heat_pumps", "central_heat_pumps"], (3, 3, 0)),
            (["individual_heat_pumps", "central_resistive_heaters"], (3, 0, 2)),
            (["central_heat_pumps", "central_resistive_heaters"], (0, 3, 2)),
            (
                [
                    "individual_heat_pumps",
                    "central_heat_pumps",
                    "central_resistive_heaters",
                ],
                (3, 3, 2),
            ),
        ],
        ids=[
            "default",
            "none",
            "individual",
            "central",
            "resistive",
            "individual-central",
            "individual-resistive",
            "central-resistive",
            "all",
        ],
    )
    def test_oedb_offline(
        self, caplog, mock_oep_heat_pump_query, import_types, expected_counts
    ):
        hp_individual = self.setup_heat_pump_data_individual_heating()
        hp_central = self.setup_heat_pump_data_dh()
        resistive_heaters_central = self.setup_resistive_heater_data_dh()
        resistive_heaters_central.at[1, "p_set"] = 0.1
        datasets = (hp_individual, hp_central, resistive_heaters_central)
        crs = self.edisgo.topology.grid_district["srid"]
        query_results = (
            hp_individual.copy(deep=True),
            _central_query_data(hp_central, crs),
            _central_query_data(resistive_heaters_central, crs),
        )
        sectors = (
            "individual_heating",
            "district_heating",
            "district_heating_resistive_heater",
        )
        query_names = (
            "_query_individual_heat_pumps_oedb",
            "_query_central_heat_pumps_oedb",
            "_query_central_resistive_heaters_oedb",
        )
        # Each mock returns fixed data. The real importer chooses which to call.
        query_mocks = [
            mock_oep_heat_pump_query(name, data)
            for name, data in zip(query_names, query_results)
        ]
        capacity_mock = mock_oep_heat_pump_query(
            "_query_individual_heat_pump_capacity_oedb", hp_individual.p_set.sum()
        )

        with caplog.at_level(logging.DEBUG):
            imported = heat_pump_import.oedb(
                self.edisgo, scenario="eGon2035", engine=None, import_types=import_types
            )

        loads_df = self.edisgo.topology.loads_df
        hp_df = loads_df[loads_df.type == "heat_pump"]
        assert len(hp_df) == sum(expected_counts)
        assert set(imported) == set(hp_df.index)
        assert "Capacity of individual heat pumps" not in caplog.text
        for sector, count, data, query_mock in zip(
            sectors, expected_counts, datasets, query_mocks
        ):
            selected = hp_df[hp_df.sector == sector]
            assert len(selected) == count
            if count:
                assert np.isclose(selected.p_set.sum(), data.p_set.sum())
                query_mock.assert_called_once_with(
                    edisgo_object=self.edisgo, scenario="eGon2035", engine=None
                )
            else:
                query_mock.assert_not_called()
        if expected_counts[0]:
            capacity_mock.assert_called_once_with(
                edisgo_object=self.edisgo, scenario="eGon2035", engine=None
            )
        else:
            capacity_mock.assert_not_called()

        if expected_counts[1] and expected_counts[2]:
            dh_hp = hp_df[hp_df.sector == "district_heating"]
            dh_rh = hp_df[hp_df.sector == "district_heating_resistive_heater"]
            assert dh_rh[dh_rh.district_heating_id == 5].bus.iat[0] in (
                dh_hp[dh_hp.district_heating_id == 5].bus.values
            )
        if expected_counts == (0, 3, 0):
            small_hp = hp_df[hp_df.p_set == 0.05]
            assert determine_bus_voltage_level(self.edisgo, small_hp.bus.iat[0]) == 7

    @pytest.mark.oep
    def test_query_heat_pump_data_oedb_live(self, oep_engine):
        hp_individual = heat_pump_import._query_individual_heat_pumps_oedb(
            edisgo_object=self.edisgo,
            scenario="eGon2035",
            engine=oep_engine,
            query_limit=2,
        )
        hp_central = heat_pump_import._query_central_heat_pumps_oedb(
            edisgo_object=self.edisgo,
            scenario="eGon2035",
            engine=oep_engine,
            query_limit=2,
        )
        resistive_heaters_central = (
            heat_pump_import._query_central_resistive_heaters_oedb(
                edisgo_object=self.edisgo,
                scenario="eGon2035",
                engine=oep_engine,
                query_limit=2,
            )
        )
        hp_individual_cap = heat_pump_import._query_individual_heat_pump_capacity_oedb(
            edisgo_object=self.edisgo, scenario="eGon2035", engine=oep_engine
        )

        assert 0 < len(hp_individual) <= 2
        assert {"building_id", "p_set", "weather_cell_id"}.issubset(
            hp_individual.columns
        )
        for central_data in [hp_central, resistive_heaters_central]:
            assert len(central_data["plants"]) <= 2
            assert {"bus0", "bus1", "p_set"}.issubset(central_data["plants"].columns)
            if not central_data["plants"].empty:
                for name, columns in {
                    "heat_buses": {"bus1", "geom"},
                    "weather_cells": {"weather_cell_id", "geom"},
                    "district_heating_areas": {
                        "district_heating_id",
                        "area_id",
                        "geom_polygon",
                    },
                    "chp_plants": {"district_heating_id", "geom"},
                }.items():
                    assert columns.issubset(central_data[name].columns)
                    assert central_data[name].crs is not None
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
