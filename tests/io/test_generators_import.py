import logging

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.io import generators_import
from edisgo.tools.tools import determine_bus_voltage_level


@pytest.fixture
def mock_oep_generator_query(monkeypatch):
    """Replace the combined OEP generator query with synthetic return data."""

    def install_mock(pv_rooftop_df, power_plants_gdf, chp_gdf):
        query_mock = Mock(
            side_effect=lambda *args, **kwargs: (
                pv_rooftop_df.copy(deep=True),
                power_plants_gdf.copy(deep=True),
                chp_gdf.copy(deep=True),
            )
        )
        monkeypatch.setattr(
            generators_import,
            "_query_generator_data_oedb",
            query_mock,
        )
        return query_mock

    return install_mock


@pytest.fixture
def mock_oep_legacy_generator_query(monkeypatch):
    """Replace the legacy OEP generator query with synthetic return data."""

    def install_mock(conv_mv, renewable_mv, renewable_lv):
        query_mock = Mock(
            side_effect=lambda *args, **kwargs: (
                conv_mv.copy(deep=True),
                renewable_mv.copy(deep=True),
                renewable_lv.copy(deep=True),
            )
        )
        monkeypatch.setattr(
            generators_import,
            "_query_generator_data_oedb_legacy",
            query_mock,
        )
        return query_mock

    return install_mock


def _setup_minimal_legacy_generator_data(
    edisgo,
    existing_types=("solar",),
    new_types=("solar",),
):
    """Build LV/MV records: renewable power in kW, conventional power in MW."""
    generators_df = edisgo.topology.generators_df
    lv_generator_names = [
        name
        for name, generator in generators_df.iterrows()
        if determine_bus_voltage_level(edisgo, generator.bus) == 7
    ][: len(existing_types)]
    generators_df = generators_df.loc[lv_generator_names].copy()

    subtype_by_type = {
        "solar": "solar_roof_mounted",
        "wind": "wind_onshore",
        "biomass": "biomass",
        "run_of_river": "run_of_river",
    }
    generators_df.loc[:, "type"] = list(existing_types)
    generators_df.loc[:, "subtype"] = [
        subtype_by_type[generator_type] for generator_type in existing_types
    ]
    edisgo.topology.generators_df = generators_df

    records = []
    for name, generator in generators_df.iterrows():
        generator_id = int(name.rsplit("_", 1)[1])
        lv_grid_id = int(name.split("_lvgd_", 1)[1].split("_", 1)[0])
        bus = generator.bus
        geom = Point(
            (
                edisgo.topology.buses_df.at[bus, "x"],
                edisgo.topology.buses_df.at[bus, "y"],
            )
        ).wkt
        records.append(
            {
                "id": generator_id,
                "generator_id": generator_id,
                "subst_id": edisgo.topology.mv_grid.id,
                "la_id": None,
                "mvlv_subst_id": lv_grid_id,
                "p_nom": generator.p_nom * 1e3,
                "generator_type": generator.type,
                "subtype": generator.subtype,
                "voltage_level": 7,
                "weather_cell_id": generator.weather_cell_id,
                "geom": geom,
                "geom_em": geom,
            }
        )

    template = records[0]
    for offset, generator_type in enumerate(new_types, start=1):
        record = template.copy()
        record.update(
            {
                "id": 2_000_000 + offset,
                "generator_id": 2_000_000 + offset,
                "p_nom": 5.0,
                "generator_type": generator_type,
                "subtype": subtype_by_type[generator_type],
            }
        )
        records.append(record)

    renewable_lv = pd.DataFrame.from_records(records).set_index("id")
    # New MV plants are query results, not yet integrated into the test grid.
    geom_mv = edisgo.topology.grid_district["geom"].representative_point().wkt
    mv_record = template.copy()
    mv_record.update(
        id=3_000_001,
        generator_id=3_000_001,
        mvlv_subst_id=None,
        p_nom=1500.0,
        generator_type="wind",
        subtype="wind_onshore",
        voltage_level=5,
        geom=geom_mv,
        geom_em=geom_mv,
    )
    renewable_mv = pd.DataFrame.from_records([mv_record]).set_index("id")
    conventional_mv = pd.DataFrame(
        {
            "generator_id": [4_000_001],
            "subst_id": [edisgo.topology.mv_grid.id],
            "la_id": [None],
            "p_nom": [0.7],
            "voltage_level": [5],
            "generator_type": ["gas"],
            "geom": [geom_mv],
        },
        index=pd.Index([4_000_001], name="id"),
    )
    return conventional_mv, renewable_mv, renewable_lv


class TestGeneratorsImport:
    """
    Tests all functions in generators_import.py except where test grid
    can be used. oedb function is tested separately as a real ding0 grid
    needs to be used.

    """

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

    def test_update_grids(self):
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom_gen_new = Point((x, y))
        generators_mv = pd.DataFrame(
            data={
                "generator_id": [2, 3, 345],
                "geom": [None, None, str(geom_gen_new)],
                "p_nom": [3.0, 2.67, 2.5],
                "generator_type": ["wind", "solar", "solar"],
                "subtype": ["wind", "solar", "solar"],
                "weather_cell_id": [1122074, 1122075, 1122074],
                "voltage_level": [4, 4, 4],
            },
            index=[2, 3, 345],
        )
        generators_lv = pd.DataFrame(
            data={
                "generator_id": [13, 14, 456],
                "geom": [None, None, str(geom_gen_new)],
                "p_nom": [0.027, 0.005, 0.3],
                "generator_type": ["solar", "solar", "solar"],
                "subtype": ["solar", "solar", "roof"],
                "weather_cell_id": [1122075, 1122075, 1122074],
                "voltage_level": [6, 6, 6],
                "mvlv_subst_id": [None, None, 6.0],
            },
            index=[13, 14, 456],
        )

        generators_import._update_grids(self.edisgo, generators_mv, generators_lv)

        # check number of generators
        assert len(self.edisgo.topology.generators_df) == 6
        assert len(self.edisgo.topology.mv_grid.generators_df) == 3

        # check removed generators
        assert "Generator_1" not in self.edisgo.topology.generators_df.index
        assert "GeneratorFluctuating_12" not in self.edisgo.topology.generators_df.index

        # check updated generators
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_2", "p_nom"]
            == 3
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_2", "subtype"]
            == "wind_wind_onshore"
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_13", "p_nom"]
            == 0.027
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_13", "subtype"]
            == "solar_solar_roof_mounted"
        )

        # check generators that stayed the same
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_3", "p_nom"]
            == 2.67
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_3", "subtype"]
            == "solar_solar_ground_mounted"
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_14", "p_nom"]
            == 0.005
        )
        assert (
            self.edisgo.topology.generators_df.at["GeneratorFluctuating_14", "subtype"]
            == "solar_solar_roof_mounted"
        )

        # check new generators
        assert (
            self.edisgo.topology.generators_df.at[
                "Generator_MVGrid_1_solar_345", "p_nom"
            ]
            == 2.5
        )
        assert (
            self.edisgo.topology.generators_df.at[
                "Generator_MVGrid_1_solar_345", "type"
            ]
            == "solar"
        )
        assert (
            self.edisgo.topology.generators_df.at[
                "Generator_LVGrid_9_solar_456", "p_nom"
            ]
            == 0.3
        )
        assert (
            self.edisgo.topology.generators_df.at[
                "Generator_LVGrid_9_solar_456", "type"
            ]
            == "solar"
        )

    def test_update_grids_target_capacity(self):
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom_gen_new = Point((x, y))
        generators_mv = pd.DataFrame(
            data={
                "generator_id": [321, 3456, 345],
                "geom": [
                    str(geom_gen_new),
                    str(geom_gen_new),
                    str(geom_gen_new),
                ],
                "p_nom": [3.0, 2.67, 2.5],
                "generator_type": ["wind", "solar", "solar"],
                "subtype": ["wind", "solar", "solar"],
                "weather_cell_id": [1122074, 1122075, 1122074],
                "voltage_level": [4, 4, 4],
            },
            index=[321, 3456, 345],
        )
        generators_lv = pd.DataFrame(
            data={
                "generator_id": [13, 145, 456, 654],
                "geom": [None, None, str(geom_gen_new), None],
                "p_nom": [0.027, 0.005, 0.3, 0.3],
                "generator_type": ["solar", "solar", "run_of_river", "wind"],
                "subtype": ["solar", "solar", "hydro", "wind"],
                "weather_cell_id": [1122075, 1122075, 1122074, 1122074],
                "voltage_level": [6, 6, 6, 7],
                "mvlv_subst_id": [None, None, 6.0, 2],
            },
            index=[13, 145, 456, 654],
        )

        gens_before = self.edisgo.topology.generators_df
        p_wind_before = gens_before[gens_before["type"] == "wind"].p_nom.sum()
        p_pv_before = gens_before[gens_before["type"] == "solar"].p_nom.sum()
        p_gas_before = gens_before[gens_before["type"] == "gas"].p_nom.sum()
        p_target = {
            "wind": p_wind_before + 6,
            "solar": p_pv_before + 3,
            "gas": p_gas_before + 1.5,
        }

        generators_import._update_grids(
            self.edisgo,
            generators_mv,
            generators_lv,
            p_target=p_target,
            remove_decommissioned=False,
            update_existing=False,
        )

        # check that all old generators still exist
        assert gens_before.index.isin(self.edisgo.topology.generators_df.index).all()

        # check that types for which no target capacity is specified are
        # not expanded
        assert "run_of_river" not in self.edisgo.topology.generators_df["type"].unique()

        # check that target capacity for specified types is met
        # wind - target capacity higher than existing capacity plus new
        # capacity (all new generators are integrated and capacity is scaled
        # up)
        assert (
            self.edisgo.topology.generators_df[
                self.edisgo.topology.generators_df["type"] == "wind"
            ].p_nom.sum()
            == p_wind_before + 6
        )
        assert (
            len(
                self.edisgo.topology.generators_df[
                    self.edisgo.topology.generators_df["type"] == "wind"
                ]
            )
            == len(gens_before[gens_before["type"] == "wind"]) + 2
        )
        assert (
            self.edisgo.topology.generators_df.at[
                "Generator_MVGrid_1_wind_321", "p_nom"
            ]
            >= 3.0
        )

        # solar - target capacity lower than existing capacity plus new
        # capacity (not all new generators are integrated)
        assert np.isclose(
            self.edisgo.topology.generators_df[
                self.edisgo.topology.generators_df["type"] == "solar"
            ].p_nom.sum(),
            p_pv_before + 3,
        )
        assert (
            len(
                self.edisgo.topology.generators_df[
                    self.edisgo.topology.generators_df["type"] == "solar"
                ]
            )
            <= len(gens_before[gens_before["type"] == "solar"]) + 4
        )

        # gas - no new generator, existing one is scaled
        assert (
            self.edisgo.topology.generators_df[
                self.edisgo.topology.generators_df["type"] == "gas"
            ].p_nom.sum()
            == p_gas_before + 1.5
        )
        assert len(
            self.edisgo.topology.generators_df[
                self.edisgo.topology.generators_df["type"] == "gas"
            ]
        ) == len(gens_before[gens_before["type"] == "gas"])
        assert (
            self.edisgo.topology.generators_df.at["Generator_1", "p_nom"] == 0.775 + 1.5
        )

    def test__integrate_pv_rooftop(self, caplog):
        # set up dataframe with:
        # * one gen where capacity will increase and voltage level
        #   changes ("SEE980819686674")
        # * one where capacity will decrease ("SEE970362202254")
        # * one where capacity stayed the same ("SEE960032475262")
        # * one with source ID that does not exist in future scenario ("SEE2")
        pv_df = pd.DataFrame(
            data={
                "p_nom": [0.005, 0.15, 0.068, 2.0],
                "weather_cell_id": [11051, 11051, 11052, 11052],
                "building_id": [430903, 445710, 431094, 446933],
                "generator_id": [1, 2, 3, 4],
                "type": ["solar", "solar", "solar", "solar"],
                "subtype": ["pv_rooftop", "pv_rooftop", "pv_rooftop", "pv_rooftop"],
                "source_id": [
                    "SEE970362202254",
                    "SEE980819686674",
                    "SEE960032475262",
                    "SEE2",
                ],
            },
            index=[1, 2, 3, 4],
        )

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

        gens_before = edisgo.topology.generators_df.copy()
        with caplog.at_level(logging.DEBUG):
            generators_import._integrate_pv_rooftop(edisgo, pv_df)

        gens_df = edisgo.topology.generators_df[
            edisgo.topology.generators_df.subtype == "pv_rooftop"
        ].copy()

        assert len(gens_df) == 4
        # check gen where capacity increases and voltage level changes
        gen_name = gens_df[gens_df.source_id == "SEE980819686674"].index[0]
        assert gen_name not in gens_before.index
        bus_gen = gens_df.at[gen_name, "bus"]
        assert determine_bus_voltage_level(edisgo, bus_gen) == 6
        # check gen where capacity decreases
        gen_name = gens_df[gens_df.source_id == "SEE970362202254"].index[0]
        assert gen_name in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 0.005
        # check gen where capacity stayed the same
        gen_name = gens_df[gens_df.source_id == "SEE960032475262"].index[0]
        assert gen_name in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 0.068
        # check new gen
        gen_name = gens_df[gens_df.source_id == "SEE2"].index[0]
        assert gen_name not in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 2.0
        # check logging
        assert (
            "2.22 MW of PV rooftop plants integrated. Of this, 0.22 MW could be "
            "matched to an existing PV rooftop plant." in caplog.text
        )

    def test__integrate_pv_rooftop_two_buildings_on_one_bus(self):
        """
        A bus carrying conventional loads of two different buildings must not
        break the PV rooftop import.

        This used to raise "cannot reindex on an axis with duplicate labels"
        (openego/eGo#213, seen on MV grids 33084 and 33695): the function built
        a bus -> building ID map from the conventional loads, and that map was
        deduplicated on "building_id" while being indexed by "bus". Two
        buildings on one bus therefore left a duplicated bus label, the join
        multiplied the generator rows and ``DataFrame.update`` rejected the
        non-unique index.

        That map has since been removed entirely -- it fed a building-ID based
        matching that was replaced by source-ID matching in d33e8515 -- so the
        function no longer reads ``loads_df`` at all and the crash is
        structurally impossible. This test is kept as a guard against
        reintroducing a per-bus load lookup here.

        Note that the shipped test grid cannot produce the situation on its
        own: it has 70 buses with more than one conventional load, but all of
        them share a single building ID. The second building is therefore
        added explicitly.
        """
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        loads_df = edisgo.topology.loads_df
        gens_df = edisgo.topology.generators_df
        # a bus that carries both a PV rooftop generator and a conventional load
        pv_buses = set(gens_df[gens_df.subtype == "pv_rooftop"].bus)
        conv = loads_df[loads_df.type == "conventional_load"]
        bus = conv[conv.bus.isin(pv_buses)].bus.iloc[0]
        existing = conv[conv.bus == bus].iloc[0]

        # second building on the same bus
        second = existing.copy()
        second["building_id"] = int(existing.building_id) + 1_000_000
        edisgo.topology.loads_df.loc["Load_second_building_same_bus"] = second

        pv_df = pd.DataFrame(
            data={
                "p_nom": [0.005],
                "weather_cell_id": [11051],
                "building_id": [430903],
                "generator_id": [1],
                "type": ["solar"],
                "subtype": ["pv_rooftop"],
                "source_id": ["SEE970362202254"],
            },
            index=[1],
        )

        # used to raise ValueError: cannot reindex on an axis with duplicate labels
        generators_import._integrate_pv_rooftop(edisgo, pv_df)

        # no generator was duplicated by the join (generators whose source_id
        # is absent from the scenario are legitimately decommissioned here)
        assert not edisgo.topology.generators_df.index.has_duplicates

    def test__integrate_new_pv_rooftop_to_buildings(self, caplog):
        pv_df = pd.DataFrame(
            data={
                "p_nom": [0.005, 0.15, 2.0],
                "weather_cell_id": [11051, 11051, 11052],
                "building_id": [430903, 445710, 446933],
                "generator_id": [1, 2, 3],
                "type": ["solar", "solar", "solar"],
                "subtype": ["pv_rooftop", "pv_rooftop", "pv_rooftop"],
                "source_id": [None, None, None],
            },
            index=[1, 2, 3],
        )

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        # manipulate grid so that building 445710 is connected to MV/LV station
        load = edisgo.topology.loads_df[
            edisgo.topology.loads_df.building_id == 445710
        ].index[0]
        busbar_bus = "BusBar_mvgd_33535_lvgd_1164120011_LV"
        edisgo.topology.loads_df.at[load, "bus"] = busbar_bus
        num_gens_before = len(edisgo.topology.generators_df)
        with caplog.at_level(logging.DEBUG):
            (
                integrated_pv,
                integrated_pv_own_grid_conn,
            ) = generators_import._integrate_new_pv_rooftop_to_buildings(edisgo, pv_df)

        assert num_gens_before + 3 == len(edisgo.topology.generators_df)
        gens_df = edisgo.topology.generators_df.loc[integrated_pv, :]
        assert len(gens_df) == 3
        # check that smallest PV plant is connected to LV
        bus_gen_voltage_level_7 = gens_df[gens_df.p_nom == 0.005].bus[0]
        assert edisgo.topology.buses_df.at[bus_gen_voltage_level_7, "v_nom"] == 0.4
        # check that medium PV plant is connected same bus as building
        bus_gen_voltage_level_6 = gens_df[gens_df.p_nom == 0.15].bus[0]
        assert bus_gen_voltage_level_6 == busbar_bus
        # check that largest heat pump is connected to MV
        bus_gen_voltage_level_5 = gens_df[gens_df.p_nom == 2.0].bus[0]
        assert edisgo.topology.buses_df.at[bus_gen_voltage_level_5, "v_nom"] == 20.0

        assert edisgo.topology.generators_df.loc[integrated_pv, "p_nom"].sum() == 2.155
        assert (
            edisgo.topology.generators_df.loc[
                integrated_pv_own_grid_conn, "p_nom"
            ].sum()
            == 2.0
        )

    def test__integrate_power_and_chp_plants(self, caplog):
        # set up test data
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        # set up dataframe with:
        # * one gen where capacity will increase but voltage level stays the same
        #   ("SEE95")
        # * one gen where capacity will increase and voltage level changes ("SEE96")
        # * one where capacity will decrease ("SEE97")
        # * one where capacity stayed the same ("SEE98")
        # * one with source ID that does not exist in future scenario ("SEE99")
        random_bus = "BusBar_mvgd_33535_lvgd_1156570000_MV"
        random_lv_bus = "BranchTee_mvgd_33535_lvgd_1150630000_35"
        x = edisgo.topology.buses_df.at[random_bus, "x"]
        y = edisgo.topology.buses_df.at[random_bus, "y"]
        geom = Point((x, y))
        gen_df = pd.DataFrame(
            data={
                "bus": [random_bus, random_lv_bus, random_bus, random_bus, random_bus],
                "p_nom": [1.9, 0.15, 2.0, 3.0, 1.3],
                "type": ["biomass", "biomass", "biomass", "biomass", "biomass"],
                "source_id": ["SEE95", "SEE96", "SEE97", "SEE98", "SEE99"],
            },
            index=[
                "dummy_gen_1",
                "dummy_gen_2",
                "dummy_gen_3",
                "dummy_gen_4",
                "dummy_gen_5",
            ],
        )
        edisgo.topology.generators_df = pd.concat(
            [edisgo.topology.generators_df, gen_df]
        )
        # set up dataframes with future generators:
        # * one without source ID
        # * one with source ID that does not exist in future scenario ("SEE94")
        # * one gen where capacity increases but voltage level stays the same ("SEE95")
        # * one gen where capacity increases and voltage level changes ("SEE96")
        # * one where capacity decreases ("SEE97")
        # * one where capacity stayed the same ("SEE98")
        new_pp_gdf = pd.DataFrame(
            data={
                "generator_id": [2853, 2854, 2855, 2856, 2857, 2858],
                "source_id": [None, "SEE94", "SEE95", "SEE96", "SEE97", "SEE98"],
                "type": [
                    "biomass",
                    "biomass",
                    "biomass",
                    "biomass",
                    "biomass",
                    "biomass",
                ],
                "subtype": [None, None, None, None, None, None],
                "p_nom": [0.005, 0.15, 2.0, 1.0, 0.05, 3.0],
                "weather_cell_id": [None, None, None, None, None, None],
                "geom": [geom, geom, None, geom, None, None],
            },
            index=[0, 1, 2, 3, 4, 5],
        )
        new_chp_gdf = pd.DataFrame(
            data={
                "generator_id": [9363],
                "type": ["biomass"],
                "district_heating_id": [None],
                "p_nom": [0.66],
                "p_nom_th": [4.66],
                "geom": [geom],
            },
            index=[0],
        )

        gens_before = edisgo.topology.generators_df.copy()
        with caplog.at_level(logging.DEBUG):
            generators_import._integrate_power_and_chp_plants(
                edisgo, new_pp_gdf, new_chp_gdf
            )

        gens_df = edisgo.topology.generators_df[
            edisgo.topology.generators_df.subtype != "pv_rooftop"
        ].copy()

        # check new gen without source id
        gen_name = gens_df[gens_df.p_nom == 0.005].index[0]
        assert gen_name not in gens_before.index
        bus_gen = gens_df.at[gen_name, "bus"]
        assert edisgo.topology.buses_df.at[bus_gen, "v_nom"] == 0.4
        # check gen with source ID that does not exist in future scenario ("SEE94")
        gen_name = gens_df[gens_df.source_id == "SEE94"].index[0]
        assert gen_name not in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 0.15
        # check gen where capacity increases but voltage level stays the same ("SEE95")
        gen_name = gens_df[gens_df.source_id == "SEE95"].index[0]
        assert gen_name in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 2.0
        # check gen where capacity increases and voltage level changes ("SEE96")
        gen_name = gens_df[gens_df.source_id == "SEE96"].index[0]
        assert gen_name not in gens_before.index
        bus_gen = gens_df.at[gen_name, "bus"]
        assert edisgo.topology.buses_df.at[bus_gen, "v_nom"] == 20.0
        # check gen where capacity decreases ("SEE97")
        gen_name = gens_df[gens_df.source_id == "SEE97"].index[0]
        assert gen_name in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 0.05
        # check gen where capacity stayed the same ("SEE98")
        gen_name = gens_df[gens_df.source_id == "SEE98"].index[0]
        assert gen_name in gens_before.index
        assert gens_df.at[gen_name, "bus"] == random_bus
        # check CHP
        gen_name = gens_df[gens_df.p_nom_th == 4.66].index[0]
        assert gen_name not in gens_before.index
        assert gens_df.at[gen_name, "p_nom"] == 0.66
        # check logging
        assert (
            "6.87 MW of power and CHP plants integrated. Of this, 6.05 MW could be "
            "matched to existing power plants." in caplog.text
        )


class TestGeneratorsImportOEDB:
    """Test generator imports that require realistic legacy or eGon grids."""

    @pytest.mark.parametrize(
        "supplied_engine", [None, object()], ids=["no-engine", "supplied-engine"]
    )
    def test_oedb_legacy_without_timeseries_offline(
        self, mock_oep_legacy_generator_query, supplied_engine
    ):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        generator_data = _setup_minimal_legacy_generator_data(edisgo)
        query_mock = mock_oep_legacy_generator_query(*generator_data)

        edisgo.import_generators(
            generator_scenario="nep2035",
            engine=supplied_engine,
        )
        edisgo.set_time_series_worst_case_analysis()

        # check number of generators
        assert len(edisgo.topology.generators_df) == 4
        # check total installed capacity
        expected_capacity = (
            generator_data[0].p_nom.sum()
            + (generator_data[1].p_nom.sum() + generator_data[2].p_nom.sum()) / 1e3
        )
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(), expected_capacity)
        for generator_id, technology, capacity in (
            (3_000_001, "wind", 1.5),
            (4_000_001, "gas", 0.7),
        ):
            generators = edisgo.topology.generators_df
            selected = generators.loc[generators.index.str.endswith(f"_{generator_id}")]
            assert len(selected) == 1
            generator = selected.iloc[0]
            assert generator.type == technology
            assert np.isclose(generator.p_nom, capacity)
            assert determine_bus_voltage_level(edisgo, generator.bus) == 5
        # A caller's engine must not override the legacy database connection.
        query_mock.assert_called_once_with(
            edisgo_object=edisgo,
            generator_scenario="nep2035",
        )

    def test_oedb_legacy_normalizes_raw_data_offline(
        self, mock_oep_legacy_generator_query
    ):
        """Convert renewable kW to MW and fill missing subtypes before integration."""
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        conventional_mv, renewable_mv, renewable_lv = (
            _setup_minimal_legacy_generator_data(edisgo)
        )
        renewable_lv["p_nom"] = renewable_lv.p_nom.astype(str)
        renewable_lv.at[2_000_001, "subtype"] = None

        renewable_mv["p_nom"] = "1500"
        renewable_mv["subtype"] = np.nan
        mock_oep_legacy_generator_query(conventional_mv, renewable_mv, renewable_lv)

        edisgo.import_generators(generator_scenario="nep2035")

        generators = edisgo.topology.generators_df
        assert len(generators) == 4
        for generator_id, expected_power in (
            (2_000_001, 0.005),
            (3_000_001, 1.5),
            (4_000_001, 0.7),
        ):
            generator = generators.loc[
                generators.index.str.endswith(f"_{generator_id}")
            ].squeeze(axis=0)
            assert np.isclose(generator.p_nom, expected_power)
            if generator_id != 4_000_001:
                assert generator.subtype == "unknown"
        existing = generators.loc[
            ~generators.index.str.endswith(("_2000001", "_3000001", "_4000001"))
        ]
        assert existing.subtype.tolist() == ["solar_roof_mounted"]

    def test_oedb_legacy_with_worst_case_timeseries_offline(
        self, mock_oep_legacy_generator_query
    ):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        generator_data = _setup_minimal_legacy_generator_data(edisgo)
        query_mock = mock_oep_legacy_generator_query(*generator_data)
        edisgo.set_time_series_worst_case_analysis()

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035", engine=None)
        edisgo.set_time_series_worst_case_analysis()

        for technology, capacity, scaling_type in (
            ("wind", 1.5, "wind"),
            ("gas", 0.7, "other"),
        ):
            selected = edisgo.topology.generators_df.query("type == @technology")
            assert len(selected) == 1
            name = selected.index[0]
            for case, timestamp in edisgo.timeseries.timeindex_worst_cases.items():
                scenario = case.rsplit("_", 1)[0]
                expected_power = (
                    capacity
                    * edisgo.config["worst_case_scale_factor"][
                        f"{scenario}_feed-in_{scaling_type}"
                    ]
                )
                assert np.isclose(
                    edisgo.timeseries.generators_active_power.at[timestamp, name],
                    expected_power,
                )
            np.testing.assert_allclose(
                edisgo.timeseries.generators_reactive_power[name],
                -edisgo.timeseries.generators_active_power[name]
                * np.tan(
                    np.arccos(edisgo.config["reactive_power_factor"]["mv_generator"])
                ),
            )

        # check number of generators
        assert len(edisgo.topology.generators_df) == 4
        # check total installed capacity
        assert np.isclose(
            edisgo.topology.generators_df.p_nom.sum(),
            generator_data[0].p_nom.sum()
            + (generator_data[1].p_nom.sum() + generator_data[2].p_nom.sum()) / 1e3,
        )

        gens_new = edisgo.topology.generators_df[
            ~edisgo.topology.generators_df.index.isin(gens_before.index)
        ]
        # check solar generator (same weather cell ID and in same voltage
        # level, wherefore p_nom is set to be below 300 kW)
        old_solar_gen = gens_before[
            (gens_before.type == "solar") & (gens_before.p_nom <= 0.3)
        ].iloc[0, :]
        new_solar_gen = gens_new[
            (gens_new.type == "solar")
            & (gens_new.weather_cell_id == old_solar_gen.weather_cell_id)
            & (gens_new.p_nom <= 0.3)
        ].iloc[0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(
            gens_ts_active_before.loc[:, old_solar_gen.name].tolist(),
            edisgo.timeseries.generators_active_power.loc[
                :, old_solar_gen.name
            ].tolist(),
        ).all()
        assert np.isclose(
            gens_ts_reactive_before.loc[:, old_solar_gen.name].tolist(),
            edisgo.timeseries.generators_reactive_power.loc[
                :, old_solar_gen.name
            ].tolist(),
        ).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:, old_solar_gen.name].tolist()
            / old_solar_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[
                :, new_solar_gen.name
            ].tolist()
            / new_solar_gen.p_nom,
        ).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[
                :, new_solar_gen.name
            ].tolist(),
            (
                edisgo.timeseries.generators_active_power.loc[:, new_solar_gen.name]
                * -np.tan(np.arccos(0.95))
            ).tolist(),
        ).all()
        # ToDo following test currently does fail sometimes as lv generators
        # connected to MV bus bar are handled as MV generators and therefore
        # assigned other cosphi
        # assert np.isclose(
        #     gens_ts_reactive_before.loc[:,
        #     old_solar_gen.name] / old_solar_gen.p_nom,
        #     edisgo.timeseries.generators_reactive_power.loc[
        #     :, new_solar_gen.name] / new_solar_gen.p_nom).all()

        query_mock.assert_called_once_with(
            edisgo_object=edisgo,
            generator_scenario="nep2035",
        )

    def test_oedb_legacy_with_timeseries_by_technology_offline(
        self, mock_oep_legacy_generator_query
    ):
        timeindex = pd.date_range("1/1/2012", periods=3, freq="H")
        ts_gen_dispatchable = pd.DataFrame(
            {"other": [0.775] * 3, "gas": [0.9] * 3}, index=timeindex
        )
        ts_gen_fluctuating = pd.DataFrame(
            {"wind": [0.1, 0.2, 0.15], "solar": [0.4, 0.5, 0.45]},
            index=timeindex,
        )

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path, timeindex=timeindex
        )
        generator_data = _setup_minimal_legacy_generator_data(edisgo)
        query_mock = mock_oep_legacy_generator_query(*generator_data)
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=ts_gen_fluctuating,
            dispatchable_generators_ts=ts_gen_dispatchable,
            conventional_loads_ts="demandlib",
        )
        edisgo.set_time_series_reactive_power_control()

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035", engine=None)
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=ts_gen_fluctuating,
            dispatchable_generators_ts=ts_gen_dispatchable,
            conventional_loads_ts="demandlib",
        )
        edisgo.set_time_series_reactive_power_control()

        # check number of generators
        assert len(edisgo.topology.generators_df) == 4
        # check total installed capacity
        assert np.isclose(
            edisgo.topology.generators_df.p_nom.sum(),
            generator_data[0].p_nom.sum()
            + (generator_data[1].p_nom.sum() + generator_data[2].p_nom.sum()) / 1e3,
        )

        for technology, capacity, profile in (
            ("wind", 1.5, ts_gen_fluctuating["wind"]),
            ("gas", 0.7, ts_gen_dispatchable["gas"]),
        ):
            selected = edisgo.topology.generators_df.query("type == @technology")
            assert len(selected) == 1
            name = selected.index[0]
            np.testing.assert_allclose(
                edisgo.timeseries.generators_active_power[name], profile * capacity
            )
            np.testing.assert_allclose(
                edisgo.timeseries.generators_reactive_power[name],
                -profile
                * capacity
                * np.tan(
                    np.arccos(edisgo.config["reactive_power_factor"]["mv_generator"])
                ),
            )

        gens_new = edisgo.topology.generators_df[
            ~edisgo.topology.generators_df.index.isin(gens_before.index)
        ]
        # check solar generator (same voltage level, wherefore p_nom is set
        # to be below 300 kW)
        old_solar_gen = gens_before[
            (gens_before.type == "solar") & (gens_before.p_nom <= 0.3)
        ].iloc[0, :]
        new_solar_gen = gens_new[
            (gens_new.type == "solar") & (gens_new.p_nom <= 0.3)
        ].iloc[0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(
            gens_ts_active_before.loc[:, old_solar_gen.name],
            edisgo.timeseries.generators_active_power.loc[:, old_solar_gen.name],
        ).all()
        assert np.isclose(
            gens_ts_reactive_before.loc[:, old_solar_gen.name],
            edisgo.timeseries.generators_reactive_power.loc[:, old_solar_gen.name],
        ).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:, old_solar_gen.name] / old_solar_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[:, new_solar_gen.name]
            / new_solar_gen.p_nom,
        ).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[:, new_solar_gen.name],
            edisgo.timeseries.generators_active_power.loc[:, new_solar_gen.name]
            * -np.tan(np.arccos(0.95)),
        ).all()
        # ToDo following test currently does fail sometimes as lv generators
        # connected to MV bus bar are handled as MV generators and therefore
        # assigned other cosphi
        # assert np.isclose(
        #     gens_ts_reactive_before.loc[:,
        #     old_solar_gen.name] / old_solar_gen.p_nom,
        #     edisgo.timeseries.generators_reactive_power.loc[
        #     :, new_solar_gen.name] / new_solar_gen.p_nom).all()

        query_mock.assert_called_once_with(
            edisgo_object=edisgo,
            generator_scenario="nep2035",
        )

    def test_target_capacity_offline(self, mock_oep_legacy_generator_query):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        generator_data = _setup_minimal_legacy_generator_data(
            edisgo,
            existing_types=("wind", "biomass", "solar", "run_of_river"),
            new_types=("wind", "biomass", "solar"),
        )
        query_mock = mock_oep_legacy_generator_query(*generator_data)
        edisgo.set_time_series_worst_case_analysis()

        gens_before = edisgo.topology.generators_df.copy()
        p_wind_before = edisgo.topology.generators_df[
            edisgo.topology.generators_df["type"] == "wind"
        ].p_nom.sum()
        p_biomass_before = edisgo.topology.generators_df[
            edisgo.topology.generators_df["type"] == "biomass"
        ].p_nom.sum()

        p_target = {
            "wind": p_wind_before * 1.6,
            "biomass": p_biomass_before * 1.0,
        }

        edisgo.import_generators(
            generator_scenario="nep2035",
            p_target=p_target,
            remove_decommissioned=False,
            update_existing=False,
            engine=None,
        )

        # check that all old generators still exist
        assert gens_before.index.isin(edisgo.topology.generators_df.index).all()

        # check that installed capacity of types, for which no target capacity
        # was specified, remained the same
        assert np.isclose(
            gens_before[gens_before["type"] == "solar"].p_nom.sum(),
            edisgo.topology.generators_df[
                edisgo.topology.generators_df["type"] == "solar"
            ].p_nom.sum(),
        )
        assert (
            gens_before[gens_before["type"] == "run_of_river"].p_nom.sum()
            == edisgo.topology.generators_df[
                edisgo.topology.generators_df["type"] == "run_of_river"
            ].p_nom.sum()
        )

        # check that installed capacity of types, for which a target capacity
        # was specified, is met
        assert np.isclose(
            edisgo.topology.generators_df[
                edisgo.topology.generators_df["type"] == "wind"
            ].p_nom.sum(),
            p_wind_before * 1.6,
        )
        assert np.isclose(
            edisgo.topology.generators_df[
                edisgo.topology.generators_df["type"] == "biomass"
            ].p_nom.sum(),
            p_biomass_before * 1.0,
        )
        query_mock.assert_called_once_with(
            edisgo_object=edisgo,
            generator_scenario="nep2035",
        )

    @pytest.mark.oep
    def test_query_generator_data_oedb_legacy_live(self, oep_engine):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)

        conventional_mv, renewable_mv, renewable_lv = (
            generators_import._query_generator_data_oedb_legacy(
                edisgo_object=edisgo,
                generator_scenario="nep2035",
                engine=oep_engine,
                query_limit=2,
            )
        )

        assert sum(map(len, (conventional_mv, renewable_mv, renewable_lv))) > 0
        assert all(
            len(data) <= 2 for data in (conventional_mv, renewable_mv, renewable_lv)
        )
        assert {
            "generator_id",
            "p_nom",
            "voltage_level",
            "generator_type",
            "geom",
        }.issubset(conventional_mv.columns)
        renewable_columns = {
            "generator_id",
            "mvlv_subst_id",
            "p_nom",
            "generator_type",
            "subtype",
            "voltage_level",
            "weather_cell_id",
            "geom",
            "geom_em",
        }
        assert renewable_columns.issubset(renewable_mv.columns)
        assert renewable_columns.issubset(renewable_lv.columns)

    @pytest.mark.parametrize(
        "raw_type, raw_source_id, expected_type, expected_subtype, expected_source_id",
        [
            ("wind_onshore", {"mastr": "SEE_WIND"}, "wind", "wind_onshore", "SEE_WIND"),
            (
                "solar",
                {"mastr": "SEE_SOLAR"},
                "solar",
                "solar_ground_mounted",
                "SEE_SOLAR",
            ),
            ("biomass", None, "biomass", None, None),
            ("biomass", "unwrapped_id", "biomass", None, None),
            (None, None, None, None, None),
        ],
        ids=[
            "wind",
            "solar",
            "missing-source",
            "non-dict-source",
            "empty-power-plants",
        ],
    )
    def test_oedb_offline(
        self,
        mock_oep_generator_query,
        raw_type,
        raw_source_id,
        expected_type,
        expected_subtype,
        expected_source_id,
    ):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        # Keep the high-level import focused on integrating one representative record
        # from each of the three source tables. Detailed update and removal behaviour
        # is covered by the lower-level integration tests above.
        edisgo.topology.generators_df = edisgo.topology.generators_df.iloc[0:0].copy()
        building_id = 430903
        load_bus = edisgo.topology.loads_df[
            edisgo.topology.loads_df.building_id == building_id
        ].bus.iat[0]
        geom = Point(
            (
                edisgo.topology.buses_df.at[load_bus, "x"],
                edisgo.topology.buses_df.at[load_bus, "y"],
            )
        )

        pv_rooftop_df = pd.DataFrame(
            {
                "generator_id": [1],
                "building_id": [building_id],
                "source_id": ["SEE_TEST_PV"],
                "p_nom": [0.005],
                "weather_cell_id": [11051],
            }
        )
        power_plants_gdf = pd.DataFrame(
            {
                "generator_id": [2],
                "source_id": [raw_source_id],
                "type": [raw_type],
                "p_nom": [0.05],
                "weather_cell_id": [None],
                "geom": [geom],
            }
        )
        chp_gdf = pd.DataFrame(
            {
                "generator_id": [3],
                "type": ["gas"],
                "district_heating_id": [5],
                "p_nom": [0.06],
                "p_nom_th": [0.12],
                "geom": [geom],
            }
        )
        if raw_type is None:
            power_plants_gdf = power_plants_gdf.iloc[0:0].copy()
        query_mock = mock_oep_generator_query(
            pv_rooftop_df,
            power_plants_gdf,
            chp_gdf,
        )

        edisgo.import_generators(generator_scenario="eGon2035", engine=None)

        generators_df = edisgo.topology.generators_df
        has_power_plant = raw_type is not None
        assert len(generators_df) == 2 + has_power_plant
        assert np.isclose(generators_df.p_nom.sum(), 0.065 + 0.05 * has_power_plant)
        expected_types = {"solar", "gas"}
        if has_power_plant:
            expected_types.add(expected_type)
            plant = generators_df.loc[generators_df.index.str.endswith("_2")].squeeze(
                axis=0
            )
            assert plant.type == expected_type
            if expected_subtype is None:
                assert pd.isna(plant.subtype)
            else:
                assert plant.subtype == expected_subtype
            if expected_source_id is None:
                assert pd.isna(plant.source_id)
            else:
                assert plant.source_id == expected_source_id
        assert set(generators_df.type) == expected_types
        rooftop = generators_df.loc[generators_df.subtype == "pv_rooftop"]
        assert len(rooftop) == 1
        assert rooftop.type.iat[0] == "solar"
        assert rooftop.source_id.iat[0] == "SEE_TEST_PV"
        query_mock.assert_called_once_with(
            edisgo_object=edisgo,
            scenario="eGon2035",
            engine=None,
            max_capacity=20,
        )

    @pytest.mark.oep
    def test_query_generator_data_oedb_live(self, oep_engine):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

        pv_rooftop_df, power_plants_gdf, chp_gdf = (
            generators_import._query_generator_data_oedb(
                edisgo_object=edisgo,
                scenario="eGon2035",
                engine=oep_engine,
                max_capacity=20,
                query_limit=2,
            )
        )

        assert sum(map(len, (pv_rooftop_df, power_plants_gdf, chp_gdf))) > 0
        assert all(
            len(data) <= 2 for data in (pv_rooftop_df, power_plants_gdf, chp_gdf)
        )
        assert {
            "generator_id",
            "building_id",
            "source_id",
            "p_nom",
            "weather_cell_id",
        }.issubset(pv_rooftop_df.columns)
        assert {
            "generator_id",
            "source_id",
            "type",
            "p_nom",
            "weather_cell_id",
            "geom",
        }.issubset(power_plants_gdf.columns)
        assert {
            "generator_id",
            "type",
            "district_heating_id",
            "p_nom",
            "p_nom_th",
            "geom",
        }.issubset(chp_gdf.columns)
