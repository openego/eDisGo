import logging

import numpy as np
import pandas as pd
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.io import generators_import
from edisgo.tools.tools import determine_bus_voltage_level


class TestGeneratorsImport:
    """
    Tests all functions in generators_import.py except where test grid
    can be used. oedb function is tested separately as a real ding0 grid
    needs to be used.

    """

    @pytest.yield_fixture(autouse=True)
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
    """
    Tests in here are marked as slow, as the used test grid is quite large
    and should at some point be changed.

    """

    @pytest.mark.slow
    def test_oedb_legacy_without_timeseries(self):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            generator_scenario="nep2035",
        )
        edisgo.set_time_series_worst_case_analysis()

        # check number of generators
        assert len(edisgo.topology.generators_df) == 524
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(), 20.18783)

    @pytest.mark.slow
    def test_oedb_legacy_with_worst_case_timeseries(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        edisgo.set_time_series_worst_case_analysis()

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035")
        edisgo.set_time_series_worst_case_analysis()

        # check number of generators
        assert len(edisgo.topology.generators_df) == 524
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(), 20.18783)

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

    @pytest.mark.slow
    def test_oedb_legacy_with_timeseries_by_technology(self):
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
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=ts_gen_fluctuating,
            dispatchable_generators_ts=ts_gen_dispatchable,
            conventional_loads_ts="demandlib",
        )
        edisgo.set_time_series_reactive_power_control()

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035")
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=ts_gen_fluctuating,
            dispatchable_generators_ts=ts_gen_dispatchable,
            conventional_loads_ts="demandlib",
        )
        edisgo.set_time_series_reactive_power_control()

        # check number of generators
        assert len(edisgo.topology.generators_df) == 524
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(), 20.18783)

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

    @pytest.mark.slow
    def test_target_capacity(self):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case",
        )

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

    @pytest.mark.local
    def test_oedb(self):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        edisgo.import_generators(generator_scenario="eGon2035", engine=pytest.engine)
        assert len(edisgo.topology.generators_df) == 677
