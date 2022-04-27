import os
import shutil

import numpy as np
import pandas as pd
import pytest

from matplotlib import pyplot as plt
from shapely.geometry import Point

from edisgo import EDisGo


class TestEDisGo:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """
        Fixture to set up new EDisGo object before each test function.

        """
        # self.edisgo = EDisGo(
        #     ding0_grid=pytest.ding0_test_network_path
        # )
        self.setup_edisgo_object()

    def setup_edisgo_object(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def setup_worst_case_time_series(self):
        self.edisgo.set_time_series_worst_case_analysis()

    def test_set_time_series_manual(self):
        # bus_name = self.edisgo.add_component(
        #     comp_type="bus", bus_name="Testbus", v_nom=20
        # )
        # Todo: implement test
        pass

    def test_set_time_series_worst_case_analysis(self):
        # Todo: implement test
        pass

    def test_set_time_series_active_power_predefined(self):
        # Todo: implement test
        pass

    def test_set_time_series_reactive_power_control(self):
        # Todo: implement test
        pass

    def test_to_pypsa(self):

        self.setup_worst_case_time_series()

        # test mode None and timesteps None (default)
        pypsa_network = self.edisgo.to_pypsa()
        assert len(pypsa_network.buses) == 140
        assert len(pypsa_network.buses_t.v_mag_pu_set) == 4

        # test mode "mv" and timesteps given
        pypsa_network = self.edisgo.to_pypsa(
            mode="mv", timesteps=self.edisgo.timeseries.timeindex[0]
        )
        assert len(pypsa_network.buses) == 31
        assert len(pypsa_network.buses_t.v_mag_pu_set) == 1

        # test exception
        msg = "The entered mode is not a valid option."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.to_pypsa(mode="unknown")

    def test_to_graph(self):
        graph = self.edisgo.to_graph()
        assert len(graph.nodes) == len(self.edisgo.topology.buses_df)
        assert len(graph.edges) == (
            len(self.edisgo.topology.lines_df)
            + len(self.edisgo.topology.transformers_df.bus0.unique())
        )

    @pytest.mark.slow
    def test_generator_import(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        edisgo.import_generators("nep2035")
        assert len(edisgo.topology.generators_df) == 1636

    def test_analyze(self):

        self.setup_worst_case_time_series()

        # test mode None and timesteps None (default)
        self.edisgo.analyze()
        assert self.edisgo.results.v_res.shape == (4, 140)

        # test mode "mv" and timesteps given
        self.edisgo.analyze(mode="mv", timesteps=self.edisgo.timeseries.timeindex[0])
        assert self.edisgo.results.v_res.shape == (1, 31)

        # test mode "lv"
        self.edisgo.analyze(mode="lv", lv_grid_name="LVGrid_1")
        assert self.edisgo.results.v_res.shape == (4, 15)

        # ToDo: test non convergence

    def test_reinforce(self):
        self.setup_worst_case_time_series()
        results = self.edisgo.reinforce(combined_analysis=True)
        assert results.unresolved_issues.empty
        assert len(results.grid_expansion_costs) == 10
        assert len(results.equipment_changes) == 10
        # Todo: test other relevant values

    def test_add_component(self):

        self.setup_worst_case_time_series()
        index = self.edisgo.timeseries.timeindex
        dummy_ts = pd.Series(data=[0.1, 0.2, 0.1, 0.2], index=index)

        # Test add bus
        num_buses = len(self.edisgo.topology.buses_df)
        bus_name = self.edisgo.add_component(
            comp_type="bus", bus_name="Testbus", v_nom=20
        )
        assert bus_name == "Testbus"
        assert len(self.edisgo.topology.buses_df) == num_buses + 1
        assert self.edisgo.topology.buses_df.loc["Testbus", "v_nom"] == 20

        # Test add line
        num_lines = len(self.edisgo.topology.lines_df)
        line_name = self.edisgo.add_component(
            comp_type="line",
            bus0="Bus_MVStation_1",
            bus1="Testbus",
            length=0.001,
            type_info="NA2XS2Y 3x1x185 RM/25",
        )
        assert line_name == "Line_Bus_MVStation_1_Testbus"
        assert len(self.edisgo.topology.lines_df) == num_lines + 1
        assert self.edisgo.topology.lines_df.loc[line_name, "bus0"] == "Bus_MVStation_1"
        assert self.edisgo.topology.lines_df.loc[line_name, "bus1"] == "Testbus"
        assert self.edisgo.topology.lines_df.loc[line_name, "length"] == 0.001

        # Test add load (with time series)
        num_loads = len(self.edisgo.topology.loads_df)
        load_name = self.edisgo.add_component(
            comp_type="load",
            type="conventional_load",
            load_id=4,
            bus="Testbus",
            p_set=0.2,
            annual_consumption=3.2,
            sector="residential",
            ts_active_power=dummy_ts,
            ts_reactive_power=dummy_ts,
        )
        assert load_name == "Conventional_Load_MVGrid_1_residential_4"
        assert len(self.edisgo.topology.loads_df) == num_loads + 1
        assert self.edisgo.topology.loads_df.loc[load_name, "bus"] == "Testbus"
        assert self.edisgo.topology.loads_df.loc[load_name, "p_set"] == 0.2
        assert self.edisgo.topology.loads_df.loc[load_name, "annual_consumption"] == 3.2
        assert self.edisgo.topology.loads_df.loc[load_name, "sector"] == "residential"
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power.loc[index[0], load_name],
            0.1,
        )
        assert np.isclose(
            self.edisgo.timeseries.loads_reactive_power.loc[index[1], load_name],
            0.2,
        )

        # Test add generator (without time series)
        num_gens = len(self.edisgo.topology.generators_df)
        gen_name = self.edisgo.add_component(
            "generator",
            add_ts=False,
            generator_id=5,
            bus="Testbus",
            p_nom=2.5,
            generator_type="solar",
        )
        assert gen_name == "Generator_MVGrid_1_solar_5"
        assert len(self.edisgo.topology.generators_df) == num_gens + 1
        assert self.edisgo.topology.generators_df.loc[gen_name, "bus"] == "Testbus"
        assert self.edisgo.topology.generators_df.loc[gen_name, "p_nom"] == 2.5
        assert self.edisgo.topology.generators_df.loc[gen_name, "type"] == "solar"
        assert self.edisgo.timeseries.generators_active_power.shape == (4, num_gens)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (4, num_gens)

        # Test add storage unit
        num_storages = len(self.edisgo.topology.storage_units_df)
        storage_name = self.edisgo.add_component(
            comp_type="storage_unit", bus="Testbus", p_nom=3.1, add_ts=False
        )
        assert storage_name == "StorageUnit_MVGrid_1_2"
        assert len(self.edisgo.topology.storage_units_df) == num_storages + 1
        assert (
            self.edisgo.topology.storage_units_df.loc[storage_name, "bus"] == "Testbus"
        )
        assert self.edisgo.topology.storage_units_df.loc[storage_name, "p_nom"] == 3.1

    def test_integrate_component(self):

        self.setup_worst_case_time_series()

        num_gens = len(self.edisgo.topology.generators_df)

        random_bus = "Bus_BranchTee_MVGrid_1_1"
        x = self.edisgo.topology.buses_df.at[random_bus, "x"]
        y = self.edisgo.topology.buses_df.at[random_bus, "y"]
        geom = Point((x, y))

        # ##### MV integration
        # test generator integration by voltage level, geom as tuple, without
        # time series
        comp_data = {
            "generator_id": 13,
            "p_nom": 4,
            "generator_type": "misc",
            "subtype": "misc_sub",
        }
        comp_name = self.edisgo.integrate_component(
            comp_type="generator",
            geolocation=(x, y),
            voltage_level=4,
            add_ts=False,
            **comp_data
        )

        assert len(self.edisgo.topology.generators_df) == num_gens + 1
        assert self.edisgo.topology.generators_df.at[comp_name, "subtype"] == "misc_sub"
        # check that generator is directly connected to HV/MV station
        assert (
            self.edisgo.topology.get_connected_lines_from_bus(
                self.edisgo.topology.generators_df.at[comp_name, "bus"]
            ).bus0[0]
            == "Bus_MVStation_1"
        )

        # test charging point integration by nominal power, geom as shapely
        # Point, with time series
        num_cps = len(self.edisgo.topology.charging_points_df)

        comp_data = {"p_set": 4, "sector": "fast"}
        dummy_ts = pd.Series(
            data=[0.1, 0.2, 0.1, 0.2], index=self.edisgo.timeseries.timeindex
        )
        ts_active_power = dummy_ts
        ts_reactive_power = dummy_ts

        comp_name = self.edisgo.integrate_component(
            comp_type="charging_point",
            geolocation=geom,
            ts_active_power=ts_active_power,
            ts_reactive_power=ts_reactive_power,
            **comp_data
        )

        assert len(self.edisgo.topology.charging_points_df) == num_cps + 1
        assert self.edisgo.topology.charging_points_df.at[comp_name, "sector"] == "fast"
        # check voltage level
        assert (
            self.edisgo.topology.buses_df.at[
                self.edisgo.topology.charging_points_df.at[comp_name, "bus"],
                "v_nom",
            ]
            == 20
        )
        # check that charging point is connected to the random bus chosen
        # above
        assert (
            self.edisgo.topology.get_connected_lines_from_bus(
                self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
            ).bus0[0]
            == random_bus
        )
        # check time series
        assert (
            self.edisgo.timeseries.loads_active_power.loc[:, comp_name].values
            == [0.1, 0.2, 0.1, 0.2]
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp_name].values
            == [0.1, 0.2, 0.1, 0.2]
        ).all()

        # ##### LV integration

        # test charging point integration by nominal power, geom as shapely
        # Point, with time series
        comp_data = {"number": 13, "p_set": 0.04, "sector": "fast"}
        comp_name = self.edisgo.integrate_component(
            comp_type="charging_point",
            geolocation=geom,
            ts_active_power=ts_active_power,
            ts_reactive_power=ts_reactive_power,
            **comp_data
        )

        assert len(self.edisgo.topology.charging_points_df) == num_cps + 2
        assert self.edisgo.topology.charging_points_df.at[comp_name, "number"] == 13
        # check bus
        assert (
            self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
            == "Bus_BranchTee_LVGrid_1_3"
        )
        # check time series
        assert (
            self.edisgo.timeseries.loads_active_power.loc[:, comp_name].values
            == [0.1, 0.2, 0.1, 0.2]
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp_name].values
            == [0.1, 0.2, 0.1, 0.2]
        ).all()

    def test_remove_component(self):

        self.setup_worst_case_time_series()

        # Test remove bus (where bus cannot be removed, because load is still connected)
        num_buses = len(self.edisgo.topology.buses_df)
        self.edisgo.remove_component(
            comp_type="bus", comp_name="Bus_BranchTee_LVGrid_2_2"
        )
        assert len(self.edisgo.topology.buses_df) == num_buses
        assert "Bus_BranchTee_LVGrid_2_2" in self.edisgo.topology.buses_df.index

        # Test remove load (with time series)
        num_loads = len(self.edisgo.topology.loads_df)
        load_name = "Load_residential_LVGrid_1_6"
        self.edisgo.remove_component(
            comp_type="load",
            comp_name=load_name,
        )
        assert len(self.edisgo.topology.loads_df) == num_loads - 1
        assert load_name not in self.edisgo.timeseries.loads_active_power.columns
        assert load_name not in self.edisgo.timeseries.loads_reactive_power.columns

        # Test remove line
        num_lines = len(self.edisgo.topology.lines_df)
        self.edisgo.remove_component(comp_type="line", comp_name="Line_20000002")
        assert len(self.edisgo.topology.lines_df) == num_lines

        # Test remove generator (without time series)
        num_gens = len(self.edisgo.topology.generators_df)
        self.edisgo.remove_component(
            "generator", comp_name="GeneratorFluctuating_10", drop_ts=False
        )
        assert len(self.edisgo.topology.generators_df) == num_gens - 1
        assert self.edisgo.timeseries.generators_active_power.shape == (4, num_gens)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (4, num_gens)

        # Test remove storage unit (with time series)
        num_storages = len(self.edisgo.topology.storage_units_df)
        self.edisgo.remove_component(comp_type="storage_unit", comp_name="Storage_1")
        assert len(self.edisgo.topology.storage_units_df) == num_storages - 1
        assert load_name not in self.edisgo.timeseries.loads_active_power.columns
        assert load_name not in self.edisgo.timeseries.loads_reactive_power.columns

    def test_aggregate_components(self):

        self.setup_worst_case_time_series()

        # ##### test without any aggregation

        self.edisgo.topology._loads_df.at[
            "Load_residential_LVGrid_1_4", "bus"
        ] = "Bus_BranchTee_LVGrid_1_10"

        # save original values
        number_gens_before = len(self.edisgo.topology.generators_df)
        number_loads_before = len(self.edisgo.topology.loads_df)

        self.edisgo.aggregate_components(
            aggregate_generators_by_cols=[], aggregate_loads_by_cols=[]
        )

        assert number_gens_before == len(self.edisgo.topology.generators_df)
        assert number_loads_before == len(self.edisgo.topology.loads_df)

        # ##### test default (aggregate by bus only) - same EDisGo object as above
        # is used

        # save original values
        gens_p_nom_before = self.edisgo.topology.generators_df.p_nom.sum()
        loads_p_set_before = self.edisgo.topology.loads_df.p_set.sum()
        gens_feedin_before = self.edisgo.timeseries.generators_active_power.sum().sum()
        gens_feedin_reactive_before = (
            self.edisgo.timeseries.generators_reactive_power.sum().sum()
        )
        loads_demand_before = self.edisgo.timeseries.loads_active_power.sum().sum()
        loads_demand_reactive_before = (
            self.edisgo.timeseries.loads_reactive_power.sum().sum()
        )
        num_gens_before = len(self.edisgo.topology.generators_df)
        num_loads_before = len(self.edisgo.topology.loads_df)
        feedin_before = (
            self.edisgo.timeseries.generators_active_power.loc[
                :, ["GeneratorFluctuating_13", "GeneratorFluctuating_14"]
            ]
            .sum()
            .sum()
        )
        load_before = (
            self.edisgo.timeseries.loads_active_power.loc[
                :,
                ["Load_residential_LVGrid_1_5", "Load_residential_LVGrid_1_4"],
            ]
            .sum()
            .sum()
        )

        self.edisgo.aggregate_components()

        # test that total p_nom and total feed-in/demand stayed the same
        assert np.isclose(
            gens_p_nom_before, self.edisgo.topology.generators_df.p_nom.sum()
        )
        assert np.isclose(
            gens_feedin_before,
            self.edisgo.timeseries.generators_active_power.sum().sum(),
        )
        assert np.isclose(
            gens_feedin_reactive_before,
            self.edisgo.timeseries.generators_reactive_power.sum().sum(),
        )
        assert np.isclose(loads_p_set_before, self.edisgo.topology.loads_df.p_set.sum())
        assert np.isclose(
            loads_demand_before,
            self.edisgo.timeseries.loads_active_power.sum().sum(),
        )
        assert np.isclose(
            loads_demand_reactive_before,
            self.edisgo.timeseries.loads_reactive_power.sum().sum(),
        )
        # test that two generators and two loads were aggregated
        assert num_gens_before - 4 == len(self.edisgo.topology.generators_df)
        assert (
            self.edisgo.topology.generators_df.at[
                "Generators_Bus_BranchTee_LVGrid_1_14", "p_nom"
            ]
            == 0.034
        )
        assert (
            self.edisgo.timeseries.generators_active_power.loc[
                :, "Generators_Bus_BranchTee_LVGrid_1_14"
            ].sum()
            == feedin_before
        )
        assert num_loads_before - 1 == len(self.edisgo.topology.loads_df)
        assert self.edisgo.topology.loads_df.at[
            "Loads_Bus_BranchTee_LVGrid_1_10", "p_set"
        ] == (2 * 0.001397)
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Loads_Bus_BranchTee_LVGrid_1_10"
            ].sum()
            == load_before
        )
        # test that analyze does not fail
        self.edisgo.analyze()

        # ##### test with charging points, aggregation of loads by bus, type and sector
        # and aggregation of generators only by bus

        # reset EDisGo object
        self.setup_edisgo_object()
        self.setup_worst_case_time_series()

        # add charging point
        self.edisgo.add_component(
            comp_type="load",
            ts_active_power=pd.Series(
                data=[0.1, 0.2, 0.1, 0.2], index=self.edisgo.timeseries.timeindex
            ),
            ts_reactive_power=pd.Series(
                data=[0, 0, 0, 0], index=self.edisgo.timeseries.timeindex
            ),
            bus="Bus_BranchTee_LVGrid_1_10",
            type="charging_point",
            sector="home",
            p_set=0.2,
        )
        # manipulate grid so that more than one load of the same sector is
        # connected at the same bus
        self.edisgo.topology._loads_df.at[
            "Load_residential_LVGrid_1_4", "bus"
        ] = "Bus_BranchTee_LVGrid_1_10"

        # save original values (only loads, as generators did not change)
        loads_p_set_before = self.edisgo.topology.loads_df.p_set.sum()
        loads_demand_before = self.edisgo.timeseries.loads_active_power.sum().sum()
        loads_demand_reactive_before = (
            self.edisgo.timeseries.loads_reactive_power.sum().sum()
        )
        num_loads_before = len(self.edisgo.topology.loads_df)

        self.edisgo.aggregate_components(
            aggregate_loads_by_cols=["bus", "type", "sector"],
            aggregate_generators_by_cols=["bus"],
        )

        # test that total p_nom and total feed-in/demand stayed the same
        assert np.isclose(
            gens_p_nom_before, self.edisgo.topology.generators_df.p_nom.sum()
        )
        assert np.isclose(
            gens_feedin_before,
            self.edisgo.timeseries.generators_active_power.sum().sum(),
        )
        assert np.isclose(
            gens_feedin_reactive_before,
            self.edisgo.timeseries.generators_reactive_power.sum().sum(),
        )
        assert np.isclose(
            loads_p_set_before,
            self.edisgo.topology.loads_df.p_set.sum(),
        )
        assert np.isclose(
            loads_demand_before,
            self.edisgo.timeseries.loads_active_power.sum().sum(),
        )
        assert np.isclose(
            loads_demand_reactive_before,
            self.edisgo.timeseries.loads_reactive_power.sum().sum(),
        )
        charging_points_df = self.edisgo.topology.charging_points_df
        assert np.isclose(0.2, charging_points_df.p_set.sum())
        assert np.isclose(
            0.6,
            self.edisgo.timeseries.loads_active_power.loc[:, charging_points_df.index]
            .sum()
            .sum(),
        )
        assert np.isclose(
            0,
            self.edisgo.timeseries.loads_reactive_power.loc[:, charging_points_df.index]
            .sum()
            .sum(),
        )
        # test that generators were aggregated
        assert num_gens_before - 4 == len(self.edisgo.topology.generators_df)
        # test that two loads were aggregated and that charging point was not aggregated
        # with load
        assert num_loads_before - 1 == len(self.edisgo.topology.loads_df)
        assert self.edisgo.topology.loads_df.at[
            "Loads_Bus_BranchTee_LVGrid_1_10_conventional_load_residential", "p_set"
        ] == (2 * 0.001397)
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Loads_Bus_BranchTee_LVGrid_1_10_conventional_load_residential"
            ].sum()
            == load_before
        )

        # test that analyze does not fail
        self.edisgo.analyze()

        # #### test without aggregation of loads and aggregation of generators
        # by bus and type

        # reset EDisGo object
        self.setup_edisgo_object()
        self.setup_worst_case_time_series()

        # manipulate grid so that two generators of different types are
        # connected at the same bus
        self.edisgo.topology._generators_df.at[
            "GeneratorFluctuating_13", "type"
        ] = "misc"

        # save original values (values of loads were changed in previous aggregation)
        loads_p_set_before = self.edisgo.topology.loads_df.p_set.sum()
        loads_demand_before = self.edisgo.timeseries.loads_active_power.sum().sum()
        loads_demand_reactive_before = (
            self.edisgo.timeseries.loads_reactive_power.sum().sum()
        )
        num_loads_before = len(self.edisgo.topology.loads_df)

        self.edisgo.aggregate_components(
            aggregate_generators_by_cols=["bus", "type"], aggregate_loads_by_cols=[]
        )
        # test that total p_nom and total feed-in/demand stayed the same
        assert np.isclose(
            gens_p_nom_before, self.edisgo.topology.generators_df.p_nom.sum()
        )
        assert np.isclose(
            gens_feedin_before,
            self.edisgo.timeseries.generators_active_power.sum().sum(),
        )
        assert np.isclose(
            gens_feedin_reactive_before,
            self.edisgo.timeseries.generators_reactive_power.sum().sum(),
        )
        assert np.isclose(loads_p_set_before, self.edisgo.topology.loads_df.p_set.sum())
        assert np.isclose(
            loads_demand_before,
            self.edisgo.timeseries.loads_active_power.sum().sum(),
        )
        assert np.isclose(
            loads_demand_reactive_before,
            self.edisgo.timeseries.loads_reactive_power.sum().sum(),
        )
        # test that generators at the same bus were aggregated and loads stayed the same
        assert num_gens_before - 3 == len(self.edisgo.topology.generators_df)
        assert num_loads_before == len(self.edisgo.topology.loads_df)

        # test that analyze does not fail
        self.edisgo.analyze()

    def test_plot_mv_grid_topology(self):
        plt.ion()
        self.edisgo.plot_mv_grid_topology(technologies=True)
        plt.close("all")
        self.edisgo.plot_mv_grid_topology()
        plt.close("all")

    def test_plot_mv_voltages(self):
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.analyze()
        self.edisgo.plot_mv_voltages()
        plt.close("all")

    def test_plot_mv_line_loading(self):
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.analyze()
        self.edisgo.plot_mv_line_loading()
        plt.close("all")

    def test_plot_mv_grid_expansion_costs(self):
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.reinforce()
        self.edisgo.plot_mv_grid_expansion_costs()
        plt.close("all")

    def test_plot_mv_storage_integration(self):
        plt.ion()
        storage_1 = self.edisgo.topology.add_storage_unit(
            "Bus_BranchTee_MVGrid_1_8", 0.3
        )
        storage_2 = self.edisgo.topology.add_storage_unit(
            "Bus_BranchTee_MVGrid_1_8", 0.6
        )
        storage_3 = self.edisgo.topology.add_storage_unit(
            "Bus_BranchTee_MVGrid_1_10", 0.3
        )
        self.edisgo.plot_mv_storage_integration()
        plt.close("all")
        self.edisgo.topology.remove_storage_unit(storage_1)
        self.edisgo.topology.remove_storage_unit(storage_2)
        self.edisgo.topology.remove_storage_unit(storage_3)

    def test_histogramm_voltage(self):
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.analyze()
        self.edisgo.histogram_voltage()
        plt.close("all")

    def test_histogramm_relative_line_load(self):
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.analyze()
        self.edisgo.histogram_relative_line_load()
        plt.close("all")

    def test_save(self):
        save_dir = os.path.join(os.getcwd(), "edisgo_network")
        self.edisgo.save(save_dir)

        # check that results, topology and timeseries directory are created
        dirs_in_save_dir = os.listdir(save_dir)
        assert len(dirs_in_save_dir) == 3
        # Todo: check anything else?
        shutil.rmtree(os.path.join(save_dir, "results"))
        shutil.rmtree(os.path.join(save_dir, "topology"))
        shutil.rmtree(os.path.join(save_dir, "timeseries"))

    def test_reduce_memory(self):

        self.setup_worst_case_time_series()
        self.edisgo.analyze()

        # check one time series attribute and one results attribute
        mem_ts_before = self.edisgo.timeseries.generators_active_power.memory_usage(
            deep=True
        ).sum()
        mem_res_before = self.edisgo.results.pfa_p.memory_usage(deep=True).sum()

        # check with default value
        self.edisgo.reduce_memory()

        mem_ts_with_default = (
            self.edisgo.timeseries.generators_active_power.memory_usage(deep=True).sum()
        )
        mem_res_with_default = self.edisgo.results.pfa_p.memory_usage(deep=True).sum()

        assert mem_ts_before > mem_ts_with_default
        assert mem_res_before > mem_res_with_default

        mem_ts_with_default_2 = self.edisgo.timeseries.loads_active_power.memory_usage(
            deep=True
        ).sum()
        mem_res_with_default_2 = self.edisgo.results.i_res.memory_usage(deep=True).sum()

        # check passing kwargs
        self.edisgo.reduce_memory(
            to_type="float16",
            results_attr_to_reduce=["pfa_p"],
            timeseries_attr_to_reduce=["generators_active_power"],
        )

        assert (
            mem_ts_with_default
            > self.edisgo.timeseries.generators_active_power.memory_usage(
                deep=True
            ).sum()
        )
        assert (
            mem_res_with_default
            > self.edisgo.results.pfa_p.memory_usage(deep=True).sum()
        )
        # check that i_res and loads_active_power were not reduced
        assert np.isclose(
            mem_ts_with_default_2,
            self.edisgo.timeseries.loads_active_power.memory_usage(deep=True).sum(),
        )
        assert np.isclose(
            mem_res_with_default_2,
            self.edisgo.results.i_res.memory_usage(deep=True).sum(),
        )
