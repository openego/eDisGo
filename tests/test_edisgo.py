import copy
import logging
import os
import shutil

from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from matplotlib import pyplot as plt
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files


class TestEDisGo:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """
        Fixture to set up new EDisGo object before each test function.

        """
        self.setup_edisgo_object()

    def setup_edisgo_object(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def setup_worst_case_time_series(self):
        self.edisgo.set_time_series_worst_case_analysis()

    def test_config_setter(self):

        save_dir = os.path.join(os.getcwd(), "config_dir")

        # test default
        config_orig = copy.deepcopy(self.edisgo.config)
        self.edisgo.config = {}
        assert config_orig._data == self.edisgo.config._data

        # test specifying different directory
        self.edisgo.config = {"config_path": save_dir}
        assert len(os.listdir(save_dir)) == 5
        shutil.rmtree(save_dir)

        # test json and config_path=None
        # save changed config to json
        self.edisgo.config["geo"]["srid"] = 2
        config_json = copy.deepcopy(self.edisgo.config)
        self.edisgo.save(
            save_dir,
            save_topology=False,
            save_timeseries=False,
            save_results=False,
            save_electromobility=False,
        )
        # overwrite config with config_path=None and check
        self.edisgo.config = {"config_path": None}
        assert config_orig._data == self.edisgo.config._data
        # overwrite config from json and check
        self.edisgo.config = {"from_json": True, "config_path": save_dir}
        assert config_json._data == self.edisgo.config._data

        # delete directory
        shutil.rmtree(save_dir)

    def test_set_time_series_manual(self, caplog):

        timeindex = pd.date_range("1/1/2018", periods=3, freq="H")
        gens_ts = pd.DataFrame(
            data={
                "GeneratorFluctuating_15": [2.0, 5.0, 6.0],
                "GeneratorFluctuating_24": [4.0, 7.0, 8.0],
            },
            index=timeindex,
        )
        loads_ts = pd.DataFrame(
            data={
                "Load_residential_LVGrid_5_3": [2.0, 5.0, 6.0],
            },
            index=timeindex,
        )
        storage_units_ts = pd.DataFrame(
            data={
                "Storage_1": [4.0, 7.0, 8.0],
            },
            index=timeindex,
        )

        # test setting some time series and with no time index being previously set
        with caplog.at_level(logging.WARNING):
            self.edisgo.set_time_series_manual(
                generators_p=gens_ts,
                generators_q=gens_ts,
                loads_p=loads_ts,
                storage_units_q=storage_units_ts,
            )
        assert (
            "When setting time series manually a time index is not automatically "
            "set" in caplog.text
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (0, 2)
        self.edisgo.set_timeindex(timeindex)
        assert_frame_equal(gens_ts, self.edisgo.timeseries.generators_active_power)
        assert_frame_equal(gens_ts, self.edisgo.timeseries.generators_reactive_power)
        assert_frame_equal(loads_ts, self.edisgo.timeseries.loads_active_power)
        assert self.edisgo.timeseries.loads_reactive_power.empty
        assert self.edisgo.timeseries.storage_units_active_power.empty
        assert_frame_equal(
            storage_units_ts, self.edisgo.timeseries.storage_units_reactive_power
        )

        # test overwriting time series and with some components that do not exist
        timeindex2 = pd.date_range("1/1/2018", periods=4, freq="H")
        gens_ts2 = pd.DataFrame(
            data={
                "GeneratorFluctuating_15": [1.0, 2.0, 5.0, 6.0],
                "GeneratorFluctuating_14": [5.0, 2.0, 5.0, 6.0],
                "GeneratorFluctuating_x": [8.0, 4.0, 7.0, 8.0],
            },
            index=timeindex2,
        )
        loads_ts2 = pd.DataFrame(
            data={
                "Load_residential_LVGrid_5_3": [2.0, 5.0, 6.0],
                "Load_residential_LVGrid_x": [2.0, 5.0, 6.0],
            },
            index=timeindex,
        )
        self.edisgo.set_time_series_manual(
            generators_p=gens_ts2, loads_p=loads_ts2, storage_units_p=storage_units_ts
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (3, 3)
        assert_frame_equal(
            gens_ts2.loc[
                timeindex, ["GeneratorFluctuating_15", "GeneratorFluctuating_14"]
            ],
            self.edisgo.timeseries.generators_active_power.loc[
                :, ["GeneratorFluctuating_15", "GeneratorFluctuating_14"]
            ],
        )
        assert_frame_equal(
            gens_ts.loc[:, ["GeneratorFluctuating_24"]],
            self.edisgo.timeseries.generators_active_power.loc[
                :, ["GeneratorFluctuating_24"]
            ],
        )
        assert_frame_equal(gens_ts, self.edisgo.timeseries.generators_reactive_power)
        assert_frame_equal(
            loads_ts2.loc[:, ["Load_residential_LVGrid_5_3"]],
            self.edisgo.timeseries.loads_active_power.loc[
                :, ["Load_residential_LVGrid_5_3"]
            ],
        )
        assert self.edisgo.timeseries.loads_reactive_power.empty
        assert_frame_equal(
            storage_units_ts, self.edisgo.timeseries.storage_units_active_power
        )
        assert_frame_equal(
            storage_units_ts, self.edisgo.timeseries.storage_units_reactive_power
        )

    def test_set_time_series_worst_case_analysis(self):
        self.edisgo.set_time_series_worst_case_analysis(
            cases="load_case", generators_names=["Generator_1"], loads_names=[]
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (2, 1)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 1)
        assert self.edisgo.timeseries.loads_active_power.shape == (2, 0)
        assert self.edisgo.timeseries.loads_reactive_power.shape == (2, 0)
        assert self.edisgo.timeseries.storage_units_active_power.shape == (2, 1)
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (2, 1)

        self.edisgo.set_time_series_worst_case_analysis()
        assert self.edisgo.timeseries.generators_active_power.shape == (
            4,
            len(self.edisgo.topology.generators_df),
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (
            4,
            len(self.edisgo.topology.generators_df),
        )
        assert self.edisgo.timeseries.loads_active_power.shape == (
            4,
            len(self.edisgo.topology.loads_df),
        )
        assert self.edisgo.timeseries.loads_reactive_power.shape == (
            4,
            len(self.edisgo.topology.loads_df),
        )
        assert self.edisgo.timeseries.storage_units_active_power.shape == (
            4,
            len(self.edisgo.topology.storage_units_df),
        )
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (
            4,
            len(self.edisgo.topology.storage_units_df),
        )

    def test_set_time_series_active_power_predefined(self, caplog):

        # check warning
        self.edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts="oedb"
        )
        assert (
            "When setting time series using predefined profiles a time index is"
            in caplog.text
        )

        # check if right functions are called
        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        self.edisgo.timeseries.timeindex = timeindex
        ts_fluc = pd.DataFrame(
            data={
                "wind": [5, 6],
            },
            index=timeindex,
        )
        ts_disp = pd.DataFrame(
            data={
                "other": [5, 6],
            },
            index=timeindex,
        )
        ts_cp = pd.DataFrame(
            data={
                "hpc": [5, 6],
            },
            index=timeindex,
        )
        self.edisgo.topology._loads_df.loc[
            "Load_residential_LVGrid_1_4", ["type", "sector"]
        ] = ("charging_point", "hpc")

        self.edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=ts_fluc,
            fluctuating_generators_names=["GeneratorFluctuating_8"],
            dispatchable_generators_ts=ts_disp,
            dispatchable_generators_names=["Generator_1"],
            conventional_loads_ts="demandlib",
            conventional_loads_names=[
                "Load_residential_LVGrid_3_2",
                "Load_residential_LVGrid_3_3",
            ],
            charging_points_ts=ts_cp,
            charging_points_names=None,
        )

        assert self.edisgo.timeseries.generators_active_power.shape == (2, 2)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 0)
        assert self.edisgo.timeseries.loads_active_power.shape == (2, 3)
        assert self.edisgo.timeseries.loads_reactive_power.shape == (2, 0)
        assert self.edisgo.timeseries.storage_units_active_power.shape == (2, 0)
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (2, 0)

    def test_set_time_series_reactive_power_control(self):
        # set active power time series for fixed cosphi
        timeindex = pd.date_range("1/1/1970", periods=3, freq="H")
        self.edisgo.set_timeindex(timeindex)
        ts_solar = np.array([0.1, 0.2, 0.3])
        ts_wind = [0.4, 0.5, 0.6]
        self.edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=pd.DataFrame(
                {"solar": ts_solar, "wind": ts_wind}, index=timeindex
            ),
            dispatchable_generators_ts=pd.DataFrame(
                {"other": ts_solar}, index=timeindex
            ),
            conventional_loads_ts="demandlib",
        )

        # test only setting reactive power for one generator
        gen = "GeneratorFluctuating_4"  # solar MV generator
        self.edisgo.set_time_series_reactive_power_control(
            generators_parametrisation=pd.DataFrame(
                {
                    "components": [[gen]],
                    "mode": ["default"],
                    "power_factor": ["default"],
                },
                index=[1],
            ),
            loads_parametrisation=None,
            storage_units_parametrisation=None,
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (3, 1)
        assert self.edisgo.timeseries.loads_reactive_power.empty
        assert self.edisgo.timeseries.storage_units_reactive_power.empty
        assert (
            np.isclose(
                self.edisgo.timeseries.generators_reactive_power.loc[:, gen],
                ts_solar * -np.tan(np.arccos(0.9)) * 1.93,
            )
        ).all()

        # test changing only configuration of one load
        load = "Load_residential_LVGrid_1_5"
        self.edisgo.set_time_series_reactive_power_control(
            loads_parametrisation=pd.DataFrame(
                {
                    "components": [
                        [load],
                        self.edisgo.topology.loads_df.index.drop([load]),
                    ],
                    "mode": ["capacitive", "default"],
                    "power_factor": [0.98, "default"],
                },
                index=[1, 2],
            ),
            storage_units_parametrisation=None,
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (3, 28)
        assert self.edisgo.timeseries.loads_reactive_power.shape == (3, 50)
        assert (
            np.isclose(
                self.edisgo.timeseries.loads_reactive_power.loc[:, load],
                self.edisgo.timeseries.loads_active_power.loc[:, load]
                * -np.tan(np.arccos(0.98)),
            )
        ).all()

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
        msg = "Provide proper mode or leave it empty to export entire network topology."
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
        assert len(edisgo.topology.generators_df) == 524

    def test_analyze(self, caplog):
        self.setup_worst_case_time_series()

        # test mode None and timesteps None (default)
        self.edisgo.analyze()
        results_analyze = deepcopy(self.edisgo.results)
        assert self.edisgo.results.v_res.shape == (4, 140)

        # test mode "mv" and timesteps given
        self.edisgo.analyze(mode="mv", timesteps=self.edisgo.timeseries.timeindex[0])
        assert self.edisgo.results.v_res.shape == (1, 31)

        # test mode "lv"
        self.edisgo.analyze(mode="lv", lv_grid_id=1)
        assert self.edisgo.results.v_res.shape == (4, 15)

        # test troubleshooting_mode "lpf"
        self.edisgo.analyze(troubleshooting_mode="lpf")
        assert self.edisgo.results.v_res.shape == (4, 140)
        assert self.edisgo.results.equality_check(results_analyze)

        # test mode None and troubleshooting_mode "iteration"
        self.edisgo.analyze(troubleshooting_mode="iteration")
        assert self.edisgo.results.v_res.shape == (4, 140)
        assert self.edisgo.results.equality_check(results_analyze)

        # test non convergence
        msg = "Power flow analysis did not converge for the"
        with pytest.raises(ValueError, match=msg):
            self.edisgo.analyze(troubleshooting_mode="iteration", range_start=5)

        caplog.clear()
        self.edisgo.analyze(
            troubleshooting_mode="iteration",
            range_start=5,
            range_num=2,
            raise_not_converged=False,
        )
        assert "Current fraction in iterative process: 5.0." in caplog.text
        assert "Current fraction in iterative process: 1.0." in caplog.text

    def test_reinforce(self):

        # ###################### test with default settings ##########################
        self.setup_worst_case_time_series()
        results = self.edisgo.reinforce()
        assert results.unresolved_issues.empty
        assert len(results.grid_expansion_costs) == 10
        assert len(results.equipment_changes) == 10
        assert results.v_res.shape == (4, 140)
        assert self.edisgo.results.v_res.shape == (4, 140)

        # ###################### test mode lv and copy grid ##########################
        self.setup_edisgo_object()
        self.setup_worst_case_time_series()
        results = self.edisgo.reinforce(mode="lv", copy_grid=True)
        assert results.unresolved_issues.empty
        assert len(results.grid_expansion_costs) == 6
        assert len(results.equipment_changes) == 6
        assert results.v_res.shape == (2, 140)
        assert self.edisgo.results.v_res.empty

        # ################# test mode mvlv and combined analysis ####################
        # self.setup_edisgo_object()
        # self.setup_worst_case_time_series()
        results = self.edisgo.reinforce(mode="mvlv", combined_analysis=False)
        assert results.unresolved_issues.empty
        assert len(results.grid_expansion_costs) == 8
        assert len(results.equipment_changes) == 8
        assert results.v_res.shape == (4, 41)

    def test_add_component(self, caplog):
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
        assert (
            self.edisgo.timeseries.loads_active_power.loc[:, load_name] == dummy_ts
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[:, load_name] == dummy_ts
        ).all()

        # Test add load (with reactive power time series default mode)
        load_name = self.edisgo.add_component(
            comp_type="load",
            type="conventional_load",
            load_id=4,
            bus="Testbus",
            p_set=0.2,
            annual_consumption=3.2,
            sector="residential",
            ts_active_power=dummy_ts,
            ts_reactive_power="default",
        )
        assert (
            self.edisgo.timeseries.loads_active_power.loc[:, load_name] == dummy_ts
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[:, load_name]
            == dummy_ts * np.tan(np.arccos(0.9))
        ).all()
        # check that reactive power time series were not all set to default
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Conventional_Load_MVGrid_1_residential_4"
            ]
            == dummy_ts
        ).all()

        # Test add generator (without time series)
        num_gens = len(self.edisgo.topology.generators_df)
        gen_name = self.edisgo.add_component(
            "generator",
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

        # Test add generator (test that warning is raised when no active power time
        # series is provided for default mode)
        gen_name = self.edisgo.add_component(
            "generator",
            generator_id=5,
            bus="Testbus",
            p_nom=2.5,
            generator_type="solar",
            ts_reactive_power="default",
        )
        assert (
            f"Default reactive power time series of {gen_name} cannot be set as "
            f"active power time series was not provided." in caplog.text
        )

        # Test add generator (with reactive power time series default mode)
        gen_name = self.edisgo.add_component(
            "generator",
            generator_id=5,
            bus="Testbus",
            p_nom=2.5,
            generator_type="solar",
            ts_active_power=dummy_ts,
            ts_reactive_power="default",
        )
        assert (
            self.edisgo.timeseries.generators_reactive_power.loc[:, gen_name]
            == dummy_ts * -np.tan(np.arccos(0.9))
        ).all()

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

    def test_integrate_component_based_on_geolocation(self):
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
        comp_name = self.edisgo.integrate_component_based_on_geolocation(
            comp_type="generator",
            geolocation=(x, y),
            voltage_level=4,
            add_ts=False,
            **comp_data,
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

        comp_name = self.edisgo.integrate_component_based_on_geolocation(
            comp_type="charging_point",
            geolocation=geom,
            ts_active_power=ts_active_power,
            ts_reactive_power=ts_reactive_power,
            **comp_data,
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

        # test heat pump integration by nominal power, geom as shapely
        # Point, with time series
        num_loads = len(self.edisgo.topology.loads_df)

        comp_data = {"p_set": 2.5}
        dummy_ts = pd.Series(
            data=[0.1, 0.2, 0.1, 0.2], index=self.edisgo.timeseries.timeindex
        )
        ts_active_power = dummy_ts
        ts_reactive_power = dummy_ts

        comp_name = self.edisgo.integrate_component_based_on_geolocation(
            comp_type="heat_pump",
            geolocation=geom,
            ts_active_power=ts_active_power,
            ts_reactive_power=ts_reactive_power,
            **comp_data,
        )

        assert len(self.edisgo.topology.loads_df) == num_loads + 1
        assert self.edisgo.topology.loads_df.at[comp_name, "type"] == "heat_pump"
        # check voltage level
        bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert (
            self.edisgo.topology.buses_df.at[
                bus,
                "v_nom",
            ]
            == 20
        )
        # check that charging point is connected to the random bus chosen
        # above
        assert (
            self.edisgo.topology.get_connected_lines_from_bus(bus).bus0[0] == random_bus
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
        comp_name = self.edisgo.integrate_component_based_on_geolocation(
            comp_type="charging_point",
            geolocation=geom,
            ts_active_power=ts_active_power,
            ts_reactive_power=ts_reactive_power,
            **comp_data,
        )

        assert len(self.edisgo.topology.loads_df) == num_loads + 2
        assert self.edisgo.topology.charging_points_df.at[comp_name, "number"] == 13
        # check bus
        assert (
            self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
            == "Bus_BranchTee_LVGrid_1_7"
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

        # test heat pump integration by voltage level, geom as shapely
        # Point, without time series
        comp_data = {
            "voltage_level": 7,
            "sector": "individual_heating",
            "p_set": 0.2,
            "mvlv_subst_id": 3.0,
        }
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        comp_name = self.edisgo.integrate_component_based_on_geolocation(
            comp_type="heat_pump",
            geolocation=geom,
            add_ts=False,
            **comp_data,
        )

        assert len(self.edisgo.topology.loads_df) == num_loads + 3
        assert (
            self.edisgo.topology.loads_df.at[comp_name, "sector"]
            == "individual_heating"
        )
        # check bus
        assert (
            self.edisgo.topology.loads_df.at[comp_name, "bus"]
            == "Bus_BranchTee_LVGrid_3_2"
        )

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

    def test_import_electromobility(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)

        # test with default parameters
        simbev_path = pytest.simbev_example_scenario_path
        tracbev_path = pytest.tracbev_example_scenario_path
        self.edisgo.import_electromobility(simbev_path, tracbev_path)

        assert len(self.edisgo.electromobility.charging_processes_df) == 48
        assert len(self.edisgo.electromobility.potential_charging_parks_gdf) == 1621
        assert self.edisgo.electromobility.eta_charging_points == 0.9

        total_charging_demand_at_charging_parks = sum(
            cp.charging_processes_df.chargingdemand_kWh.sum()
            for cp in list(self.edisgo.electromobility.potential_charging_parks)
            if cp.designated_charging_point_capacity > 0
        )
        total_charging_demand = (
            self.edisgo.electromobility.charging_processes_df.chargingdemand_kWh.sum()
        )
        assert np.isclose(
            total_charging_demand_at_charging_parks, total_charging_demand
        )

        # fmt: off
        charging_park_ids = (
            self.edisgo.electromobility.charging_processes_df.charging_park_id.
            sort_values().unique()
        )
        potential_charging_parks_with_capacity = np.sort(
            [
                cp.id
                for cp in list(self.edisgo.electromobility.potential_charging_parks)
                if cp.designated_charging_point_capacity > 0.0
            ]
        )
        # fmt: on

        assert set(charging_park_ids) == set(potential_charging_parks_with_capacity)

        assert len(self.edisgo.electromobility.integrated_charging_parks_df) == 3

        # fmt: off
        assert set(
            self.edisgo.electromobility.integrated_charging_parks_df.edisgo_id.
            sort_values().values
        ) == set(
            self.edisgo.topology.loads_df[
                self.edisgo.topology.loads_df.type == "charging_point"
            ]
            .index.sort_values()
            .values
        )
        # fmt: on

        # test with kwargs
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        self.edisgo.import_electromobility(
            simbev_path,
            tracbev_path,
            {"mode_parking_times": "not_frugal"},
            {"mode": "grid_friendly"},
        )

        # Length of charging_processes_df, potential_charging_parks_gdf and
        # integrated_charging_parks_df changed compared to test without kwargs
        # TODO: needs to be checked if that is correct
        assert len(self.edisgo.electromobility.charging_processes_df) == 427
        assert len(self.edisgo.electromobility.potential_charging_parks_gdf) == 1621
        assert self.edisgo.electromobility.simulated_days == 7

        assert np.isclose(
            total_charging_demand,
            self.edisgo.electromobility.charging_processes_df.chargingdemand_kWh.sum(),
        )

        # fmt: off
        charging_park_ids = (
            self.edisgo.electromobility.charging_processes_df.charging_park_id.dropna(
            ).unique()
        )
        # fmt: on

        potential_charging_parks_with_capacity = np.sort(
            [
                cp.id
                for cp in list(self.edisgo.electromobility.potential_charging_parks)
                if cp.designated_charging_point_capacity > 0.0
            ]
        )
        assert set(charging_park_ids) == set(potential_charging_parks_with_capacity)

        assert len(self.edisgo.electromobility.integrated_charging_parks_df) == 3

        # fmt: off
        assert set(
            self.edisgo.electromobility.integrated_charging_parks_df.edisgo_id.
            sort_values().values
        ) == set(
            self.edisgo.topology.loads_df[
                self.edisgo.topology.loads_df.type == "charging_point"
            ]
            .index.sort_values()
            .values
        )
        # fmt: on

    def test_apply_charging_strategy(self):
        self.edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        self.edisgo_obj.set_timeindex(timeindex)

        self.edisgo_obj.resample_timeseries()
        # test with default parameters
        simbev_path = pytest.simbev_example_scenario_path
        tracbev_path = pytest.tracbev_example_scenario_path
        self.edisgo_obj.import_electromobility(simbev_path, tracbev_path)
        self.edisgo_obj.apply_charging_strategy()

        # Check if all charging points have a valid chargingdemand_kWh > 0
        cps = self.edisgo_obj.topology.loads_df[
            self.edisgo_obj.topology.loads_df.type == "charging_point"
        ].index
        ts = self.edisgo_obj.timeseries.loads_active_power.loc[:, cps]
        df = ts.loc[:, (ts <= 0).any(axis=0)]
        assert df.shape == ts.shape

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
        # test with storage
        self.setup_worst_case_time_series()
        plt.ion()
        self.edisgo.reinforce()
        self.edisgo.plot_mv_grid_expansion_costs()
        plt.close("all")

        # test without storage
        self.setup_edisgo_object()
        self.edisgo.remove_component("storage_unit", "Storage_1", False)
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
        self.setup_worst_case_time_series()
        save_dir = os.path.join(os.getcwd(), "edisgo_network")

        # add heat pump and electromobility dummy data
        self.edisgo.heat_pump.cop = pd.DataFrame(
            data={
                "hp1": [5.0, 6.0, 5.0, 6.0],
                "hp2": [7.0, 8.0, 7.0, 8.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.electromobility.charging_processes_df = pd.DataFrame(
            data={
                "ags": [5.0, 6.0],
                "car_id": [7.0, 8.0],
            },
            index=[0, 1],
        )
        self.edisgo.electromobility.potential_charging_parks_gdf = pd.DataFrame(
            data={
                "ags": [5.0, 6.0],
                "car_id": [7.0, 8.0],
            },
            index=[0, 1],
        )

        # ################### test with default parameters ###################
        self.edisgo.save(save_dir)

        # check that sub-directory are created
        dirs_in_save_dir = os.listdir(save_dir)
        assert len(dirs_in_save_dir) == 4
        assert "configs.json" in dirs_in_save_dir

        shutil.rmtree(save_dir)

        # ############## test with saving heat pump and electromobility #############
        self.edisgo.save(
            save_dir,
            save_electromobility=True,
            save_heatpump=True,
            electromobility_attributes=["charging_processes_df"],
        )

        # check that sub-directory are created
        dirs_in_save_dir = os.listdir(save_dir)
        assert len(dirs_in_save_dir) == 6
        assert "electromobility" in dirs_in_save_dir

        shutil.rmtree(save_dir)

        # ############## test with archiving and electromobility ##############
        self.edisgo.save(save_dir, archive=True, save_electromobility=True)
        zip_file = os.path.join(os.path.dirname(save_dir), "edisgo_network.zip")
        assert os.path.exists(zip_file)

        zip = ZipFile(zip_file)
        files = zip.namelist()
        zip.close()
        assert len(files) == 25

        os.remove(zip_file)

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

    def test_check_integrity(self, caplog):
        self.edisgo.check_integrity()
        assert (
            "The following generators are missing in generators_active_power: "
            "{}".format(self.edisgo.topology.generators_df.index.values) in caplog.text
        )
        assert (
            "The following generators are missing in generators_reactive_power: "
            "{}".format(self.edisgo.topology.generators_df.index.values) in caplog.text
        )
        assert (
            "The following loads are missing in loads_active_power: "
            "{}".format(self.edisgo.topology.loads_df.index.values) in caplog.text
        )
        assert (
            "The following loads are missing in loads_reactive_power: "
            "{}".format(self.edisgo.topology.loads_df.index.values) in caplog.text
        )
        assert (
            "The following storage_units are missing in storage_units_active_power"
            ": {}".format(self.edisgo.topology.storage_units_df.index.values)
            in caplog.text
        )
        assert (
            "The following storage_units are missing in storage_units_reactive_power"
            ": {}".format(self.edisgo.topology.storage_units_df.index.values)
            in caplog.text
        )
        caplog.clear()
        # set timeseries
        index = pd.date_range("1/1/2018", periods=3, freq="H")
        ts_gens = pd.DataFrame(
            index=index, columns=self.edisgo.topology.generators_df.index, data=0
        )
        ts_loads = pd.DataFrame(
            index=index, columns=self.edisgo.topology.loads_df.index, data=0
        )
        ts_stor = pd.DataFrame(
            index=index, columns=self.edisgo.topology.storage_units_df.index, data=0
        )
        self.edisgo.timeseries.timeindex = index
        self.edisgo.timeseries.generators_active_power = ts_gens
        self.edisgo.timeseries.generators_reactive_power = ts_gens
        self.edisgo.timeseries.loads_active_power = ts_loads
        self.edisgo.timeseries.loads_reactive_power = ts_loads
        self.edisgo.timeseries.storage_units_active_power = ts_stor
        self.edisgo.timeseries.storage_units_reactive_power = ts_stor
        # check that no warning is raised
        self.edisgo.check_integrity()
        assert not caplog.text
        manipulated_comps = {
            "generators": ["Generator_1", "GeneratorFluctuating_4"],
            "loads": ["Load_agricultural_LVGrid_1_3"],
            "storage_units": ["Storage_1"],
        }
        for comp_type, comp_names in manipulated_comps.items():
            comps = getattr(self.edisgo.topology, comp_type + "_df")
            # remove timeseries of single components and check for warning
            for ts_type in ["active_power", "reactive_power"]:
                comp_ts_tmp = getattr(
                    self.edisgo.timeseries, "_".join([comp_type, ts_type])
                )
                setattr(
                    self.edisgo.timeseries,
                    "_".join([comp_type, ts_type]),
                    comp_ts_tmp.drop(columns=comp_names),
                )
                self.edisgo.check_integrity()
                assert (
                    "The following {type} are missing in {ts}: {comps}".format(
                        type=comp_type,
                        ts="_".join([comp_type, ts_type]),
                        comps=str(comp_names).replace(",", ""),
                    )
                    in caplog.text
                )
                setattr(
                    self.edisgo.timeseries, "_".join([comp_type, ts_type]), comp_ts_tmp
                )
                caplog.clear()
            # remove topology entries for single components and check for warning
            setattr(self.edisgo.topology, comp_type + "_df", comps.drop(comp_names))
            self.edisgo.check_integrity()
            for ts_type in ["active_power", "reactive_power"]:
                assert (
                    "The following {type} have entries in {type}_{ts_type}, but not "
                    "in {top}: {comps}".format(
                        type=comp_type,
                        top=comp_type + "_df",
                        comps=str(comp_names).replace(",", ""),
                        ts_type=ts_type,
                    )
                    in caplog.text
                )
            caplog.clear()
            setattr(self.edisgo.topology, comp_type + "_df", comps)
            # set values higher than nominal power for single components and check for
            # warning
            comp_ts_tmp = getattr(
                self.edisgo.timeseries, "_".join([comp_type, "active_power"])
            )
            comp_ts_tmp_adapted = comp_ts_tmp.copy()
            comp_ts_tmp_adapted.loc[index[2], comp_names] = 100
            setattr(
                self.edisgo.timeseries,
                "_".join([comp_type, "active_power"]),
                comp_ts_tmp_adapted,
            )
            self.edisgo.check_integrity()
            if comp_type in ["generators", "storage_units"]:
                attr = "p_nom"
            else:
                attr = "p_set"
            assert (
                "Values of active power in the timeseries object exceed {} for "
                "the following {}: {}".format(
                    attr, comp_type, str(comp_names).replace(",", "")
                )
                in caplog.text
            )
            setattr(
                self.edisgo.timeseries,
                "_".join([comp_type, "active_power"]),
                comp_ts_tmp,
            )
            caplog.clear()


class TestEDisGoFunc:
    def test_import_edisgo_from_files(self):
        edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_obj.set_time_series_worst_case_analysis()
        edisgo_obj.analyze()
        save_dir = os.path.join(os.getcwd(), "edisgo_network")

        # add heat pump and electromobility dummy data
        edisgo_obj.heat_pump.cop = pd.DataFrame(
            data={
                "hp1": [5.0, 6.0, 5.0, 6.0],
                "hp2": [7.0, 8.0, 7.0, 8.0],
            },
            index=edisgo_obj.timeseries.timeindex,
        )
        edisgo_obj.electromobility.charging_processes_df = pd.DataFrame(
            data={
                "ags": [5.0, 6.0],
                "car_id": [7.0, 8.0],
            },
            index=[0, 1],
        )
        flex_bands = {
            "upper_energy": pd.DataFrame(
                {"cp_1": [1, 2]}, index=edisgo_obj.timeseries.timeindex[0:2]
            ),
            "upper_power": pd.DataFrame(
                {"cp_1": [1, 2]}, index=edisgo_obj.timeseries.timeindex[0:2]
            ),
        }
        edisgo_obj.electromobility.flexibility_bands = flex_bands

        # ######################## test with default ########################
        edisgo_obj.save(
            save_dir, save_results=False, save_electromobility=True, save_heatpump=True
        )

        edisgo_obj_loaded = import_edisgo_from_files(save_dir)

        # check topology
        assert_frame_equal(
            edisgo_obj_loaded.topology.loads_df, edisgo_obj.topology.loads_df
        )
        # check time series
        assert edisgo_obj_loaded.timeseries.timeindex.empty
        # check configs
        assert edisgo_obj_loaded.config._data == edisgo_obj.config._data
        # check results
        assert edisgo_obj_loaded.results.i_res.empty

        # ############ test with loading electromobility and heat pump data ###########

        edisgo_obj_loaded = import_edisgo_from_files(
            save_dir,
            import_electromobility=True,
            import_heat_pump=True,
        )

        # check electromobility
        assert_frame_equal(
            edisgo_obj_loaded.electromobility.charging_processes_df,
            edisgo_obj.electromobility.charging_processes_df,
        )
        # check heat pump
        assert_frame_equal(
            edisgo_obj_loaded.heat_pump.cop_df, edisgo_obj.heat_pump.cop_df
        )

        # delete directory
        shutil.rmtree(save_dir)

        # ########### test with loading time series, results, emob from zip ###########
        edisgo_obj.save(save_dir, archive=True, save_electromobility=True)
        zip_file = f"{save_dir}.zip"
        edisgo_obj_loaded = import_edisgo_from_files(
            zip_file,
            import_results=True,
            import_timeseries=True,
            import_electromobility=True,
            from_zip_archive=True,
        )

        # check topology
        assert_frame_equal(
            edisgo_obj_loaded.topology.loads_df, edisgo_obj.topology.loads_df
        )
        # check time series
        assert_frame_equal(
            edisgo_obj_loaded.timeseries.loads_active_power,
            edisgo_obj.timeseries.loads_active_power,
            check_freq=False,
        )
        # check configs
        assert edisgo_obj_loaded.config._data == edisgo_obj.config._data
        # check results
        assert_frame_equal(
            edisgo_obj_loaded.results.i_res, edisgo_obj.results.i_res, check_freq=False
        )
        # check electromobility
        assert_frame_equal(
            edisgo_obj_loaded.electromobility.flexibility_bands["upper_energy"],
            edisgo_obj.electromobility.flexibility_bands["upper_energy"],
            check_freq=False,
        )
        assert_frame_equal(
            edisgo_obj_loaded.electromobility.charging_processes_df,
            edisgo_obj.electromobility.charging_processes_df,
        )

        # delete zip file
        os.remove(zip_file)
