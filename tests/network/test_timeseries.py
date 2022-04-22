from math import acos, tan

import pandas as pd
import pytest

from pandas.util.testing import assert_series_equal

from edisgo import EDisGo
from edisgo.tools.tools import assign_voltage_level_to_component


class TestTimeSeries:
    @pytest.yield_fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def test_set_active_power_manual(self):

        # create dummy time series
        index_2 = pd.date_range("1/1/2018", periods=2, freq="H")
        index_3 = pd.date_range("1/1/2018", periods=3, freq="H")
        dummy_ts_1 = pd.Series([1.4, 2.3], index=index_2)
        dummy_ts_2 = pd.Series([1.4, 2.3, 1.5], index=index_3)
        # set TimeSeries timeindex
        self.edisgo.timeseries.timeindex = index_2

        # test only existing components without prior time series being set
        self.edisgo.timeseries.set_active_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame({"GeneratorFluctuating_8": dummy_ts_1}),
            ts_loads=pd.DataFrame(
                {
                    "Load_residential_LVGrid_8_6": dummy_ts_2,
                    "Load_residential_LVGrid_7_2": dummy_ts_2,
                }
            ),
            ts_storage_units=pd.DataFrame({"Storage_1": dummy_ts_2}),
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries.generators_active_power.loc[
                :, "GeneratorFluctuating_8"
            ]
            == dummy_ts_1
        ).all()
        assert self.edisgo.timeseries.loads_active_power.shape == (2, 2)
        assert (
            self.edisgo.timeseries._loads_active_power.loc[
                :, "Load_residential_LVGrid_8_6"
            ]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Load_residential_LVGrid_7_2"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.storage_units_active_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries._storage_units_active_power.loc[:, "Storage_1"]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.storage_units_active_power.loc[:, "Storage_1"]
            == dummy_ts_2.loc[index_2]
        ).all()

        # test overwriting and adding time series
        self.edisgo.timeseries.set_active_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame(
                {
                    "GeneratorFluctuating_8": dummy_ts_2,
                    "GeneratorFluctuating_17": dummy_ts_2,
                }
            ),
            ts_loads=pd.DataFrame(
                {
                    "Load_residential_LVGrid_8_6": dummy_ts_1,
                    "Load_residential_LVGrid_1_4": dummy_ts_1,
                }
            ),
            ts_storage_units=pd.DataFrame({"Storage_1": dummy_ts_1}),
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (2, 2)
        assert (
            self.edisgo.timeseries.generators_active_power.loc[
                :, "GeneratorFluctuating_8"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert (
            self.edisgo.timeseries._generators_active_power.loc[
                :, "GeneratorFluctuating_17"
            ]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.generators_active_power.loc[
                :, "GeneratorFluctuating_17"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.loads_active_power.shape == (2, 3)
        assert (
            self.edisgo.timeseries._loads_active_power.loc[
                :, "Load_residential_LVGrid_8_6"
            ]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries._loads_active_power.loc[
                :, "Load_residential_LVGrid_1_4"
            ]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Load_residential_LVGrid_7_2"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.storage_units_active_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries._storage_units_active_power.loc[:, "Storage_1"]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries.storage_units_active_power.loc[:, "Storage_1"]
            == dummy_ts_1
        ).all()

        # test non-existent components
        self.edisgo.timeseries.set_active_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame(
                {"Dummy_gen_1": dummy_ts_2, "GeneratorFluctuating_27": dummy_ts_2}
            ),
            ts_loads=pd.DataFrame(
                {"Dummy_load_1": dummy_ts_1, "Load_agricultural_LVGrid_1_3": dummy_ts_1}
            ),
            ts_storage_units=pd.DataFrame({"Dummy_storage_1": dummy_ts_1}),
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (2, 3)
        assert (
            "Dummy_gen_1" not in self.edisgo.timeseries.generators_active_power.columns
        )
        assert (
            self.edisgo.timeseries._generators_active_power.loc[
                :, "GeneratorFluctuating_27"
            ]
            == dummy_ts_2
        ).all()
        assert self.edisgo.timeseries.loads_active_power.shape == (2, 4)
        assert "Dummy_load_1" not in self.edisgo.timeseries.loads_active_power.columns
        assert (
            self.edisgo.timeseries.loads_active_power.loc[
                :, "Load_agricultural_LVGrid_1_3"
            ]
            == dummy_ts_1
        ).all()
        assert self.edisgo.timeseries.storage_units_active_power.shape == (2, 1)
        assert (
            "Dummy_storage_1"
            not in self.edisgo.timeseries.storage_units_active_power.columns
        )

    def test_set_reactive_power_manual(self):

        # create dummy time series
        index_2 = pd.date_range("1/1/2018", periods=2, freq="H")
        index_3 = pd.date_range("1/1/2018", periods=3, freq="H")
        dummy_ts_1 = pd.Series([1.4, 2.3], index=index_2)
        dummy_ts_2 = pd.Series([1.4, 2.3, 1.5], index=index_3)
        # set TimeSeries timeindex
        self.edisgo.timeseries.timeindex = index_2

        # test only existing components without prior time series being set
        self.edisgo.timeseries.set_reactive_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame({"GeneratorFluctuating_8": dummy_ts_1}),
            ts_loads=pd.DataFrame(
                {
                    "Load_residential_LVGrid_8_6": dummy_ts_2,
                    "Load_residential_LVGrid_7_2": dummy_ts_2,
                }
            ),
            ts_storage_units=pd.DataFrame({"Storage_1": dummy_ts_2}),
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries.generators_reactive_power.loc[
                :, "GeneratorFluctuating_8"
            ]
            == dummy_ts_1
        ).all()
        assert self.edisgo.timeseries.loads_reactive_power.shape == (2, 2)
        assert (
            self.edisgo.timeseries._loads_reactive_power.loc[
                :, "Load_residential_LVGrid_8_6"
            ]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[
                :, "Load_residential_LVGrid_7_2"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries._storage_units_reactive_power.loc[:, "Storage_1"]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, "Storage_1"]
            == dummy_ts_2.loc[index_2]
        ).all()

        # test overwriting and adding time series
        self.edisgo.timeseries.set_reactive_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame(
                {
                    "GeneratorFluctuating_8": dummy_ts_2,
                    "GeneratorFluctuating_17": dummy_ts_2,
                }
            ),
            ts_loads=pd.DataFrame(
                {
                    "Load_residential_LVGrid_8_6": dummy_ts_1,
                    "Load_residential_LVGrid_1_4": dummy_ts_1,
                }
            ),
            ts_storage_units=pd.DataFrame({"Storage_1": dummy_ts_1}),
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 2)
        assert (
            self.edisgo.timeseries.generators_reactive_power.loc[
                :, "GeneratorFluctuating_8"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert (
            self.edisgo.timeseries._generators_reactive_power.loc[
                :, "GeneratorFluctuating_17"
            ]
            == dummy_ts_2
        ).all()
        assert (
            self.edisgo.timeseries.generators_reactive_power.loc[
                :, "GeneratorFluctuating_17"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.loads_reactive_power.shape == (2, 3)
        assert (
            self.edisgo.timeseries._loads_reactive_power.loc[
                :, "Load_residential_LVGrid_8_6"
            ]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries._loads_reactive_power.loc[
                :, "Load_residential_LVGrid_1_4"
            ]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[
                :, "Load_residential_LVGrid_7_2"
            ]
            == dummy_ts_2.loc[index_2]
        ).all()
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (2, 1)
        assert (
            self.edisgo.timeseries._storage_units_reactive_power.loc[:, "Storage_1"]
            == dummy_ts_1
        ).all()
        assert (
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, "Storage_1"]
            == dummy_ts_1
        ).all()

        # test non-existent components
        self.edisgo.timeseries.set_reactive_power_manual(
            edisgo_object=self.edisgo,
            ts_generators=pd.DataFrame(
                {"Dummy_gen_1": dummy_ts_2, "GeneratorFluctuating_27": dummy_ts_2}
            ),
            ts_loads=pd.DataFrame(
                {"Dummy_load_1": dummy_ts_1, "Load_agricultural_LVGrid_1_3": dummy_ts_1}
            ),
            ts_storage_units=pd.DataFrame({"Dummy_storage_1": dummy_ts_1}),
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 3)
        assert (
            "Dummy_gen_1"
            not in self.edisgo.timeseries.generators_reactive_power.columns
        )
        assert (
            self.edisgo.timeseries._generators_reactive_power.loc[
                :, "GeneratorFluctuating_27"
            ]
            == dummy_ts_2
        ).all()
        assert self.edisgo.timeseries.loads_reactive_power.shape == (2, 4)
        assert "Dummy_load_1" not in self.edisgo.timeseries.loads_reactive_power.columns
        assert (
            self.edisgo.timeseries.loads_reactive_power.loc[
                :, "Load_agricultural_LVGrid_1_3"
            ]
            == dummy_ts_1
        ).all()
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (2, 1)
        assert (
            "Dummy_storage_1"
            not in self.edisgo.timeseries.storage_units_reactive_power.columns
        )

    def test_worst_case_generators(self):

        # ######### check both feed-in and load case
        df = assign_voltage_level_to_component(
            self.edisgo.topology.generators_df, self.edisgo.topology.buses_df
        )
        p_ts, q_ts = self.edisgo.timeseries._worst_case_generators(
            cases=["feed-in_case", "load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        number_of_cols = len(df.index)
        assert p_ts.shape == (4, number_of_cols)
        assert q_ts.shape == (4, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv", "load_case_mv", "load_case_lv"]
        comp = "Generator_1"  # gas, mv
        p_nom = 0.775
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom, 0.0, 0.0],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "GeneratorFluctuating_2"  # wind, mv
        p_nom = 2.3
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom, 0.0, 0.0],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "GeneratorFluctuating_3"  # solar, mv
        p_nom = 2.67
        exp = pd.Series(
            data=[0.85 * p_nom, 0.85 * p_nom, 0.0, 0.0],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "GeneratorFluctuating_20"  # solar, lv
        p_nom = 0.005
        exp = pd.Series(
            data=[0.85 * p_nom, 0.85 * p_nom, 0.0, 0.0],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.95))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["Generator_1", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only feed-in case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_generators(
            cases=["feed-in_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv"]
        comp = "GeneratorFluctuating_2"  # wind, mv
        p_nom = 2.3
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at[
                "GeneratorFluctuating_2", "type"
            ]
            == "fixed_cosphi"
        )

        # ########### test for only load case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_generators(
            cases=["load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["load_case_mv", "load_case_lv"]
        comp = "GeneratorFluctuating_20"  # solar, lv
        exp = pd.Series(
            data=[0.0, 0.0],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.95))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at[
                "GeneratorFluctuating_20", "type"
            ]
            == "fixed_cosphi"
        )

        # ########## test error raising in case of missing load/generator parameter

        comp = "GeneratorFluctuating_14"
        df.at[comp, "type"] = None
        with pytest.raises(AttributeError):
            self.edisgo.timeseries._worst_case_generators(
                cases=["load_case"], df=df, configs=self.edisgo.config
            )

    def test_worst_case_conventional_load(self):

        # connect one load to MV
        self.edisgo.topology._loads_df.at[
            "Load_agricultural_LVGrid_1_1", "bus"
        ] = "Bus_BranchTee_MVGrid_1_2"

        # ######### check both feed-in and load case
        df = assign_voltage_level_to_component(
            self.edisgo.topology.loads_df, self.edisgo.topology.buses_df
        )
        p_ts, q_ts = self.edisgo.timeseries._worst_case_conventional_load(
            cases=["feed-in_case", "load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        number_of_cols = len(df.index)
        assert p_ts.shape == (4, number_of_cols)
        assert q_ts.shape == (4, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv", "load_case_mv", "load_case_lv"]
        comp = "Load_agricultural_LVGrid_1_1"  # mv
        p_nom = 0.0523
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "Load_agricultural_LVGrid_8_1"  # lv
        p_nom = 0.0478
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at[
                "Load_agricultural_LVGrid_8_1", "type"
            ]
            == "fixed_cosphi"
        )

        # ########### test for only feed-in case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_conventional_load(
            cases=["feed-in_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv"]
        comp = "Load_agricultural_LVGrid_8_1"  # lv
        p_nom = 0.0478
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at[
                "Load_agricultural_LVGrid_8_1", "type"
            ]
            == "fixed_cosphi"
        )

        # ########### test for only load case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_conventional_load(
            cases=["load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["load_case_mv", "load_case_lv"]
        comp = "Load_agricultural_LVGrid_1_1"  # mv
        p_nom = 0.0523
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at[
                "Load_agricultural_LVGrid_1_1", "type"
            ]
            == "fixed_cosphi"
        )

        # ########## test error raising in case of missing load/generator parameter

        comp = "Load_agricultural_LVGrid_1_1"
        df.at[comp, "voltage_level"] = None
        with pytest.raises(AttributeError):
            self.edisgo.timeseries._worst_case_conventional_load(
                cases=["load_case"], df=df, configs=self.edisgo.config
            )

    def test_worst_case_charging_points(self):
        # add charging points to MV and LV
        df_cp = pd.DataFrame(
            {
                "bus": [
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_LVGrid_1_5",
                    "Bus_BranchTee_LVGrid_1_5",
                ],
                "p_nom": [0.1, 0.2, 0.3, 0.4],
                "type": [
                    "charging_point",
                    "charging_point",
                    "charging_point",
                    "charging_point",
                ],
                "sector": ["hpc", "public", "home", "work"],
            },
            index=["CP1", "CP2", "CP3", "CP4"],
        )

        # ######### check both feed-in and load case
        df = assign_voltage_level_to_component(df_cp, self.edisgo.topology.buses_df)
        p_ts, q_ts = self.edisgo.timeseries._worst_case_charging_points(
            cases=["feed-in_case", "load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        number_of_cols = len(df.index)
        assert p_ts.shape == (4, number_of_cols)
        assert q_ts.shape == (4, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv", "load_case_mv", "load_case_lv"]
        comp = "CP1"  # mv, hpc
        p_nom = 0.1
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP2"  # mv, public
        p_nom = 0.2
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP3"  # lv, home
        p_nom = 0.3
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 0.3 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP4"  # lv, work
        p_nom = 0.4
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 0.3 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["CP4", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only feed-in case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_charging_points(
            cases=["feed-in_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv"]
        comp = "CP3"  # lv, home
        p_nom = 0.3
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["CP3", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only load case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_charging_points(
            cases=["load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["load_case_mv", "load_case_lv"]
        comp = "CP2"  # mv, public
        p_nom = 0.2
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["CP2", "type"]
            == "fixed_cosphi"
        )

        # ########## test error raising in case of missing load/generator parameter

        comp = "CP2"
        df.at[comp, "voltage_level"] = None
        with pytest.raises(AttributeError):
            self.edisgo.timeseries._worst_case_charging_points(
                cases=["load_case"], df=df, configs=self.edisgo.config
            )

    def test_worst_case_heat_pumps(self):
        # add heat pumps to MV and LV
        df_hp = pd.DataFrame(
            {
                "bus": ["Bus_BranchTee_MVGrid_1_2", "Bus_BranchTee_LVGrid_1_5"],
                "p_nom": [0.1, 0.2],
                "type": ["heat_pump", "heat_pump"],
            },
            index=["HP1", "HP2"],
        )

        # ######### check both feed-in and load case
        df = assign_voltage_level_to_component(df_hp, self.edisgo.topology.buses_df)
        p_ts, q_ts = self.edisgo.timeseries._worst_case_heat_pumps(
            cases=["feed-in_case", "load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        number_of_cols = len(df.index)
        assert p_ts.shape == (4, number_of_cols)
        assert q_ts.shape == (4, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv", "load_case_mv", "load_case_lv"]
        comp = "HP1"  # mv
        p_nom = 0.1
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 0.9 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.98))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "HP2"  # lv
        p_nom = 0.2
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom, 0.9 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.98))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["HP1", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only feed-in case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_heat_pumps(
            cases=["feed-in_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv"]
        comp = "HP2"  # lv
        p_nom = 0.2
        exp = pd.Series(
            data=[0.15 * p_nom, 0.1 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.98))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["HP2", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only load case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_heat_pumps(
            cases=["load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["load_case_mv", "load_case_lv"]
        comp = "HP1"  # mv
        p_nom = 0.1
        exp = pd.Series(
            data=[0.9 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.98))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["HP1", "type"]
            == "fixed_cosphi"
        )

        # ########## test error raising in case of missing load/generator parameter

        comp = "HP1"
        df.at[comp, "voltage_level"] = None
        with pytest.raises(AttributeError):
            self.edisgo.timeseries._worst_case_heat_pumps(
                cases=["load_case"], df=df, configs=self.edisgo.config
            )

    def test_worst_case_storage_units(self):

        # ######### check both feed-in and load case
        df = assign_voltage_level_to_component(
            self.edisgo.topology.storage_units_df, self.edisgo.topology.buses_df
        )
        p_ts, q_ts = self.edisgo.timeseries._worst_case_storage_units(
            cases=["feed-in_case", "load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        number_of_cols = len(df.index)
        assert p_ts.shape == (4, number_of_cols)
        assert q_ts.shape == (4, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv", "load_case_mv", "load_case_lv"]
        comp = "Storage_1"
        p_nom = 0.4
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom, -1.0 * p_nom, -1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["Storage_1", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only feed-in case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_storage_units(
            cases=["feed-in_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["feed-in_case_mv", "feed-in_case_lv"]
        comp = "Storage_1"
        p_nom = 0.4
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["Storage_1", "type"]
            == "fixed_cosphi"
        )

        # ########### test for only load case
        p_ts, q_ts = self.edisgo.timeseries._worst_case_storage_units(
            cases=["load_case"], df=df, configs=self.edisgo.config
        )

        # check shape
        assert p_ts.shape == (2, number_of_cols)
        assert q_ts.shape == (2, number_of_cols)

        # check values
        index = ["load_case_mv", "load_case_lv"]
        comp = "Storage_1"
        p_nom = 0.4
        exp = pd.Series(
            data=[-1.0 * p_nom, -1.0 * p_nom],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        # check TimeSeriesRaw
        assert len(self.edisgo.timeseries.time_series_raw.q_control) == len(df)
        assert (
            self.edisgo.timeseries.time_series_raw.q_control.at["Storage_1", "type"]
            == "fixed_cosphi"
        )

        # ########## test error raising in case of missing load/generator parameter

        comp = "Storage_1"
        df.at[comp, "voltage_level"] = None
        with pytest.raises(AttributeError):
            self.edisgo.timeseries._worst_case_storage_units(
                cases=["load_case"], df=df, configs=self.edisgo.config
            )

    def test_to_csv(self):
        # ToDo implement
        pass
        # timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        # timeseries_obj = timeseries.TimeSeries(timeindex=timeindex)
        #
        # # create dummy time series
        # loads_active_power = pd.DataFrame(
        #     {"load1": [1.4, 2.3], "load2": [2.4, 1.3]}, index=timeindex
        # )
        # timeseries_obj.loads_active_power = loads_active_power
        # generators_reactive_power = pd.DataFrame(
        #     {"gen1": [1.4, 2.3], "gen2": [2.4, 1.3]}, index=timeindex
        # )
        # timeseries_obj.generators_reactive_power = generators_reactive_power
        #
        # # test with default values
        # dir = os.path.join(os.getcwd(), "timeseries_csv")
        # timeseries_obj.to_csv(dir)
        #
        # files_in_timeseries_dir = os.listdir(dir)
        # assert len(files_in_timeseries_dir) == 2
        # assert "loads_active_power.csv" in files_in_timeseries_dir
        # assert "generators_reactive_power.csv" in files_in_timeseries_dir
        #
        # shutil.rmtree(dir)
        #
        # # test with reduce memory True
        # timeseries_obj.to_csv(dir, reduce_memory=True)
        #
        # assert timeseries_obj.loads_active_power.load1.dtype == "float32"
        #
        # shutil.rmtree(dir, ignore_errors=True)

    def test_from_csv(self):
        # ToDo implement
        pass
        # timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        # timeseries_obj = timeseries.TimeSeries(timeindex=timeindex)
        #
        # # create dummy time series
        # loads_active_power = pd.DataFrame(
        #     {"load1": [1.4, 2.3], "load2": [2.4, 1.3]}, index=timeindex
        # )
        # timeseries_obj.loads_active_power = loads_active_power
        # generators_reactive_power = pd.DataFrame(
        #     {"gen1": [1.4, 2.3], "gen2": [2.4, 1.3]}, index=timeindex
        # )
        # timeseries_obj.generators_reactive_power = generators_reactive_power
        #
        # # write to csv
        # dir = os.path.join(os.getcwd(), "timeseries_csv")
        # timeseries_obj.to_csv(dir)
        #
        # # reset TimeSeries
        # timeseries_obj = timeseries.TimeSeries()
        #
        # timeseries_obj.from_csv(dir)
        #
        # pd.testing.assert_frame_equal(
        #     timeseries_obj.loads_active_power,
        #     loads_active_power,
        #     check_freq=False,
        # )
        # pd.testing.assert_frame_equal(
        #     timeseries_obj.generators_reactive_power,
        #     generators_reactive_power,
        #     check_freq=False,
        # )
        #
        # shutil.rmtree(dir)


class TestTimeSeriesRaw:
    def test_reduce_memory(self):
        # ToDo implement
        pass

    def test_to_csv(self):
        # ToDo implement
        pass

    def test_from_csv(self):
        # ToDo implement
        pass


class TestTimeSeriesHelperFunctions:
    def test_drop_component_time_series(self):
        # ToDo implement
        pass
        # """Test for _drop_existing_timseries_method"""
        # storage_1 = self.topology.add_storage_unit("Bus_MVStation_1", 0.3)
        # timeindex = pd.date_range("1/1/1970", periods=2, freq="H")
        # timeseries.get_component_timeseries(edisgo_obj=self, mode="worst-case")
        # # test drop load timeseries
        # assert hasattr(
        #     self.timeseries.loads_active_power, "Load_agricultural_LVGrid_1_1"
        # )
        # assert hasattr(
        #     self.timeseries.loads_reactive_power,
        #     "Load_agricultural_LVGrid_1_1",
        # )
        # timeseries._drop_existing_component_timeseries(
        #     self, "loads", ["Load_agricultural_LVGrid_1_1"]
        # )
        # with pytest.raises(KeyError):
        #     self.timeseries.loads_active_power.loc[
        #         timeindex, "Load_agricultural_LVGrid_1_1"
        #     ]
        # with pytest.raises(KeyError):
        #     self.timeseries.loads_reactive_power.loc[
        #         timeindex, "Load_agricultural_LVGrid_1_1"
        #     ]
        # # test drop generators timeseries
        # assert hasattr(
        #     self.timeseries.generators_active_power, "GeneratorFluctuating_7"
        # )
        # assert hasattr(
        #     self.timeseries.generators_reactive_power, "GeneratorFluctuating_7"
        # )
        # timeseries._drop_existing_component_timeseries(
        #     self, "generators", "GeneratorFluctuating_7"
        # )
        # with pytest.raises(KeyError):
        #     self.timeseries.generators_active_power.loc[
        #         timeindex, "GeneratorFluctuating_7"
        #     ]
        # with pytest.raises(KeyError):
        #     self.timeseries.generators_reactive_power.loc[
        #         timeindex, "GeneratorFluctuating_7"
        #     ]
        # # test drop storage units timeseries
        # assert hasattr(self.timeseries.storage_units_active_power, storage_1)
        # assert hasattr(self.timeseries.storage_units_reactive_power, storage_1)
        # timeseries._drop_existing_component_timeseries(
        # self, "storage_units", storage_1
        # )
        # with pytest.raises(KeyError):
        #     self.timeseries.storage_units_active_power.loc[timeindex, storage_1]
        # with pytest.raises(KeyError):
        #     self.timeseries.storage_units_reactive_power.loc[timeindex, storage_1]
        # self.topology.remove_storage_unit(storage_1)

    def test_add_component_time_series(self):
        # ToDo implement
        pass

    def test_check_if_components_exist(self):
        # ToDo implement
        pass
