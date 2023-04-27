import copy
import logging
import os
import shutil

from math import acos, tan

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from edisgo import EDisGo
from edisgo.network import timeseries
from edisgo.tools.tools import assign_voltage_level_to_component


class TestTimeSeries:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def test_timeseries_getters(self, caplog):
        index_2 = pd.date_range("1/1/2018", periods=2, freq="H")
        index_3 = pd.date_range("1/1/2018", periods=3, freq="H")
        timeseries = pd.DataFrame(index=index_2, columns=["Comp_1"], data=[1.3, 2])
        self.edisgo.timeseries.timeindex = index_3
        for attribute in self.edisgo.timeseries._attributes:
            assert_frame_equal(
                getattr(self.edisgo.timeseries, attribute), pd.DataFrame(index=index_3)
            )
            setattr(self.edisgo.timeseries, attribute, timeseries)
            with caplog.at_level(logging.WARNING):
                assert_frame_equal(
                    getattr(self.edisgo.timeseries, attribute),
                    pd.DataFrame(index=index_3),
                )
            assert (
                "Timeindex and {} have deviating indices. "
                "Empty dataframe will be returned.".format(attribute) in caplog.text
            )

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

    def test_set_worst_case(self):
        # test - check if right functions are called for all components

        # change load types to have charging point, heat pump and load without set
        # type in the network
        self.edisgo.topology._loads_df.loc[
            "Load_residential_LVGrid_1_4", ["type", "sector"]
        ] = ("charging_point", "hpc")
        self.edisgo.topology._loads_df.at[
            "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1", "type"
        ] = "heat_pump"
        self.edisgo.topology._loads_df.at["Load_agricultural_LVGrid_8_1", "type"] = None

        self.edisgo.timeseries.set_worst_case(
            self.edisgo, cases=["feed-in_case", "load_case"]
        )

        timeindex = pd.date_range("1/1/1970", periods=4, freq="H")
        # check generator
        comp = "Generator_1"  # gas, mv
        p_nom = 0.775
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom, 0.0, 0.0],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.generators_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.edisgo.timeseries.generators_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
        # check conventional load
        comp = "Load_agricultural_LVGrid_1_1"  # lv
        p_set = 0.0523
        exp = pd.Series(
            data=[0.15 * p_set, 0.1 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.loads_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = tan(acos(0.95))
        assert_series_equal(
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
        # check charging point
        comp = "Load_residential_LVGrid_1_4"  # lv, hpc
        p_set = 0.001397
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.loads_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = tan(acos(1.0))
        assert_series_equal(
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
        # check heat pump
        comp = "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"  # mv
        p_set = 0.31
        exp = pd.Series(
            data=[0.0 * p_set, 0.0 * p_set, 0.8 * p_set, 1.0 * p_set],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.loads_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = tan(acos(1.0))
        assert_series_equal(
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
        # check load without type specification
        comp = "Load_agricultural_LVGrid_8_1"  # lv
        p_set = 0.0478
        exp = pd.Series(
            data=[0.15 * p_set, 0.1 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.loads_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = tan(acos(0.95))
        assert_series_equal(
            self.edisgo.timeseries.loads_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
        # check storage
        comp = "Storage_1"
        p_nom = 0.4
        exp = pd.Series(
            data=[1.0 * p_nom, 1.0 * p_nom, -1.0 * p_nom, -1.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.storage_units_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )

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

        # #############################################################################
        # test with components that do not exist and setting only one case
        self.edisgo.timeseries.set_worst_case(
            self.edisgo,
            cases=["load_case"],
            generators_names=["genX", "GeneratorFluctuating_8"],
            loads_names=[],
            storage_units_names=[],
        )

        comp = "GeneratorFluctuating_8"
        exp = pd.Series(
            data=[np.nan, np.nan, 0.0, 0.0],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(
            self.edisgo.timeseries.generators_active_power.loc[:, comp],
            exp,
            check_dtype=False,
        )
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.edisgo.timeseries.generators_reactive_power.loc[:, comp],
            exp * pf,
            check_dtype=False,
        )
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

        # #############################################################################
        # test reset of time series - set other time series before and only set
        # worst case time series for other components
        timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        self.edisgo.timeseries.timeindex = timeindex
        self.edisgo.timeseries._generators_active_power = pd.DataFrame(
            {"Generator_1": [1.4, 2.3]}, index=timeindex
        )
        self.edisgo.timeseries.set_worst_case(
            self.edisgo,
            cases=["load_case"],
            generators_names=["GeneratorFluctuating_8"],
        )
        assert (
            "GeneratorFluctuating_8"
            in self.edisgo.timeseries.generators_active_power.columns
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (2, 1)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (2, 1)

        # #############################################################################
        # test setting other case now to see if time index is set correctly
        self.edisgo.timeseries.set_worst_case(
            self.edisgo,
            cases=["feed-in_case"],
            generators_names=["GeneratorFluctuating_8"],
        )
        assert self.edisgo.timeseries.generators_active_power.shape == (4, 1)
        assert self.edisgo.timeseries.generators_reactive_power.shape == (4, 1)
        exp = pd.Series(
            data=pd.date_range("1/1/1970", periods=4, freq="H"),
            index=[
                "load_case_mv",
                "load_case_lv",
                "feed-in_case_mv",
                "feed-in_case_lv",
            ],
        )
        assert_series_equal(
            self.edisgo.timeseries.timeindex_worst_cases, exp, check_dtype=False
        )
        assert (
            self.edisgo.timeseries.timeindex
            == self.edisgo.timeseries.timeindex_worst_cases.values
        ).all()

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
        p_set = 0.0523
        exp = pd.Series(
            data=[0.15 * p_set, 0.1 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "Load_agricultural_LVGrid_8_1"  # lv
        p_set = 0.0478
        exp = pd.Series(
            data=[0.15 * p_set, 0.1 * p_set, 1.0 * p_set, 1.0 * p_set],
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
        p_set = 0.0478
        exp = pd.Series(
            data=[0.15 * p_set, 0.1 * p_set],
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
        p_set = 0.0523
        exp = pd.Series(
            data=[1.0 * p_set, 1.0 * p_set],
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
                "p_set": [0.1, 0.2, 0.3, 0.4],
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
        p_set = 0.1
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP2"  # mv, public
        p_set = 0.2
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set, 1.0 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP3"  # lv, home
        p_set = 0.3
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set, 0.2 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "CP4"  # lv, work
        p_set = 0.4
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set, 0.2 * p_set, 1.0 * p_set],
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
        p_set = 0.3
        exp = pd.Series(
            data=[0.15 * p_set, 0.0 * p_set],
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
        p_set = 0.2
        exp = pd.Series(
            data=[1.0 * p_set, 1.0 * p_set],
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
                "p_set": [0.1, 0.2],
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
        p_set = 0.1
        exp = pd.Series(
            data=[0.0 * p_set, 0.0 * p_set, 0.8 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
        assert_series_equal(q_ts.loc[:, comp], exp * pf, check_dtype=False)

        comp = "HP2"  # lv
        p_set = 0.2
        exp = pd.Series(
            data=[0.0 * p_set, 0.0 * p_set, 0.8 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
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
        p_set = 0.2
        exp = pd.Series(
            data=[0.0 * p_set, 0.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
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
        p_set = 0.1
        exp = pd.Series(
            data=[0.8 * p_set, 1.0 * p_set],
            name=comp,
            index=index,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        pf = tan(acos(1.0))
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

    @pytest.mark.slow
    def test_predefined_fluctuating_generators_by_technology(self):
        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        self.edisgo.timeseries.timeindex = timeindex

        # ############# oedb, all generators (default)
        self.edisgo.timeseries.predefined_fluctuating_generators_by_technology(
            self.edisgo, "oedb"
        )

        # check shape
        fluctuating_gens = self.edisgo.topology.generators_df[
            self.edisgo.topology.generators_df.type.isin(["wind", "solar"])
        ]
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, len(fluctuating_gens))
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                fluctuating_generators_active_power_by_technology.shape
                == (2, 8)
        )
        # fmt: on

        # check values
        comp = "GeneratorFluctuating_2"  # wind, w_id = 1122074
        p_nom = 2.3
        exp = pd.Series(
            data=[0.0 * p_nom, 0.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)
        comp = "GeneratorFluctuating_8"  # wind, w_id = 1122075
        p_nom = 3.0
        exp = pd.Series(
            data=[0.0029929 * p_nom, 0.009521 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)
        comp = "GeneratorFluctuating_25"  # solar, w_id = 1122075
        p_nom = 0.006
        exp = pd.Series(
            data=[0.07824 * p_nom, 0.11216 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)

        # ############# own settings (without weather cell ID), all generators
        gens_p = pd.DataFrame(
            data={
                "wind": [1, 2],
                "solar": [3, 4],
            },
            index=timeindex,
        )
        self.edisgo.timeseries.predefined_fluctuating_generators_by_technology(
            self.edisgo, gens_p
        )

        # check shape
        fluctuating_gens = self.edisgo.topology.generators_df[
            self.edisgo.topology.generators_df.type.isin(["wind", "solar"])
        ]
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, len(fluctuating_gens))
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                fluctuating_generators_active_power_by_technology.shape
                == (2, 10)
        )
        # fmt: on

        # check values
        comp = "GeneratorFluctuating_2"  # wind
        p_nom = 2.3
        exp = pd.Series(
            data=[1.0 * p_nom, 2.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        comp = "GeneratorFluctuating_20"  # solar
        p_nom = 0.005
        exp = pd.Series(
            data=[3.0 * p_nom, 4.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)

        # ############# own settings (with weather cell ID), selected generators
        self.edisgo.timeseries.timeindex = timeindex
        gens_p = pd.DataFrame(
            data={
                ("wind", 1122074): [5, 6],
                ("solar", 1122075): [7, 8],
            },
            index=timeindex,
        )
        self.edisgo.timeseries.predefined_fluctuating_generators_by_technology(
            self.edisgo,
            gens_p,
            generator_names=["GeneratorFluctuating_4", "GeneratorFluctuating_2"],
        )

        # check shape (should be the same as before, as time series are not reset but
        # overwritten)
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, len(fluctuating_gens))
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                fluctuating_generators_active_power_by_technology.shape
                == (2, 10)
        )
        # fmt: on

        # check values (check that values are overwritten)
        comp = "GeneratorFluctuating_2"  # wind
        p_nom = 2.3
        exp = pd.Series(
            data=[5.0 * p_nom, 6.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        comp = "GeneratorFluctuating_4"  # solar
        p_nom = 1.93
        exp = pd.Series(
            data=[7.0 * p_nom, 8.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False)
        # fmt: off
        assert_series_equal(
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology.loc[
                :, ("wind", 1122074)
            ],
            gens_p.loc[:, ("wind", 1122074)],
            check_dtype=False,
        )
        # fmt: on

        # ############# own settings (with weather cell ID), all generators (check, that
        # time series for generators are set for those for which time series are
        # provided)
        self.edisgo.timeseries.reset()
        self.edisgo.timeseries.timeindex = timeindex
        self.edisgo.timeseries.predefined_fluctuating_generators_by_technology(
            self.edisgo, gens_p
        )

        # check shape
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, 22)
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                fluctuating_generators_active_power_by_technology.shape
                == (2, 2)
        )
        # fmt: on

    @pytest.mark.local
    def test_predefined_fluctuating_generators_by_technology_oedb(self):
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        edisgo_object.timeseries.timeindex = timeindex

        # ############# oedb, all generators (default)
        edisgo_object.timeseries.predefined_fluctuating_generators_by_technology(
            edisgo_object, "oedb", engine=pytest.engine
        )

        # check shape
        fluctuating_gens = edisgo_object.topology.generators_df[
            edisgo_object.topology.generators_df.type.isin(["wind", "solar"])
        ]
        p_ts = edisgo_object.timeseries.generators_active_power
        assert p_ts.shape == (2, len(fluctuating_gens))
        # fmt: off
        assert (
                edisgo_object.timeseries.time_series_raw.
                fluctuating_generators_active_power_by_technology.shape
                == (2, 4)
        )
        # fmt: on

        # check values
        # solar, w_id = 11052
        comp = "Generator_mvgd_33535_lvgd_1204030000_pv_rooftop_337"
        p_nom = 0.00441
        exp = pd.Series(
            data=[0.548044 * p_nom, 0.568356 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)
        # solar, w_id = 11051
        comp = "Generator_mvgd_33535_lvgd_1164120002_pv_rooftop_324"
        p_nom = 0.0033
        exp = pd.Series(
            data=[0.505049 * p_nom, 0.555396 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)

    def test_predefined_dispatchable_generators_by_technology(self):
        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        self.edisgo.timeseries.timeindex = timeindex

        # ############# all generators (default), with "other"
        gens_p = pd.DataFrame(
            data={
                "other": [5, 6],
            },
            index=timeindex,
        )

        self.edisgo.timeseries.predefined_dispatchable_generators_by_technology(
            self.edisgo, gens_p
        )

        # check shape
        dispatchable_gens = self.edisgo.topology.generators_df[
            ~self.edisgo.topology.generators_df.type.isin(["wind", "solar"])
        ]
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, len(dispatchable_gens))
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                dispatchable_generators_active_power_by_technology.shape
                == (2, 1)
        )
        # fmt: on

        # check values
        comp = "Generator_1"  # gas
        p_nom = 0.775
        exp = pd.Series(
            data=[5.0 * p_nom, 6.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)

        # ############# all generators (default), with "gas" and "other"
        # overwrite type of generator GeneratorFluctuating_2
        self.edisgo.topology._generators_df.at[
            "GeneratorFluctuating_2", "type"
        ] = "coal"
        gens_p = pd.DataFrame(
            data={
                "other": [5, 6],
                "gas": [7, 8],
            },
            index=timeindex,
        )

        self.edisgo.timeseries.predefined_dispatchable_generators_by_technology(
            self.edisgo, gens_p
        )

        # check shape
        dispatchable_gens = self.edisgo.topology.generators_df[
            ~self.edisgo.topology.generators_df.type.isin(["wind", "solar"])
        ]
        p_ts = self.edisgo.timeseries.generators_active_power
        assert p_ts.shape == (2, len(dispatchable_gens))
        # fmt: off
        assert (
                self.edisgo.timeseries.time_series_raw.
                dispatchable_generators_active_power_by_technology.shape
                == (2, 2)
        )
        # fmt: on

        # check values
        comp = "Generator_1"  # gas
        p_nom = 0.775
        exp = pd.Series(
            data=[7.0 * p_nom, 8.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)
        comp = "GeneratorFluctuating_2"  # coal (other)
        p_nom = 2.3
        exp = pd.Series(
            data=[5.0 * p_nom, 6.0 * p_nom],
            name=comp,
            index=timeindex,
        )
        assert_series_equal(p_ts.loc[:, comp], exp, check_dtype=False, atol=1e-5)
        # fmt: off
        assert_series_equal(
            self.edisgo.timeseries.time_series_raw.
            dispatchable_generators_active_power_by_technology.loc[
                :, "other"
            ],
            gens_p.loc[:, "other"],
            check_dtype=False,
        )
        # fmt: on

    def test_predefined_conventional_loads_by_sector(self, caplog):
        index = pd.date_range("1/1/2018", periods=3, freq="H")
        self.edisgo.timeseries.timeindex = index

        # test assertion error
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, pd.DataFrame()
        )
        assert "The profile you entered is empty. Method is skipped." in caplog.text

        # define expected profiles
        profiles = pd.DataFrame(
            index=index,
            columns=["cts", "residential", "agricultural", "industrial"],
            data=[
                [0.0000597, 0.0000782, 0.0000654, 0.0000992],
                [0.0000526, 0.0000563, 0.0000611, 0.0000992],
                [0.0000459, 0.0000451, 0.0000585, 0.0000992],
            ],
        )

        # test demandlib - single loads
        loads = [
            "Load_agricultural_LVGrid_5_2",
            "Load_agricultural_LVGrid_9_1",
            "Load_residential_LVGrid_9_2",
            "Load_retail_LVGrid_9_14",
            "Load_residential_LVGrid_5_3",
            "Load_industrial_LVGrid_6_1",
            "Load_agricultural_LVGrid_7_1",
        ]
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, "demandlib", load_names=loads
        )
        # fmt: off
        assert self.edisgo.timeseries.time_series_raw.\
            conventional_loads_active_power_by_sector.shape\
               == (3, 4)
        assert_frame_equal(
            self.edisgo.timeseries.time_series_raw.
            conventional_loads_active_power_by_sector,
            profiles,
            atol=1e-7,
        )
        # fmt: on
        assert self.edisgo.timeseries.loads_active_power.shape == (3, 7)
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_agricultural_LVGrid_5_2"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_agricultural_LVGrid_5_2", "annual_consumption"
                ]
                * profiles["agricultural"]
            ).values,
            atol=1e-4,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_residential_LVGrid_5_3"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_residential_LVGrid_5_3", "annual_consumption"
                ]
                * profiles["residential"]
            ).values,
            atol=1e-4,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power["Load_retail_LVGrid_9_14"].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_retail_LVGrid_9_14", "annual_consumption"
                ]
                * profiles["cts"]
            ).values,
            atol=1e-4,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_industrial_LVGrid_6_1"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_industrial_LVGrid_6_1", "annual_consumption"
                ]
                * profiles["industrial"]
            ).values,
            atol=1e-4,
        ).all()
        # test demandlib - all
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, "demandlib"
        )
        # fmt: off
        assert self.edisgo.timeseries.time_series_raw.\
            conventional_loads_active_power_by_sector.shape\
               == (3, 4)
        # fmt: on
        assert self.edisgo.timeseries.loads_active_power.shape == (3, 50)
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_industrial_LVGrid_6_1"
            ].values,
            [0.05752256] * 3,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power.loc[
                index[1], "Load_agricultural_LVGrid_5_2"
            ],
            0.0274958,
        )
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power.loc[
                index, "Load_residential_LVGrid_9_2"
            ].values,
            [0.00038328, 0.00027608, 0.00022101],
        ).all()
        # test assertion error
        with pytest.raises(ValueError) as exc_info:
            self.edisgo.timeseries.predefined_conventional_loads_by_sector(
                self.edisgo, "random"
            )
        assert (
            exc_info.value.args[0]
            == "'ts_loads' must either be a pandas DataFrame or 'demandlib'."
        )
        # test manual - all
        profiles = pd.DataFrame(
            index=index,
            columns=["cts", "residential", "agricultural", "industrial"],
            data=[
                [0.003, 0.02, 0.00, 0.1],
                [0.004, 0.01, 0.10, 0.2],
                [0.002, 0.06, 0.25, 1.0],
            ],
        )
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, profiles
        )
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_agricultural_LVGrid_5_2"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_agricultural_LVGrid_5_2", "annual_consumption"
                ]
                * profiles["agricultural"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_residential_LVGrid_5_3"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_residential_LVGrid_5_3", "annual_consumption"
                ]
                * profiles["residential"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power["Load_retail_LVGrid_9_14"].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_retail_LVGrid_9_14", "annual_consumption"
                ]
                * profiles["cts"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_industrial_LVGrid_6_1"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_industrial_LVGrid_6_1", "annual_consumption"
                ]
                * profiles["industrial"]
            ).values,
        ).all()
        # test manual - single loads
        profiles_new = (
            pd.DataFrame(
                index=index,
                columns=["cts", "residential", "agricultural", "industrial"],
                data=[
                    [0.003, 0.02, 0.00, 0.1],
                    [0.004, 0.01, 0.10, 0.2],
                    [0.002, 0.06, 0.25, 1.0],
                ],
            )
            * 5
        )
        loads = ["Load_industrial_LVGrid_6_1", "Load_residential_LVGrid_5_3"]
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, profiles_new, load_names=loads
        )
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_agricultural_LVGrid_5_2"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_agricultural_LVGrid_5_2", "annual_consumption"
                ]
                * profiles["agricultural"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_residential_LVGrid_5_3"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_residential_LVGrid_5_3", "annual_consumption"
                ]
                * profiles_new["residential"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power["Load_retail_LVGrid_9_14"].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_retail_LVGrid_9_14", "annual_consumption"
                ]
                * profiles["cts"]
            ).values,
        ).all()
        assert np.isclose(
            self.edisgo.timeseries.loads_active_power[
                "Load_industrial_LVGrid_6_1"
            ].values,
            (
                self.edisgo.topology.loads_df.loc[
                    "Load_industrial_LVGrid_6_1", "annual_consumption"
                ]
                * profiles_new["industrial"]
            ).values,
        ).all()

    def test_predefined_charging_points_by_use_case(self, caplog):
        index = pd.date_range("1/1/2018", periods=3, freq="H")
        self.edisgo.timeseries.timeindex = index

        # test assertion error
        self.edisgo.timeseries.predefined_conventional_loads_by_sector(
            self.edisgo, pd.DataFrame()
        )
        assert "The profile you entered is empty. Method is skipped." in caplog.text

        # add charging points to MV and LV
        df_cp = pd.DataFrame(
            {
                "bus": [
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_LVGrid_1_5",
                    "Bus_BranchTee_LVGrid_1_5",
                ],
                "p_set": [0.1, 0.2, 0.3, 0.4],
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
        self.edisgo.topology.loads_df = pd.concat(
            [
                self.edisgo.topology.loads_df,
                df_cp,
            ]
        )

        # test all charging points
        profiles = pd.DataFrame(
            index=index,
            columns=["hpc", "public", "home", "work"],
            data=[
                [3.03, 0.22, 0.01, 0.1],
                [2.04, 0.41, 0.20, 0.2],
                [7.01, 0.16, 0.24, 1.0],
            ],
        )
        self.edisgo.timeseries.predefined_charging_points_by_use_case(
            self.edisgo, profiles
        )
        # fmt: off
        assert self.edisgo.timeseries.time_series_raw.\
            charging_points_active_power_by_use_case.shape\
               == (3, 4)
        # fmt: on

        for name, cp in df_cp.iterrows():
            assert np.isclose(
                self.edisgo.timeseries.loads_active_power[name].values,
                (
                    self.edisgo.topology.charging_points_df.loc[name, "p_set"]
                    * profiles[cp.sector]
                ).values,
            ).all()
        # test single charging points
        profiles_new = profiles * 0.5
        self.edisgo.timeseries.predefined_charging_points_by_use_case(
            self.edisgo, profiles_new, load_names=["CP1", "CP3"]
        )
        for name, cp in df_cp.iterrows():
            if name in ["CP1", "CP3"]:
                assert np.isclose(
                    self.edisgo.timeseries.loads_active_power[name].values,
                    (
                        self.edisgo.topology.charging_points_df.loc[name, "p_set"]
                        * profiles_new[cp.sector]
                    ).values,
                ).all()
            else:
                assert np.isclose(
                    self.edisgo.timeseries.loads_active_power[name].values,
                    (
                        self.edisgo.topology.charging_points_df.loc[name, "p_set"]
                        * profiles[cp.sector]
                    ).values,
                ).all()
        # test warning
        profiles = pd.DataFrame(
            index=index,
            columns=["residential", "public", "home"],
            data=[[3.03, 0.01, 0.1], [2.04, 0.20, 0.2], [7.01, 0.24, 1.0]],
        )
        with pytest.raises(Warning) as exc_info:
            self.edisgo.timeseries.predefined_charging_points_by_use_case(
                self.edisgo, profiles
            )
        assert (
            exc_info.value.args[0]
            == "Not all affected loads are charging points. Please check and"
            " adapt if necessary."
        )
        # fmt: off
        assert self.edisgo.timeseries.time_series_raw.\
            charging_points_active_power_by_use_case.shape\
               == (3, 5)
        # fmt: on

    def test_fixed_cosphi(self):
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
        self.edisgo.set_time_series_manual(
            storage_units_p=pd.DataFrame({"Storage_1": ts_wind}, index=timeindex)
        )
        # create heat pumps and charging points in MV and LV
        df_cp = pd.DataFrame(
            {
                "bus": [
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_MVGrid_1_2",
                    "Bus_BranchTee_LVGrid_1_5",
                    "Bus_BranchTee_LVGrid_1_5",
                ],
                "p_set": [0.1, 0.2, 0.3, 0.4],
                "type": [
                    "charging_point",
                    "heat_pump",
                    "charging_point",
                    "heat_pump",
                ],
            },
            index=["CP1", "HP1", "CP2", "HP2"],
        )
        self.edisgo.topology.loads_df = pd.concat(
            [
                self.edisgo.topology.loads_df,
                df_cp,
            ]
        )
        self.edisgo.set_time_series_manual(
            loads_p=pd.DataFrame(
                {"CP1": ts_wind, "HP1": ts_wind, "CP2": ts_wind, "HP2": ts_wind},
                index=timeindex,
            )
        )

        # test different options (default, Dataframe with default, Dataframe with
        # different settings) - None is already tested in eDisGo class tests
        gen = "GeneratorFluctuating_14"  # solar LV generator
        load_1 = "Load_agricultural_LVGrid_3_1"
        load_2 = "Load_residential_LVGrid_7_3"
        load_3 = "Load_residential_LVGrid_8_12"
        self.edisgo.set_time_series_reactive_power_control(
            generators_parametrisation=pd.DataFrame(
                {
                    "components": [[gen]],
                    "mode": ["default"],
                    "power_factor": ["default"],
                },
                index=[1],
            ),
            loads_parametrisation=pd.DataFrame(
                {
                    "components": [
                        [load_1, "CP1", "HP1", "CP2", "HP2"],
                        [load_2, load_3],
                    ],
                    "mode": ["default", "capacitive"],
                    "power_factor": ["default", 0.98],
                },
                index=[1, 2],
            ),
            storage_units_parametrisation="default",
        )
        assert self.edisgo.timeseries.generators_reactive_power.shape == (3, 1)
        assert self.edisgo.timeseries.loads_reactive_power.shape == (3, 7)
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (3, 1)
        assert (
            np.isclose(
                self.edisgo.timeseries.generators_reactive_power.loc[:, gen],
                ts_solar * -np.tan(np.arccos(0.95)) * 0.005,
            )
        ).all()
        assert (
            np.isclose(
                self.edisgo.timeseries.loads_reactive_power.loc[:, load_1],
                self.edisgo.timeseries.loads_active_power.loc[:, load_1]
                * np.tan(np.arccos(0.95)),
            )
        ).all()
        assert (
            np.isclose(
                self.edisgo.timeseries.loads_reactive_power.loc[
                    :, ["CP1", "HP1", "CP2", "HP2"]
                ],
                0.0,
            )
        ).all()
        assert (
            (
                np.isclose(
                    self.edisgo.timeseries.loads_reactive_power.loc[
                        :, [load_2, load_3]
                    ],
                    self.edisgo.timeseries.loads_active_power.loc[:, [load_2, load_3]]
                    * -np.tan(np.arccos(0.98)),
                )
            )
            .all()
            .all()
        )
        assert (
            np.isclose(
                self.edisgo.timeseries.storage_units_reactive_power.loc[:, "Storage_1"],
                self.edisgo.timeseries.storage_units_active_power.loc[:, "Storage_1"]
                * -np.tan(np.arccos(0.9)),
            )
        ).all()

    def test_residual_load(self):
        self.edisgo.set_time_series_worst_case_analysis()
        time_steps_load_case = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ].values
        peak_load = (
            self.edisgo.topology.loads_df.p_set.sum()
            + self.edisgo.topology.storage_units_df.p_nom.sum()
        )
        assert np.allclose(
            self.edisgo.timeseries.residual_load.loc[time_steps_load_case], peak_load
        )
        time_steps_feedin_case = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("feed")
        ].values
        assert (
            self.edisgo.timeseries.residual_load.loc[time_steps_feedin_case] < 0
        ).all()

    def test_timesteps_load_feedin_case(self):
        self.edisgo.set_time_series_worst_case_analysis()
        time_steps_load_case = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ].values
        assert (
            self.edisgo.timeseries.timesteps_load_feedin_case.loc[time_steps_load_case]
            == "load_case"
        ).all()
        time_steps_feedin_case = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("feed")
        ].values
        assert (
            self.edisgo.timeseries.timesteps_load_feedin_case.loc[
                time_steps_feedin_case
            ]
            == "feed-in_case"
        ).all()

    def test_reduce_memory(self):
        self.edisgo.set_time_series_worst_case_analysis()
        # fmt: off
        self.edisgo.timeseries.time_series_raw.\
            fluctuating_generators_active_power_by_technology = pd.DataFrame(
                data={
                    "wind": [1.23, 2.0, 5.0, 6.0],
                    "solar": [3.0, 4.0, 7.0, 8.0],
                },
                index=self.edisgo.timeseries.timeindex,
            )
        # fmt: on

        # check with default value
        assert (self.edisgo.timeseries.loads_active_power.dtypes == "float64").all()
        # fmt: off
        assert (
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology.dtypes
            == "float64"
        ).all()
        # fmt: on
        self.edisgo.timeseries.reduce_memory()
        assert (self.edisgo.timeseries.loads_active_power.dtypes == "float32").all()
        assert (self.edisgo.timeseries.loads_reactive_power.dtypes == "float32").all()
        # fmt: off
        assert (
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology.dtypes
            == "float32"
        ).all()
        # fmt: on

        # check arguments
        self.edisgo.timeseries.reduce_memory(
            to_type="float16",
            attr_to_reduce=["loads_reactive_power"],
            time_series_raw=False,
        )

        assert (self.edisgo.timeseries.loads_active_power.dtypes == "float32").all()
        assert (self.edisgo.timeseries.loads_reactive_power.dtypes == "float16").all()
        # fmt: off
        assert (
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology.dtypes
            == "float32"
        ).all()
        # fmt: on

    def test_to_csv(self):
        timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        self.edisgo.set_timeindex(timeindex)

        # create dummy time series
        loads_active_power = pd.DataFrame(
            {"load1": [1.4, 2.3], "load2": [2.4, 1.3]}, index=timeindex
        )
        self.edisgo.timeseries.loads_active_power = loads_active_power
        generators_reactive_power = pd.DataFrame(
            {"gen1": [1.4, 2.3], "gen2": [2.4, 1.3]}, index=timeindex
        )
        self.edisgo.timeseries.generators_reactive_power = generators_reactive_power
        # fmt: off
        self.edisgo.timeseries.time_series_raw. \
            fluctuating_generators_active_power_by_technology = pd.DataFrame(
                data={
                    "wind": [1.23, 2.0],
                    "solar": [3.0, 4.0],
                },
                index=self.edisgo.timeseries.timeindex,
            )
        # fmt: on

        # test with default values
        save_dir = os.path.join(os.getcwd(), "timeseries_csv")
        self.edisgo.timeseries.to_csv(save_dir)

        files_in_timeseries_dir = os.listdir(save_dir)
        assert len(files_in_timeseries_dir) == 2
        assert "loads_active_power.csv" in files_in_timeseries_dir
        assert "generators_reactive_power.csv" in files_in_timeseries_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16 and saving TimeSeriesRaw
        self.edisgo.timeseries.to_csv(
            save_dir, reduce_memory=True, to_type="float16", time_series_raw=True
        )

        assert (
            self.edisgo.timeseries.generators_reactive_power.dtypes == "float16"
        ).all()
        files_in_timeseries_dir = os.listdir(save_dir)
        assert len(files_in_timeseries_dir) == 3
        files_in_timeseries_raw_dir = os.listdir(
            os.path.join(save_dir, "time_series_raw")
        )
        assert len(files_in_timeseries_raw_dir) == 1
        assert (
            "fluctuating_generators_active_power_by_technology.csv"
            in files_in_timeseries_raw_dir
        )

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):
        timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        self.edisgo.set_timeindex(timeindex)

        # create dummy time series
        loads_reactive_power = pd.DataFrame(
            {"load1": [1.4, 2.3], "load2": [2.4, 1.3]}, index=timeindex
        )
        self.edisgo.timeseries.loads_reactive_power = loads_reactive_power
        generators_active_power = pd.DataFrame(
            {"gen1": [1.4, 2.3], "gen2": [2.4, 1.3]}, index=timeindex
        )
        self.edisgo.timeseries.generators_active_power = generators_active_power
        fluc_gen = pd.DataFrame(
            data={
                "wind": [1.23, 2.0],
                "solar": [3.0, 4.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        # fmt: off
        self.edisgo.timeseries.time_series_raw. \
            fluctuating_generators_active_power_by_technology = fluc_gen
        # fmt: on

        # write to csv
        save_dir = os.path.join(os.getcwd(), "timeseries_csv")
        self.edisgo.timeseries.to_csv(save_dir, time_series_raw=True)

        # reset TimeSeries
        self.edisgo.timeseries.reset()

        self.edisgo.timeseries.from_csv(save_dir)

        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.loads_reactive_power,
            loads_reactive_power,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.generators_active_power,
            generators_active_power,
            check_freq=False,
        )
        # fmt: off
        assert (
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology.empty
        )
        # fmt: on

        self.edisgo.timeseries.from_csv(save_dir, time_series_raw=True)

        # fmt: off
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.time_series_raw.
            fluctuating_generators_active_power_by_technology,
            fluc_gen,
            check_freq=False,
        )
        # fmt: on

        shutil.rmtree(save_dir)

    def test_integrity_check(self, caplog):
        attrs = [
            "loads_active_power",
            "loads_reactive_power",
            "generators_active_power",
            "generators_reactive_power",
            "storage_units_active_power",
            "storage_units_reactive_power",
        ]
        # check warning empty timeindex
        self.edisgo.timeseries.check_integrity()
        assert "No time index set. Empty time series will be returned." in caplog.text
        caplog.clear()
        # add timeseries
        index = pd.date_range("1/1/2018", periods=3, freq="H")
        self.edisgo.timeseries.timeindex = index
        for attr in attrs:
            tmp = attr.split("_")
            if len(tmp) == 3:
                comp_type = tmp[0]
            elif len(tmp) == 4:
                comp_type = "_".join(tmp[0:2])
            comps = getattr(self.edisgo.topology, comp_type + "_df").index
            setattr(
                self.edisgo.timeseries,
                comp_type + "_active_power",
                pd.DataFrame(index=index, columns=comps, data=0),
            )
            setattr(
                self.edisgo.timeseries,
                comp_type + "_reactive_power",
                pd.DataFrame(index=index, columns=comps, data=0),
            )
        # check warning for null values
        for attr in attrs:
            ts_tmp = getattr(self.edisgo.timeseries, attr)
            if not ts_tmp.empty:
                ts_tmp.iloc[0, 0] = np.NaN
                setattr(self.edisgo.timeseries, attr, ts_tmp)
                self.edisgo.timeseries.check_integrity()
                assert "There are null values in {}".format(attr) in caplog.text
                caplog.clear()
                ts_tmp.iloc[0, 0] = 0
                setattr(self.edisgo.timeseries, attr, ts_tmp)
        # check warning for duplicated indices and columns
        for attr in attrs:
            ts_tmp = getattr(self.edisgo.timeseries, attr)
            if not ts_tmp.empty:
                # check for duplicated indices
                ts_tmp_duplicated = pd.concat([ts_tmp, ts_tmp.iloc[0:2]])
                setattr(self.edisgo.timeseries, attr, ts_tmp_duplicated)
                self.edisgo.timeseries.check_integrity()
                assert (
                    "{} has duplicated indices: {}".format(
                        attr, ts_tmp.iloc[0:2].index.values
                    )
                    in caplog.text
                )
                caplog.clear()
                setattr(self.edisgo.timeseries, attr, ts_tmp)
                # check for duplicated columns
                ts_tmp_duplicated = pd.concat([ts_tmp, ts_tmp.iloc[:, 0:2]], axis=1)
                setattr(self.edisgo.timeseries, attr, ts_tmp_duplicated)
                self.edisgo.timeseries.check_integrity()
                assert (
                    "{} has duplicated columns: {}".format(
                        attr, ts_tmp.iloc[:, 0:2].columns.values
                    )
                    in caplog.text
                )
                caplog.clear()
                setattr(self.edisgo.timeseries, attr, ts_tmp)

    def test_drop_component_time_series(self):
        time_series_obj = timeseries.TimeSeries()

        # check that no error is raised in case of empty dataframe
        time_series_obj.drop_component_time_series("loads_active_power", "Load1")

        # add dummy time series
        time_series_obj.timeindex = pd.date_range("1/1/2018", periods=4, freq="H")
        df = pd.DataFrame(
            data={
                "load_1": [1.23, 2.0, 5.0, 6.0],
                "load_2": [3.0, 4.0, 7.0, 8.0],
            },
            index=time_series_obj.timeindex,
        )
        time_series_obj.loads_active_power = df

        # check with dropping one existing load and one non-existing load
        time_series_obj.drop_component_time_series(
            "loads_active_power", ["Load1", "load_1"]
        )
        assert time_series_obj.loads_active_power.shape == (4, 1)
        assert "load_1" not in time_series_obj.loads_active_power.columns

        # check with dropping all existing loads
        time_series_obj.drop_component_time_series("loads_active_power", ["load_2"])
        assert time_series_obj.loads_active_power.empty

    def test_add_component_time_series(self):
        time_series_obj = timeseries.TimeSeries()
        time_series_obj.timeindex = pd.date_range("1/1/2018", periods=4, freq="H")

        df = pd.DataFrame(
            data={
                "load_1": [1.23, 2.0, 5.0, 6.0],
                "load_2": [3.0, 4.0, 7.0, 8.0],
            },
            index=time_series_obj.timeindex,
        )

        # check with matching time index
        time_series_obj.add_component_time_series("loads_active_power", df)
        assert time_series_obj.loads_active_power.shape == (4, 2)
        assert "load_1" in time_series_obj.loads_active_power.columns

        # check with time indexes that do not match
        df = pd.DataFrame(
            data={
                "load_3": [5.0, 6.0],
                "load_4": [7.0, 8.0],
            },
            index=time_series_obj.timeindex[0:2],
        )
        time_series_obj.add_component_time_series("loads_active_power", df.iloc[:2])
        assert time_series_obj.loads_active_power.shape == (4, 4)
        assert "load_3" in time_series_obj.loads_active_power.columns

    def test_check_if_components_exist(self):
        edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_path)

        # check all components exist
        component_names = edisgo_obj.timeseries._check_if_components_exist(
            edisgo_obj,
            ["GeneratorFluctuating_15", "GeneratorFluctuating_24"],
            "generators",
        )
        assert len(component_names) == 2
        assert "GeneratorFluctuating_15" in component_names

        # check no components exist
        component_names = edisgo_obj.timeseries._check_if_components_exist(
            edisgo_obj, ["Storage_3"], "storage_units"
        )
        assert len(component_names) == 0

        # check some components exist
        component_names = edisgo_obj.timeseries._check_if_components_exist(
            edisgo_obj,
            ["Load_residential_LVGrid_5_3", "Load_residential_LVGrid_5"],
            "loads",
        )
        assert len(component_names) == 1
        assert "Load_residential_LVGrid_5_3" in component_names

    def test_resample(self):
        self.edisgo.set_time_series_worst_case_analysis()

        len_timeindex_orig = len(self.edisgo.timeseries.timeindex)
        mean_value_orig = self.edisgo.timeseries.generators_active_power.mean()
        index_orig = self.edisgo.timeseries.timeindex.copy()

        # test up-sampling
        self.edisgo.timeseries.resample()
        # check if resampled length of time index is 4 times original length of
        # timeindex
        assert len(self.edisgo.timeseries.timeindex) == 4 * len_timeindex_orig
        # check if mean value of resampled data is the same as mean value of original
        # data
        assert (
            np.isclose(
                self.edisgo.timeseries.generators_active_power.mean(),
                mean_value_orig,
                atol=1e-5,
            )
        ).all()
        # check if index is the same after resampled back
        self.edisgo.timeseries.resample(freq="1h")
        assert_index_equal(self.edisgo.timeseries.timeindex, index_orig)

        # same tests for down-sampling
        self.edisgo.timeseries.resample(freq="2h")
        assert len(self.edisgo.timeseries.timeindex) == 0.5 * len_timeindex_orig
        assert (
            np.isclose(
                self.edisgo.timeseries.generators_active_power.mean(),
                mean_value_orig,
                atol=1e-5,
            )
        ).all()

        # test bfill
        self.edisgo.timeseries.resample(method="bfill")
        assert len(self.edisgo.timeseries.timeindex) == 4 * len_timeindex_orig
        assert np.isclose(
            self.edisgo.timeseries.generators_active_power.iloc[1:, :].loc[
                :, "GeneratorFluctuating_3"
            ],
            2.26950,
            atol=1e-5,
        ).all()

        # test interpolate
        self.edisgo.timeseries.reset()
        self.edisgo.set_time_series_worst_case_analysis()
        len_timeindex_orig = len(self.edisgo.timeseries.timeindex)
        ts_orig = self.edisgo.timeseries.generators_active_power.loc[
            :, "GeneratorFluctuating_3"
        ]
        self.edisgo.timeseries.resample(method="interpolate")
        assert len(self.edisgo.timeseries.timeindex) == 4 * len_timeindex_orig
        assert np.isclose(
            self.edisgo.timeseries.generators_active_power.at[
                pd.Timestamp("1970-01-01 01:30:00"), "GeneratorFluctuating_3"
            ],
            (
                ts_orig.at[pd.Timestamp("1970-01-01 01:00:00")]
                + ts_orig.at[pd.Timestamp("1970-01-01 02:00:00")]
            )
            / 2,
            atol=1e-5,
        )

    def test_scale_timeseries(self):
        self.edisgo.set_time_series_worst_case_analysis()
        edisgo_scaled = copy.deepcopy(self.edisgo)
        edisgo_scaled.timeseries.scale_timeseries(
            p_scaling_factor=0.5, q_scaling_factor=0.4
        )

        assert_frame_equal(
            edisgo_scaled.timeseries.generators_active_power,
            self.edisgo.timeseries.generators_active_power * 0.5,
        )
        assert_frame_equal(
            edisgo_scaled.timeseries.generators_reactive_power,
            self.edisgo.timeseries.generators_reactive_power * 0.4,
        )
        assert_frame_equal(
            edisgo_scaled.timeseries.loads_active_power,
            self.edisgo.timeseries.loads_active_power * 0.5,
        )
        assert_frame_equal(
            edisgo_scaled.timeseries.loads_reactive_power,
            self.edisgo.timeseries.loads_reactive_power * 0.4,
        )
        assert_frame_equal(
            edisgo_scaled.timeseries.storage_units_active_power,
            self.edisgo.timeseries.storage_units_active_power * 0.5,
        )
        assert_frame_equal(
            edisgo_scaled.timeseries.storage_units_reactive_power,
            self.edisgo.timeseries.storage_units_reactive_power * 0.4,
        )


class TestTimeSeriesRaw:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        # add dummy time series
        self.time_series_raw = timeseries.TimeSeriesRaw()
        timeindex = pd.date_range("1/1/2018", periods=4, freq="H")
        self.df = pd.DataFrame(
            data={
                "residential": [1.23, 2.0, 5.0, 6.0],
                "industrial": [3.0, 4.0, 7.0, 8.0],
            },
            index=timeindex,
        )
        self.time_series_raw.conventional_loads_active_power_by_sector = self.df
        self.time_series_raw.charging_points_active_power_by_use_case = self.df
        self.q_control = pd.DataFrame(
            {
                "type": ["fixed_cosphi", "fixed_cosphi"],
                "q_sign": [1, -1],
                "power_factor": [1.0, 0.98],
                "parametrisation": [np.nan, np.nan],
            },
            index=["gen_1", "laod_2"],
        )
        self.time_series_raw.q_control = self.q_control

    def test_reduce_memory(self):
        # check with default value
        assert (
            self.time_series_raw.conventional_loads_active_power_by_sector.dtypes
            == "float64"
        ).all()
        assert self.time_series_raw.q_control.power_factor.dtype == "float64"
        self.time_series_raw.reduce_memory()
        assert (
            self.time_series_raw.conventional_loads_active_power_by_sector.dtypes
            == "float32"
        ).all()
        assert (
            self.time_series_raw.charging_points_active_power_by_use_case.dtypes
            == "float32"
        ).all()
        assert self.time_series_raw.q_control.power_factor.dtype == "float64"

        # check arguments
        self.time_series_raw.reduce_memory(
            to_type="float16",
            attr_to_reduce=["conventional_loads_active_power_by_sector"],
        )

        assert (
            self.time_series_raw.conventional_loads_active_power_by_sector.dtypes
            == "float16"
        ).all()
        assert (
            self.time_series_raw.charging_points_active_power_by_use_case.dtypes
            == "float32"
        ).all()

    def test_to_csv(self):
        # test with default values
        save_dir = os.path.join(os.getcwd(), "timeseries_csv")
        self.time_series_raw.to_csv(save_dir)

        files_in_timeseries_dir = os.listdir(save_dir)
        assert len(files_in_timeseries_dir) == 3
        assert (
            "conventional_loads_active_power_by_sector.csv" in files_in_timeseries_dir
        )
        assert "charging_points_active_power_by_use_case.csv" in files_in_timeseries_dir
        assert "q_control.csv" in files_in_timeseries_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16 and saving TimeSeriesRaw
        self.time_series_raw.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (
            self.time_series_raw.conventional_loads_active_power_by_sector.dtypes
            == "float16"
        ).all()
        files_in_timeseries_dir = os.listdir(save_dir)
        assert len(files_in_timeseries_dir) == 3

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):
        # write to csv
        save_dir = os.path.join(os.getcwd(), "timeseries_csv")
        self.time_series_raw.to_csv(save_dir, time_series_raw=True)

        # reset TimeSeriesRaw
        self.time_series_raw = timeseries.TimeSeriesRaw()

        self.time_series_raw.from_csv(save_dir)

        pd.testing.assert_frame_equal(
            self.time_series_raw.conventional_loads_active_power_by_sector,
            self.df,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.time_series_raw.charging_points_active_power_by_use_case,
            self.df,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.time_series_raw.q_control,
            self.q_control,
            check_freq=False,
        )

        shutil.rmtree(save_dir)
