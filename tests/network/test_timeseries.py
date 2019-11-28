import os
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from math import tan, acos
import pytest
import shutil

from edisgo.network.topology import Topology
from edisgo.tools.config import Config
from edisgo.network.timeseries import TimeSeriesControl, TimeSeries, \
    import_load_timeseries
from edisgo.io import ding0_import


class TestTimeSeriesControl:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        self.topology = Topology()
        self.timeseries = TimeSeries()
        self.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, self)

    def test_to_csv(self):
        cur_dir = os.getcwd()
        TimeSeriesControl(edisgo_obj=self, mode='worst-case')
        self.timeseries.to_csv(cur_dir)
        #create edisgo obj to compare
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        edisgo = pd.DataFrame()
        edisgo.topology = Topology()
        edisgo.timeseries = TimeSeries()
        edisgo.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, edisgo)
        TimeSeriesControl(
            edisgo, mode='manual',
            timeindex=self.timeseries.loads_active_power.index,
            loads_active_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_active_power.csv')),
            loads_reactive_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_reactive_power.csv')),
            generators_active_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir, 'timeseries', 'generators_active_power.csv')),
            generators_reactive_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir,
                        'timeseries', 'generators_reactive_power.csv')),
            storage_units_active_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir,
                        'timeseries', 'storage_units_active_power.csv')),
            storage_units_reactive_power=pd.DataFrame.from_csv(
                os.path.join(cur_dir,
                             'timeseries', 'storage_units_reactive_power.csv'))
        )
        # check if timeseries are the same
        assert_frame_equal(self.timeseries.loads_active_power,
                           edisgo.timeseries.loads_active_power,
                           check_names=False)
        assert_frame_equal(self.timeseries.loads_reactive_power,
                           edisgo.timeseries.loads_reactive_power,
                           check_names=False)
        assert_frame_equal(self.timeseries.generators_active_power,
                           edisgo.timeseries.generators_active_power,
                           check_names=False)
        assert_frame_equal(self.timeseries.generators_reactive_power,
                           edisgo.timeseries.generators_reactive_power,
                           check_names=False)
        assert_frame_equal(self.timeseries.storage_units_active_power,
                           edisgo.timeseries.storage_units_active_power,
                           check_names=False)
        assert_frame_equal(self.timeseries.storage_units_reactive_power,
                           edisgo.timeseries.storage_units_reactive_power,
                           check_names=False)
        # delete folder
        # Todo: check files before rmtree?
        shutil.rmtree(os.path.join(cur_dir, 'timeseries'), ignore_errors=True)
        self.timeseries = TimeSeries()

    def test_timeseries_imported(self):
        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'Generator_1': [0.775]*8760},
                                           index=timeindex)
        # test error raising in case of missing ts for dispatchable gens
        msg = \
            'Your input for "timeseries_generation_dispatchable" is not valid.'
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb')
        # test error raising in case of missing ts for loads
        msg = 'Your input for "timeseries_load" is not valid.'
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                  timeseries_generation_fluctuating='oedb',
                  timeseries_generation_dispatchable=ts_gen_dispatchable)

        TimeSeriesControl(edisgo_obj=self,
                          timeseries_generation_fluctuating='oedb',
                          timeseries_generation_dispatchable=ts_gen_dispatchable,
                          timeseries_load='demandlib')

        #Todo: test with inserted reactive generation and/or reactive load
        print()

    def test_import_load_timeseries(self):
        with pytest.raises(NotImplementedError):
            import_load_timeseries(self.config, '')
        timeindex = pd.date_range('1/1/2018', periods=8760, freq='H')
        load = import_load_timeseries(self.config, 'demandlib')
        assert (load.columns == ['retail', 'residential',
                                 'agricultural', 'industrial']).all()
        assert load.loc[timeindex[453], 'retail'] == 8.335076810751597e-05
        assert load.loc[timeindex[13], 'residential'] == 0.00017315167492271323
        assert load.loc[timeindex[6328], 'agricultural'] == \
               0.00010134645909959844
        assert load.loc[timeindex[4325], 'industrial'] == 9.91768322919766e-05

    def test_worst_case(self):
        """Test creation of worst case time series"""

        ts_control = TimeSeriesControl(edisgo_obj=self, mode='worst-case')

        # check type
        assert isinstance(
            self.timeseries.generators_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.generators_reactive_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_reactive_power, pd.DataFrame)

        # check shape
        number_of_timesteps = len(self.timeseries.timeindex)
        number_of_cols = len(self.topology.generators_df.index)
        assert self.timeseries.generators_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.generators_reactive_power.shape == (
            number_of_timesteps, number_of_cols)
        number_of_cols = len(self.topology.loads_df.index)
        assert self.timeseries.loads_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.loads_reactive_power.shape == (
            number_of_timesteps, number_of_cols)

        # value
        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[1 * 0.775, 0 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_2'  # wind, mv
        exp = pd.Series(data=[1 * 2.3, 0 * 2.3], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_3'  # solar, mv
        exp = pd.Series(data=[0.85 * 2.67, 0 * 2.67], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_20'  # solar, lv
        exp = pd.Series(data=[0.85 * 0.005, 0 * 0.005], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.95))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        load = 'Load_retail_MVGrid_1_Load_aggregated_retail_' \
               'MVGrid_1_1'  # retail, mv
        exp = pd.Series(data=[0.15 * 0.31, 1.0 * 0.31],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_agricultural_LVGrid_1_2'  # agricultural, lv
        exp = pd.Series(data=[0.1 * 0.0523, 1.0 * 0.0523],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_residential_LVGrid_3_3'  # residential, lv
        exp = pd.Series(data=[0.1 * 0.001209, 1.0 * 0.001209],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        # test for only feed-in case
        TimeSeriesControl(edisgo_obj=self, mode='worst-case-feedin')

        # value
        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[1 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)
        load = 'Load_retail_LVGrid_9_14'  # industrial, lv
        exp = pd.Series(data=[0.1 * 0.001222],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        # test for only load case
        TimeSeriesControl(edisgo_obj=self, mode='worst-case-load')

        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[0 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)
        load = 'Load_retail_LVGrid_9_14'  # industrial, lv
        exp = pd.Series(data=[1.0 * 0.001222],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        # test error raising in case of missing load/generator parameter

        gen = 'GeneratorFluctuating_14'
        val_pre = self.topology._generators_df.at[gen, 'bus']
        self.topology._generators_df.at[gen, 'bus'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        self.topology._generators_df.at[gen, 'bus'] = val_pre
        gen = 'GeneratorFluctuating_24'
        val_pre = self.topology._generators_df.at[gen, 'p_nom']
        self.topology._generators_df.at[gen, 'p_nom'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        self.topology._generators_df.at[gen, 'p_nom'] = val_pre
        load = 'Load_agricultural_LVGrid_1_1'
        val_pre = self.topology._loads_df.at[load, 'peak_load']
        self.topology._loads_df.at[load, 'peak_load'] = None
        with pytest.raises(AttributeError, match=load):
            ts_control._worst_case_load(modes=None)
        self.topology._loads_df.at[load, 'peak_load'] = val_pre

        # test no other generators
