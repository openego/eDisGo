import os
import pandas as pd
from pandas.util.testing import assert_series_equal
from math import tan, acos
import pytest

from edisgo.network.network import Network
from edisgo.network.timeseries import TimeSeriesControl, TimeSeries
from edisgo.tools.config import Config
from edisgo.data import import_data
from edisgo import EDisGo


class TestEDisGo:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')

    def test_reinforce(self):
        print()
        #self.edisgo.reinforce()


class TestNetwork:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')

    def test_to_pypsa(self):
        # run powerflow and check results
        timesteps = pd.date_range('1/1/1970', periods=1, freq='H')
        pypsa_network = self.edisgo.to_pypsa()
        pf_results = pypsa_network.pf(timesteps)

        if all(pf_results['converged']['0'].tolist()):
            print('converged')
        else:
            raise ValueError("Power flow analysis did not converge.")


class TestTimeSeriesControl:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.network = Network()
        self.timeseries = TimeSeries()
        self.config = Config()
        import_data.import_ding0_grid(test_network_directory, self)

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
        number_of_cols = len(self.network.generators_df.index)
        assert self.timeseries.generators_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.generators_reactive_power.shape == (
            number_of_timesteps, number_of_cols)
        number_of_cols = len(self.network.loads_df.index)
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
        exp = pd.Series(data=[0.15 * 1520 * 0.0002404, 1.0 * 1520 * 0.0002404],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_agricultural_LVGrid_1_2'  # agricultural, lv
        exp = pd.Series(data=[0.1 * 514 * 0.00024036, 1.0 * 514 * 0.00024036],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_residential_LVGrid_3_3'  # residential, lv
        exp = pd.Series(data=[0.1 * 4.3 * 0.00021372, 1.0 * 4.3 * 0.00021372],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_industrial_LVGrid_6_1'  # industrial, lv
        exp = pd.Series(data=[0.1 * 580 * 0.000132, 1.0 * 580 * 0.000132],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_retail_LVGrid_9_14'  # industrial, lv
        exp = pd.Series(data=[0.1 * 143 * 0.0002404, 1.0 * 143 * 0.0002404],
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
        self.network._generators_df.at[gen, 'bus'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        gen = 'GeneratorFluctuating_24'
        self.network._generators_df.at[gen, 'p_nom'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)

        load = 'Load_agricultural_LVGrid_1_1'
        self.network._loads_df.at[load, 'annual_consumption'] = None
        with pytest.raises(AttributeError, match=load):
            ts_control._worst_case_load(modes=None)

        # test for only feed-in or load case
        # test no other generators
