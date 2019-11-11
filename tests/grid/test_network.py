import os
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from math import tan, acos
import pytest

from edisgo.network.topology import Topology
from edisgo.network.timeseries import TimeSeriesControl, TimeSeries
from edisgo.tools.config import Config
from edisgo.io import ding0_import
from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks


class TestEDisGo:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')

    def test_exceptions(self):
        msg = "No results pfa_p to check. Please analyze grid first."
        with pytest.raises(Exception, match=msg):
            checks.mv_lv_station_load(self.edisgo)
        msg = "No results i_res to check. Please analyze grid first."
        with pytest.raises(Exception, match=msg):
            checks.mv_line_load(self.edisgo)
        self.edisgo.analyze()
        msg = "Inserted grid of unknown type."
        with pytest.raises(ValueError, match=msg):
            checks._line_load(self.edisgo, None, pd.DataFrame)
        with pytest.raises(ValueError, match=msg):
            checks._station_load(self.edisgo, None, pd.DataFrame)

    def test_crit_station(self):
        timesteps = pd.date_range('1/1/1970', periods=2, freq='H')
        # calculate results if not already existing
        if self.edisgo.results.pfa_p is None:
            self.edisgo.analyze()
        # check results
        overloaded_mv_station = checks.hv_mv_station_load(self.edisgo)
        assert overloaded_mv_station.empty
        overloaded_lv_station = checks.mv_lv_station_load(self.edisgo)
        assert(len(overloaded_lv_station) == 6)
        assert (np.isclose(
            overloaded_lv_station.at['Bus_secondary_LVStation_1', 's_pfa'],
            0.41762))
        assert (overloaded_lv_station.at[
                    'Bus_secondary_LVStation_1', 'time_index'] == timesteps[1])
        assert (np.isclose(
            overloaded_lv_station.at['Bus_secondary_LVStation_4', 's_pfa'],
            0.084253))
        assert (overloaded_lv_station.at[
                    'Bus_secondary_LVStation_4', 'time_index'] == timesteps[0])

    def test_crit_lines(self):
        timesteps = pd.date_range('1/1/1970', periods=2, freq='H')
        if self.edisgo.results.i_res is None:
            self.edisgo.analyze()
        mv_crit_lines = checks.mv_line_load(self.edisgo)
        lv_crit_lines = checks.lv_line_load(self.edisgo)
        assert len(lv_crit_lines) == 10
        assert (lv_crit_lines.time_index == timesteps[1]).all()
        assert lv_crit_lines.at[
                   'Line_10000016', 'max_rel_overload'] == 1.1936977825332047
        assert lv_crit_lines.at[
                   'Line_50000007', 'max_rel_overload'] == 1.418679228498681
        assert len(mv_crit_lines) == 9
        assert (mv_crit_lines.time_index == timesteps[0]).all()
        assert mv_crit_lines.at[
                   'Line_10006', 'max_rel_overload'] == 2.3256986822390515
        assert mv_crit_lines.at[
                   'Line_10026', 'max_rel_overload'] == 2.1246019520230495

    def test_analyze(self):
        timesteps = pd.date_range('1/1/1970', periods=2, freq='H')
        if self.edisgo.results.grid_losses is None:
            self.edisgo.analyze()
        # check results
        assert(np.isclose(self.edisgo.results.grid_losses.loc[timesteps].values,
               np.array([[0.20826765, 0.20945498], [0.06309538, 0.06346827]])).all())
        assert(np.isclose(self.edisgo.results.hv_mv_exchanges.loc[timesteps].values,
                np.array([[-21.26234, 10.69626], [1.29379, 0.52485]])).all())
        assert(np.isclose(self.edisgo.results.pfa_v_mag_pu.lv.loc[
                    timesteps, 'GeneratorFluctuating_18'].values,
                np.array([1.01699, 0.99915])).all())
        assert(np.isclose(self.edisgo.results.pfa_v_mag_pu.mv.loc[
                     timesteps, 'virtual_Bus_primary_LVStation_4'].values,
                 np.array([1.00629, 0.99917])).all())
        assert (np.isclose(
            self.edisgo.results.pfa_p.loc[timesteps, 'Line_60000003'].values,
            np.array([0.00765636, 0.07659877])).all())
        assert (np.isclose(
            self.edisgo.results.pfa_q.loc[timesteps, 'Line_60000003'].values,
            np.array([0.00251644, 0.0251678])).all())
        assert (np.isclose(self.edisgo.results.i_res.loc[
                     timesteps, ['Line_10002', 'Line_90000025']].values,
                 np.array(
                     [[0.00175807, 0.00015960], [0.01172047, 0.00164367]])).all())

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
            print('network converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # test exception
        msg = "The entered mode is not a valid option."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.to_pypsa(mode='unknown')

    def test_mv_to_pypsa(self):
        # test only mv
        timesteps = pd.date_range('1/1/1970', periods=1, freq='H')
        pypsa_network = self.edisgo.to_pypsa(mode='mv')
        pf_results = pypsa_network.pf(timesteps)
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('mv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # test mvlv
        pypsa_network = self.edisgo.to_pypsa(mode='mvlv')
        pf_results = pypsa_network.pf(timesteps)
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('mvlv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")

    def test_lv_to_pypsa(self):
        # test lv to pypsa
        timesteps = pd.date_range('1/1/1970', periods=1, freq='H')
        pypsa_network = self.edisgo.to_pypsa(mode='lv', lv_grid_name='LVGrid_2')
        pf_results = pypsa_network.pf(timesteps)
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('lv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # test exception
        msg = "For exporting lv grids, name of lv_grid has to be provided."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.to_pypsa(mode='lv')




class TestTimeSeriesControl:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.topology = Topology()
        self.timeseries = TimeSeries()
        self.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, self)

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
        number_of_cols = len(self.topology._generators_df.index)
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
        exp = pd.Series(data=[0.1 * 143 * 0.0002404],
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
        exp = pd.Series(data=[1.0 * 143 * 0.0002404],
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
        self.topology._generators_df.at[gen, 'bus'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        gen = 'GeneratorFluctuating_24'
        self.topology._generators_df.at[gen, 'p_nom'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)

        load = 'Load_agricultural_LVGrid_1_1'
        self.topology._loads_df.at[load, 'annual_consumption'] = None
        with pytest.raises(AttributeError, match=load):
            ts_control._worst_case_load(modes=None)

        # test no other generators
