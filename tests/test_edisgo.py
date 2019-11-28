import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import reinforce_measures as reinforce
from matplotlib import pyplot as plt


class TestEDisGo:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        dirname = os.path.dirname(__file__)
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')
        self.timesteps = pd.date_range('1/1/1970', periods=2, freq='H')

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
        msg = "More than one MV station to extend was given. " \
              "There should only exist one station, please check."
        with pytest.raises(Exception, match=msg):
            reinforce.extend_substation_overloading(self.edisgo,
                                                    [pd.DataFrame(),
                                                     pd.DataFrame])

    def test_crit_station(self):
        # TODO: have checks of technical constraints not require edisgo
        # object and then move this test
        # calculate results if not already existing
        if self.edisgo.results.pfa_p is None:
            self.edisgo.analyze()
        # check results
        overloaded_mv_station = checks.hv_mv_station_load(self.edisgo)
        assert (len(overloaded_mv_station) == 1)
        assert (np.isclose(
            overloaded_mv_station.at['MVGrid_1', 's_pfa'],
            23.824099, atol=1e-5))
        assert (overloaded_mv_station.at[
                   'MVGrid_1', 'time_index'] == self.timesteps[0])
        overloaded_lv_station = checks.mv_lv_station_load(self.edisgo)
        assert(len(overloaded_lv_station) == 4)
        assert (np.isclose(
            overloaded_lv_station.at['LVGrid_1', 's_pfa'],
            0.17942, atol=1e-5))
        assert (overloaded_lv_station.at[
                    'LVGrid_1', 'time_index'] == self.timesteps[1])
        assert (np.isclose(
            overloaded_lv_station.at['LVGrid_4', 's_pfa'],
            0.08426, atol=1e-5))
        assert (overloaded_lv_station.at[
                    'LVGrid_4', 'time_index'] == self.timesteps[0])

    def test_crit_lines(self):
        # TODO: have checks of technical constraints not require edisgo
        # object and then move this test
        if self.edisgo.results.i_res is None:
            self.edisgo.analyze()
        mv_crit_lines = checks.mv_line_load(self.edisgo)
        lv_crit_lines = checks.lv_line_load(self.edisgo)
        assert len(lv_crit_lines) == 4
        assert (lv_crit_lines.time_index == self.timesteps[1]).all()
        assert (np.isclose(
            lv_crit_lines.at['Line_50000002', 'max_rel_overload'],
            1.02105, atol=1e-5))
        assert (np.isclose(
            lv_crit_lines.at['Line_60000003', 'max_rel_overload'],
            1.03784, atol=1e-5))
        assert len(mv_crit_lines) == 9
        assert (mv_crit_lines.time_index == self.timesteps[0]).all()
        assert (np.isclose(
            mv_crit_lines.at['Line_10006', 'max_rel_overload'],
            2.32612, atol=1e-5))
        assert (np.isclose(
            mv_crit_lines.at['Line_10026', 'max_rel_overload'],
            2.12460, atol=1e-5))

    def test_analyze(self):
        if self.edisgo.results.grid_losses is None:
            self.edisgo.analyze()
        # check results
        assert(np.isclose(
            self.edisgo.results.grid_losses.loc[self.timesteps].values,
            np.array([[0.20814, 0.20948], [0.01854, 0.01985]]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.hv_mv_exchanges.loc[self.timesteps].values,
            np.array([[-21.29377, 10.68470], [0.96392, 0.37883]]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.pfa_v_mag_pu.lv.loc[
                self.timesteps, 'Bus_GeneratorFluctuating_18'].values,
            np.array([1.01699, 0.99917]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.pfa_v_mag_pu.mv.loc[
                self.timesteps, 'virtual_Bus_primary_LVStation_4'].values,
            np.array([1.00630, 0.99929]),
            atol=1e-5).all())
        assert (np.isclose(
            self.edisgo.results.pfa_p.loc[
                self.timesteps, 'Line_60000003'].values,
            np.array([0.00799, 0.07996]), atol=1e-5).all())
        assert (np.isclose(
            self.edisgo.results.pfa_q.loc[
                self.timesteps, 'Line_60000003'].values,
            np.array([0.00263, 0.026273]), atol=1e-5).all())
        assert (np.isclose(
            self.edisgo.results.i_res.loc[
                self.timesteps, ['Line_10002', 'Line_90000025']].values,
            np.array([[0.001491, 0.000186], [0.009943, 0.001879]]),
            atol=1e-6).all())

    def test_reinforce(self):
        results = self.edisgo.reinforce(combined_analysis=True)
        assert not results.unresolved_issues
        assert len(results.grid_expansion_costs) == 18
        assert len(results.equipment_changes) == 18
        #Todo: test other relevant values

    def test_to_pypsa(self):
        # run powerflow and check results
        pypsa_network = self.edisgo.to_pypsa()
        pf_results = pypsa_network.pf(self.timesteps[0])

        if all(pf_results['converged']['0'].tolist()):
            print('network converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # ToDo maybe move slack test somewhere else
        slack_df = pypsa_network.generators[
            pypsa_network.generators.control == 'Slack']
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == 'Bus_MVStation_1'
        # test exception
        msg = "The entered mode is not a valid option."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.to_pypsa(mode='unknown')

    def test_mv_to_pypsa(self):
        # test only mv
        pypsa_network = self.edisgo.to_pypsa(mode='mv')
        pf_results = pypsa_network.pf(self.timesteps[0])
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('mv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # ToDo maybe move slack test somewhere else
        slack_df = pypsa_network.generators[
            pypsa_network.generators.control == 'Slack']
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == 'Bus_MVStation_1'
        # test mvlv
        pypsa_network = self.edisgo.to_pypsa(mode='mvlv')
        pf_results = pypsa_network.pf(self.timesteps[0])
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('mvlv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # ToDo maybe move slack test somewhere else
        slack_df = pypsa_network.generators[
            pypsa_network.generators.control == 'Slack']
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == 'Bus_MVStation_1'
        # test only mv aggregating loads by sector and generators by
        # curtailability
        pypsa_network = self.edisgo.to_pypsa(mode='mv',
                                             aggregate_generators='curtailable',
                                             aggregate_loads='sectoral')
        pf_results = pypsa_network.pf(self.timesteps[0])
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('mv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        # ToDo maybe move slack test somewhere else
        slack_df = pypsa_network.generators[
            pypsa_network.generators.control == 'Slack']
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == 'Bus_MVStation_1'
        assert np.isclose(pypsa_network.generators_t['p_set'].loc[
            self.timesteps, 'LVGrid_1_fluctuating'], [0.04845, 0]).all()
        assert np.isclose(pypsa_network.loads_t['p_set'].loc[
            self.timesteps, 'LVGrid_1_agricultural'], [0.01569, 0.1569]).all()

    def test_lv_to_pypsa(self):
        # test lv to pypsa
        pypsa_network = self.edisgo.to_pypsa(
            mode='lv', lv_grid_name='LVGrid_2')
        pf_results = pypsa_network.pf(self.timesteps[0])
        # check if pf converged
        if all(pf_results['converged']['0'].tolist()):
            print('lv converged')
        else:
            raise ValueError("Power flow analysis did not converge.")
        #ToDo maybe move slack test somewhere else
        slack_df = pypsa_network.generators[
            pypsa_network.generators.control == 'Slack']
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == 'Bus_secondary_LVStation_2'
        # test exception
        msg = "For exporting lv grids, name of lv_grid has to be provided."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.to_pypsa(mode='lv')

    def test_to_graph(self):
        graph = self.edisgo.to_graph()
        assert len(graph.nodes) == len(self.edisgo.topology.buses_df)
        assert len(graph.edges) == (len(self.edisgo.topology.lines_df) + \
            len(self.edisgo.topology.transformers_df.bus0.unique()))

    def test_edisgo_timeseries_analysis(self):
        dirname = os.path.dirname(__file__)
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'Generator_1': [0.775] * 8760},
                                           index=timeindex)
        edisgo = EDisGo(ding0_grid=test_network_directory,
                        timeseries_generation_fluctuating='oedb',
                        timeseries_generation_dispatchable=ts_gen_dispatchable,
                        timeseries_load='demandlib')
        # check if export to pypsa is possible to make sure all values are set
        pypsa_network = edisgo.to_pypsa()
        assert len(pypsa_network.generators_t['p_set']) == 8760
        assert len(pypsa_network.generators_t['q_set']) == 8760
        assert len(pypsa_network.loads_t['p_set']) == 8760
        assert len(pypsa_network.loads_t['q_set']) == 8760
        # Todo: relocate? Check other values
        edisgo.analyze(timesteps=timeindex[range(10)])
        print()

    def test_plot_mv_grid_topology(self):
        plt.ion()
        self.edisgo.plot_mv_grid_topology(technologies=True)
        plt.close('all')
        self.edisgo.plot_mv_grid_topology()
        plt.close('all')

    def test_plot_mv_voltages(self):
        plt.ion()
        # if not already done so, analyse grid
        try:
            if self.results.pfa_v_mag_pu is None:
                self.edisgo.analyze()
        except AttributeError:
            self.edisgo.analyze()
        except ValueError:
            pass
        # plot mv voltages
        self.edisgo.plot_mv_voltages()
        plt.close('all')

    def test_plot_mv_line_loading(self):
        # if not already done so, analyse grid
        plt.ion()
        try:
            if self.edisgo.results.i_res is None:
                self.edisgo.analyze()
        except AttributeError:
            self.edisgo.analyze()
        # plot mv line loading
        self.edisgo.plot_mv_line_loading()
        plt.close('all')

    def test_plot_mv_grid_expansion_costs(self):
        plt.ion()
        try:
            if self.edisgo.results.grid_expansion_costs is None:
                self.edisgo.reinforce()
        except AttributeError:
            self.edisgo.reinforce()
        # plot grid expansion costs
        self.edisgo.plot_mv_grid_expansion_costs()
        plt.close('all')

    def test_plot_mv_storage_integration(self):
        plt.ion()
        self.edisgo.topology.add_storage_unit(1, 'Bus_BranchTee_MVGrid_1_8',
                                              0.3)
        self.edisgo.topology.add_storage_unit(1, 'Bus_BranchTee_MVGrid_1_8',
                                              0.6)
        self.edisgo.topology.add_storage_unit(1, 'Bus_BranchTee_MVGrid_1_10',
                                              0.3)
        self.edisgo.plot_mv_storage_integration()
        plt.close('all')
        self.edisgo.topology.remove_storage('StorageUnit_MVGrid_1_1')
        self.edisgo.topology.remove_storage('StorageUnit_MVGrid_1_2')
        self.edisgo.topology.remove_storage('StorageUnit_MVGrid_1_3')

    def test_histogramm_voltage(self):
        plt.ion()
        # if not already done so, analyse grid
        try:
            if self.results.pfa_v_mag_pu is None:
                self.edisgo.analyze()
        except AttributeError:
            self.edisgo.analyze()

        self.edisgo.histogram_voltage()
        plt.close('all')

    def test_histogramm_relative_line_load(self):
        plt.ion()
        try:
            if self.edisgo.results.i_res is None:
                self.edisgo.analyze()
        except AttributeError:
            self.edisgo.analyze()

        self.edisgo.histogram_relative_line_load()
        plt.close('all')
