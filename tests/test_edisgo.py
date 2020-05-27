import os
import pandas as pd
import numpy as np
import pytest
import shutil
from math import tan, acos

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks
from matplotlib import pyplot as plt


class TestEDisGo:

    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path,
                             worst_case_analysis='worst-case')
        self.timesteps = pd.date_range('1/1/1970', periods=2, freq='H')

    def test_exceptions(self):
        msg = "No power flow results to check over-load for. Please perform " \
              "power flow analysis first."
        with pytest.raises(Exception, match=msg):
            checks.mv_line_load(self.edisgo)
        self.edisgo.analyze()
        msg = "Inserted grid is invalid."
        with pytest.raises(ValueError, match=msg):
            checks._station_load(self.edisgo, None)

    def test_save(self):
        cur_dir = os.getcwd()
        self.edisgo.save(cur_dir)
        # Todo: check values?
        # Todo: check files before rmtree?
        shutil.rmtree(os.path.join(cur_dir, 'results'))
        shutil.rmtree(os.path.join(cur_dir, 'topology'))
        shutil.rmtree(os.path.join(cur_dir, 'timeseries'))

    def test_crit_station(self):
        # TODO: have checks of technical constraints not require edisgo
        # object and then move this test
        # calculate results if not already existing
        if self.edisgo.results.pfa_p is None:
            self.edisgo.analyze()
        # check results
        overloaded_mv_station = checks.hv_mv_station_load(self.edisgo)
        assert overloaded_mv_station.empty
        overloaded_lv_station = checks.mv_lv_station_load(self.edisgo)
        assert(len(overloaded_lv_station) == 4)
        assert (np.isclose(
            overloaded_lv_station.at['LVGrid_1', 's_missing'],
            0.01942, atol=1e-5))
        assert (overloaded_lv_station.at[
                    'LVGrid_1', 'time_index'] == self.timesteps[1])
        assert (np.isclose(
            overloaded_lv_station.at['LVGrid_4', 's_missing'],
            0.03426, atol=1e-5))
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
        assert len(mv_crit_lines) == 4
        assert (mv_crit_lines.time_index == self.timesteps[0]).all()
        assert (np.isclose(
            mv_crit_lines.at['Line_10006', 'max_rel_overload'],
            1.16306, atol=1e-5))
        assert (np.isclose(
            mv_crit_lines.at['Line_10026', 'max_rel_overload'],
            1.06230, atol=1e-5))

    def test_analyze(self):
        if self.edisgo.results.grid_losses is None:
            self.edisgo.analyze()
        # check results
        #ToDo negative grid losses?
        assert(np.isclose(
            self.edisgo.results.grid_losses.loc[self.timesteps].values,
            np.array([[-0.19186, 0.40321], [0.41854, -0.17388]]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.hv_mv_exchanges.loc[self.timesteps].values,
            np.array([[-21.69377, 10.87843], [1.36392, 0.18510]]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.v_res.loc[
                self.timesteps, 'Bus_GeneratorFluctuating_18'].values,
            np.array([1.01699, 0.99917]),
            atol=1e-5).all())
        assert(np.isclose(
            self.edisgo.results.v_res.loc[
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
        assert len(results.grid_expansion_costs) == 12
        assert len(results.equipment_changes) == 12
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
        ts_gen_dispatchable = pd.DataFrame({'other': [0.775] * 8760},
                                           index=timeindex)
        ts_storage = pd.DataFrame({'Storage_1': [0.0] * 8760},
                                           index=timeindex)
        edisgo = EDisGo(ding0_grid=test_network_directory,
                        timeseries_generation_fluctuating='oedb',
                        timeseries_generation_dispatchable=ts_gen_dispatchable,
                        timeseries_load='demandlib',
                        timeseries_storage_units=ts_storage)
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
            if self.results.v_res is None:
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
        storage_1 = self.edisgo.topology.add_storage_unit(
            'Bus_BranchTee_MVGrid_1_8', 0.3)
        storage_2 = self.edisgo.topology.add_storage_unit(
            'Bus_BranchTee_MVGrid_1_8', 0.6)
        storage_3 = self.edisgo.topology.add_storage_unit(
            'Bus_BranchTee_MVGrid_1_10', 0.3)
        self.edisgo.plot_mv_storage_integration()
        plt.close('all')
        self.edisgo.topology.remove_storage(storage_1)
        self.edisgo.topology.remove_storage(storage_2)
        self.edisgo.topology.remove_storage(storage_3)

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

    def test_add_component(self):
        """Test add_component method"""
        # Test add bus
        num_buses = len(self.edisgo.topology.buses_df)
        bus_name = self.edisgo.add_component('Bus', name="Testbus", v_nom=20)
        assert bus_name == "Testbus"
        assert len(self.edisgo.topology.buses_df) == num_buses+1
        assert self.edisgo.topology.buses_df.loc["Testbus", 'v_nom'] == 20
        # Test add line
        num_lines = len(self.edisgo.topology.lines_df)
        line_name = self.edisgo.add_component('Line', bus0="Bus_MVStation_1",
                                  bus1="Testbus", length=0.001)
        assert line_name == "Line_Bus_MVStation_1_Testbus"
        assert len(self.edisgo.topology.lines_df) == num_lines+1
        assert self.edisgo.topology.lines_df.loc[line_name, 'bus0'] == \
               "Bus_MVStation_1"
        assert self.edisgo.topology.lines_df.loc[line_name, 'bus1'] == \
               "Testbus"
        assert self.edisgo.topology.lines_df.loc[line_name, 'length'] == 0.001
        # Test add load
        num_loads = len(self.edisgo.topology.loads_df)
        load_name = self.edisgo.add_component('Load', load_id=4, bus="Testbus",
                                              peak_load=0.2,
                                              annual_consumption=3.2,
                                              sector='residential')
        assert load_name == "Load_residential_MVGrid_1_4"
        assert len(self.edisgo.topology.loads_df) == num_loads+1
        assert self.edisgo.topology.loads_df.loc[load_name, 'bus'] == "Testbus"
        assert self.edisgo.topology.loads_df.loc[load_name, 'peak_load'] == 0.2
        assert self.edisgo.topology.loads_df.loc[
                   load_name, 'annual_consumption'] == 3.2
        assert self.edisgo.topology.loads_df.loc[load_name, 'sector'] == \
               'residential'
        index = self.edisgo.timeseries.timeindex
        assert np.isclose(self.edisgo.timeseries.loads_active_power.loc[
            index[0], load_name], 0.15*0.2)
        assert np.isclose(self.edisgo.timeseries.loads_active_power.loc[
            index[1], load_name], 0.2)
        assert np.isclose(self.edisgo.timeseries.loads_reactive_power.loc[
                              index[0], load_name], tan(acos(0.9))*0.15*0.2)
        assert np.isclose(self.edisgo.timeseries.loads_reactive_power.loc[
                              index[1], load_name], tan(acos(0.9))*0.2)
        # Todo: test other modes of timeseries (manual, None)
        # Test add generator
        num_gens = len(self.edisgo.topology.generators_df)
        gen_name = self.edisgo.add_component('Generator', generator_id=5,
                                             bus="Testbus", p_nom=2.5,
                                             generator_type='solar')
        assert gen_name == "Generator_solar_MVGrid_1_5"
        assert len(self.edisgo.topology.generators_df) == num_gens + 1
        assert self.edisgo.topology.generators_df.loc[gen_name, 'bus'] == \
               "Testbus"
        assert self.edisgo.topology.generators_df.loc[gen_name, 'p_nom'] == 2.5
        assert self.edisgo.topology.generators_df.loc[
                   gen_name, 'type'] == 'solar'
        assert np.isclose(self.edisgo.timeseries.generators_active_power.loc[
                              index[0], gen_name], 0.85*2.5)
        assert np.isclose(self.edisgo.timeseries.generators_active_power.loc[
                              index[1], gen_name], 0)
        assert np.isclose(self.edisgo.timeseries.generators_reactive_power.loc[
                              index[0], gen_name],
                          -tan(acos(0.9)) * 0.85 * 2.5)
        assert np.isclose(self.edisgo.timeseries.generators_reactive_power.loc[
                              index[1], gen_name], 0)
        # Todo: test other modes of timeseries (manual, None)
        # Test add storage unit
        num_storages = len(self.edisgo.topology.storage_units_df)
        storage_name = self.edisgo.add_component('StorageUnit',
                                             bus="Testbus", p_nom=3.1)
        assert storage_name == "StorageUnit_MVGrid_1_1"
        assert len(self.edisgo.topology.storage_units_df) == num_storages + 1
        assert self.edisgo.topology.storage_units_df.loc[storage_name, 'bus'] \
               == "Testbus"
        assert self.edisgo.topology.storage_units_df.loc[storage_name,
                                                         'p_nom'] == 3.1
        assert np.isclose(self.edisgo.timeseries.storage_units_active_power.loc[
                              index[0], storage_name], 3.1)
        assert np.isclose(self.edisgo.timeseries.storage_units_active_power.loc[
                              index[1], storage_name], -3.1)
        assert np.isclose(self.edisgo.timeseries.storage_units_reactive_power.
                          loc[index[0], storage_name], -tan(acos(0.9))*3.1)
        assert np.isclose(self.edisgo.timeseries.storage_units_reactive_power.loc[
                              index[1], storage_name], tan(acos(0.9))*3.1)
        # Todo: test other modes of timeseries (manual, None)
        # Remove test objects
        msg = "Bus Testbus is not isolated. " \
              "Remove all connected elements first to remove bus."
        with pytest.raises(AssertionError, match=msg):
            self.edisgo.remove_component('Bus', bus_name)
        msg = "Removal of line {} would create isolated node.".format(line_name)
        with pytest.raises(AssertionError, match=msg):
            self.edisgo.remove_component('Line', line_name)
        self.edisgo.remove_component('StorageUnit', storage_name)
        self.edisgo.remove_component('Load', load_name)
        self.edisgo.remove_component('Generator', gen_name)
        # Todo: check if components were removed
