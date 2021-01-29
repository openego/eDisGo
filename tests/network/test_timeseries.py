import os
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from math import tan, acos
import pytest
import shutil
import numpy as np

from edisgo.network.topology import Topology
from edisgo.tools.config import Config
from edisgo.network import timeseries
from edisgo.io import ding0_import


class Testget_component_timeseries:

    @classmethod
    def setup_class(self):
        self.topology = Topology()
        self.timeseries = timeseries.TimeSeries()
        self.config = Config()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

    def test_to_csv(self):
        cur_dir = os.getcwd()
        timeseries.get_component_timeseries(edisgo_obj=self, mode='worst-case')
        self.timeseries.to_csv(cur_dir)
        #create edisgo obj to compare
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network_1')
        edisgo = pd.DataFrame()
        edisgo.topology = Topology()
        edisgo.timeseries = timeseries.TimeSeries()
        edisgo.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, edisgo)
        timeseries.get_component_timeseries(
            edisgo, mode='manual',
            timeindex=pd.read_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_active_power.csv'),
                index_col=0).index,
            loads_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_active_power.csv'),
                index_col=0),
            loads_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'loads_reactive_power.csv'), index_col=0),
            generators_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'generators_active_power.csv'), index_col=0),
            generators_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'generators_reactive_power.csv'), index_col=0),
            storage_units_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'storage_units_active_power.csv'), index_col=0),
            storage_units_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'storage_units_reactive_power.csv'), index_col=0)
        )
        # check if timeseries are the same
        assert np.isclose(self.timeseries.loads_active_power,
                          edisgo.timeseries.loads_active_power).all()
        assert np.isclose(self.timeseries.loads_reactive_power,
                          edisgo.timeseries.loads_reactive_power).all()
        assert np.isclose(self.timeseries.generators_active_power,
                          edisgo.timeseries.generators_active_power).all()
        assert np.isclose(self.timeseries.generators_reactive_power,
                          edisgo.timeseries.generators_reactive_power).all()
        assert np.isclose(self.timeseries.storage_units_active_power,
                          edisgo.timeseries.storage_units_active_power).all()
        assert np.isclose(self.timeseries.storage_units_reactive_power,
                          edisgo.timeseries.storage_units_reactive_power).all()
        # delete folder
        # Todo: check files before rmtree?
        shutil.rmtree(os.path.join(cur_dir, 'timeseries'), ignore_errors=True)
        self.timeseries = timeseries.TimeSeries()

    def test_timeseries_imported(self):
        # test storage ts
        storage_1 = self.topology.add_storage_unit(
            'Bus_MVStation_1', 0.3)
        storage_2 = self.topology.add_storage_unit(
            'Bus_GeneratorFluctuating_2', 0.45)
        storage_3 = self.topology.add_storage_unit(
            'Bus_BranchTee_LVGrid_1_10', 0.05)

        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'other': [0.775]*8760},
                                           index=timeindex)
        # test error raising in case of missing ts for dispatchable gens
        msg = \
            'Your input for "timeseries_generation_dispatchable" is not valid.'
        with pytest.raises(ValueError, match=msg):
            timeseries.get_component_timeseries(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb')
        # test error raising in case of missing ts for loads
        msg = 'Your input for "timeseries_load" is not valid.'
        with pytest.raises(ValueError, match=msg):
            timeseries.get_component_timeseries(edisgo_obj=self,
                  timeseries_generation_fluctuating='oedb',
                  timeseries_generation_dispatchable=ts_gen_dispatchable)

        msg = "No timeseries for storage units provided."
        with pytest.raises(ValueError, match=msg):
            timeseries.get_component_timeseries(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb',
                              timeseries_generation_dispatchable=ts_gen_dispatchable,
                              timeseries_load='demandlib')

        msg = "Columns or indices of inserted storage timeseries do not match " \
              "topology and timeindex."
        with pytest.raises(ValueError, match=msg):
            timeseries.get_component_timeseries(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb',
                              timeseries_generation_dispatchable=ts_gen_dispatchable,
                              timeseries_load='demandlib',
                              timeseries_storage_units=pd.DataFrame())

        storage_ts = pd.concat([self.topology.storage_units_df.p_nom]*8760,
                               axis=1, keys=timeindex).T
        timeseries.get_component_timeseries(edisgo_obj=self,
                          timeseries_generation_fluctuating='oedb',
                          timeseries_generation_dispatchable=ts_gen_dispatchable,
                          timeseries_load='demandlib',
                          timeseries_storage_units=storage_ts)

        #Todo: test with inserted reactive generation and/or reactive load

        # remove storages
        self.topology.remove_storage(storage_1)
        self.topology.remove_storage(storage_2)
        self.topology.remove_storage(storage_3)

    def test_import_load_timeseries(self):
        with pytest.raises(NotImplementedError):
            timeseries.import_load_timeseries(self.config, '')
        timeindex = pd.date_range('1/1/2018', periods=8760, freq='H')
        load = timeseries.import_load_timeseries(self.config, 'demandlib',
                                      timeindex[0].year)
        assert (load.columns == ['retail', 'residential',
                                 'agricultural', 'industrial']).all()
        assert load.loc[timeindex[453], 'retail'] == 8.335076810751597e-05
        assert load.loc[timeindex[13], 'residential'] == 0.00017315167492271323
        assert load.loc[timeindex[6328], 'agricultural'] == \
               0.00010134645909959844
        assert load.loc[timeindex[4325], 'industrial'] == 9.91768322919766e-05

    def test_worst_case(self):
        """Test creation of worst case time series"""
        # test storage ts
        storage_1 = self.topology.add_storage_unit(
            'Bus_MVStation_1', 0.3)
        # storage_2 = self.topology.add_storage_unit(
        #     'Bus_GeneratorFluctuating_2', 0.45)
        storage_3 = self.topology.add_storage_unit(
            'Bus_BranchTee_LVGrid_1_10', 0.05)

        timeseries.get_component_timeseries(edisgo_obj=self,
                                            mode='worst-case')

        # check type
        assert isinstance(
            self.timeseries.generators_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.generators_reactive_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_reactive_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.storage_units_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.storage_units_reactive_power, pd.DataFrame)

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
        number_of_cols = len(self.topology.storage_units_df.index)
        assert self.timeseries.storage_units_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.storage_units_reactive_power.shape == (
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

        storage = storage_1 # storage, mv
        exp = pd.Series(data=[1 * 0.3, -1 * 0.3],
                        name=storage, index=self.timeseries.timeindex)

        assert_series_equal(
            self.timeseries.storage_units_active_power.loc[:, storage], exp,
            check_exact=False, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.storage_units_reactive_power.loc[:, storage],
            exp * pf, check_exact=False, check_dtype=False)

        storage = storage_3 # storage, lv
        exp = pd.Series(data=[1 * 0.05, -1 * 0.05],
                        name=storage, index=self.timeseries.timeindex)

        assert_series_equal(
            self.timeseries.storage_units_active_power.loc[:, storage], exp,
            check_exact=False, check_dtype=False)
        pf = -tan(acos(0.95))
        assert_series_equal(
            self.timeseries.storage_units_reactive_power.loc[:, storage],
            exp * pf, check_exact=False, check_dtype=False)

        # remove storages
        self.topology.remove_storage(storage_1)
        #self.topology.remove_storage(storage_2)
        self.topology.remove_storage(storage_3)

        # test for only feed-in case
        timeseries.get_component_timeseries(edisgo_obj=self,
                                            mode='worst-case-feedin')

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
        timeseries.get_component_timeseries(edisgo_obj=self,
                                            mode='worst-case-load')

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
            timeseries._worst_case_generation(self, modes=None)
        self.topology._generators_df.at[gen, 'bus'] = val_pre
        gen = 'GeneratorFluctuating_24'
        val_pre = self.topology._generators_df.at[gen, 'p_nom']
        self.topology._generators_df.at[gen, 'p_nom'] = None
        with pytest.raises(AttributeError, match=gen):
            timeseries._worst_case_generation(self, modes=None)
        self.topology._generators_df.at[gen, 'p_nom'] = val_pre
        load = 'Load_agricultural_LVGrid_1_1'
        val_pre = self.topology._loads_df.at[load, 'peak_load']
        self.topology._loads_df.at[load, 'peak_load'] = None
        with pytest.raises(AttributeError, match=load):
            timeseries._worst_case_load(self, modes=None)
        self.topology._loads_df.at[load, 'peak_load'] = val_pre

        # test no other generators

    def test_add_loads_timeseries(self):
        """Test method add_loads_timeseries"""
        peak_load = 2.3
        annual_consumption = 3.4
        num_loads = len(self.topology.loads_df)
        # add single load for which timeseries is added
        # test worst-case
        timeseries.get_component_timeseries(edisgo_obj=self, mode='worst-case')
        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        timeseries.add_loads_timeseries(self, load_name)
        active_power_new_load = \
            self.timeseries.loads_active_power.loc[:,
                ['Load_retail_MVGrid_1_4']]
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        assert (self.timeseries.loads_active_power.shape == (2, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (2, num_loads+1))
        assert (active_power_new_load.index == timeindex).all()
        assert np.isclose(
            active_power_new_load.loc[timeindex[0], load_name],
            (0.15*peak_load))
        assert np.isclose(
            active_power_new_load.loc[timeindex[1], load_name],
            peak_load)
        self.topology.remove_load(load_name)

        # test manual
        timeindex = pd.date_range('1/1/2018', periods=24, freq='H')
        generators_active_power, generators_reactive_power, \
            loads_active_power, loads_reactive_power, \
            storage_units_active_power, storage_units_reactive_power = \
            self.create_random_timeseries_for_topology(timeindex)

        timeseries.get_component_timeseries(
            edisgo_obj=self, mode='manual', timeindex=timeindex,
            loads_active_power=loads_active_power,
            loads_reactive_power=loads_reactive_power,
            generators_active_power=generators_active_power,
            generators_reactive_power=generators_reactive_power,
            storage_units_active_power=storage_units_active_power,
            storage_units_reactive_power=storage_units_reactive_power)

        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        new_load_active_power = pd.DataFrame(
            index=timeindex, columns=[load_name],
            data=([peak_load] * len(timeindex)))
        new_load_reactive_power = pd.DataFrame(
            index=timeindex, columns=[load_name],
            data=([peak_load*0.5] * len(timeindex)))
        timeseries.add_loads_timeseries(self, load_name,
                                 loads_active_power=new_load_active_power,
                                 loads_reactive_power=new_load_reactive_power)
        active_power = \
            self.timeseries.loads_active_power[load_name]
        reactive_power = \
            self.timeseries.loads_reactive_power[load_name]
        assert (active_power.values == peak_load).all()
        assert (reactive_power.values == peak_load * 0.5).all()
        assert (self.timeseries.loads_active_power.shape == (24, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (24, num_loads + 1))

        self.topology.remove_load(load_name)

        # test import timeseries from dbs
        timeindex = pd.date_range('1/1/2011', periods=24, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'other': [0.775] * 24},
                                           index=timeindex)
        storage_units_active_power.index = timeindex
        timeseries.get_component_timeseries(timeindex=timeindex,
            edisgo_obj=self, timeseries_generation_fluctuating='oedb',
            timeseries_generation_dispatchable=ts_gen_dispatchable,
            timeseries_load='demandlib',
            timeseries_storage_units=storage_units_active_power)

        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        timeseries.add_loads_timeseries(self, load_name)
        active_power = \
            self.timeseries.loads_active_power[load_name]
        reactive_power = \
            self.timeseries.loads_reactive_power[load_name]
        assert np.isclose(active_power.iloc[4],
                          (4.150392788534633e-05*annual_consumption))
        assert np.isclose(reactive_power.iloc[13],
                          (7.937985538711569e-05 * annual_consumption *
                           tan(acos(0.9))))

        assert (self.timeseries.loads_active_power.shape == (24, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (24, num_loads+1))
        self.topology.remove_load(load_name)
        # Todo: add more than one load

    def test_add_generators_timeseries(self):
        """Test add_generators_timeseries method"""
        # TEST WORST-CASE
        timeseries.get_component_timeseries(edisgo_obj=self, mode='worst-case')
        num_gens = len(self.topology.generators_df)
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        # add single generator
        p_nom = 1.7
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                    bus="Bus_BranchTee_LVGrid_1_7",
                                    generator_type='solar')
        timeseries.add_generators_timeseries(self, gen_name)
        assert self.timeseries.generators_active_power.shape == (2, num_gens+1)
        assert self.timeseries.generators_reactive_power.shape == \
            (2, num_gens+1)
        assert \
            (self.timeseries.generators_active_power.index == timeindex).all()
        assert (self.timeseries.generators_active_power.loc[
            timeindex, gen_name].values == [0.85*p_nom, 0]).all()
        assert np.isclose(self.timeseries.generators_reactive_power.loc[
            timeindex, gen_name], [-tan(acos(0.95))*0.85*p_nom, 0]).all()
        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        timeseries.add_generators_timeseries(self, [gen_name2, gen_name3])
        # check expected values
        assert self.timeseries.generators_active_power.shape == (2, num_gens+3)
        assert self.timeseries.generators_reactive_power.shape == (
            2, num_gens + 3)
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [[p_nom2, p_nom3], [0, 0]]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [[-p_nom2*tan(acos(0.9)), -p_nom3*tan(acos(0.95))], [0, 0]]).all()
        # remove added generators
        self.topology.remove_generator(gen_name)
        self.topology.remove_generator(gen_name2)
        self.topology.remove_generator(gen_name3)
        # TEST MANUAL
        timeindex = pd.date_range('1/1/2018', periods=24, freq='H')
        generators_active_power, generators_reactive_power, \
            loads_active_power, loads_reactive_power, \
            storage_units_active_power, storage_units_reactive_power = \
            self.create_random_timeseries_for_topology(timeindex)

        timeseries.get_component_timeseries(
            edisgo_obj=self, mode='manual', timeindex=timeindex,
            loads_active_power=loads_active_power,
            loads_reactive_power=loads_reactive_power,
            generators_active_power=generators_active_power,
            generators_reactive_power=generators_reactive_power,
            storage_units_active_power=storage_units_active_power,
            storage_units_reactive_power=storage_units_reactive_power)
        # add single mv solar generator
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                               bus="Bus_BranchTee_LVGrid_1_7",
                                               generator_type='solar')
        new_gen_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name],
            data=([p_nom * 0.97] * len(timeindex)))
        new_gen_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name],
            data=([p_nom * 0.5] * len(timeindex)))
        timeseries.add_generators_timeseries(self,
            gen_name, generators_active_power=new_gen_active_power,
            generators_reactive_power=new_gen_reactive_power)
        # check expected values
        assert self.timeseries.generators_active_power.shape == (
            24, num_gens + 1)
        assert self.timeseries.generators_reactive_power.shape == \
            (24, num_gens + 1)
        assert \
            (self.timeseries.generators_active_power.index == timeindex).all()
        assert (self.timeseries.generators_active_power.loc[
                    timeindex, gen_name].values == 0.97 * p_nom).all()
        assert np.isclose(self.timeseries.generators_reactive_power.loc[
                              timeindex, gen_name], p_nom*0.5).all()
        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        new_gens_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.97], [p_nom3 * 0.98]])
                  .repeat(len(timeindex), axis=1).T))
        new_gens_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.5], [p_nom3 * 0.4]])
                  .repeat(len(timeindex), axis=1).T))
        timeseries.add_generators_timeseries(self,
            [gen_name2, gen_name3],
            generators_active_power=new_gens_active_power,
            generators_reactive_power=new_gens_reactive_power)
        # check expected values
        assert self.timeseries.generators_active_power.shape == (
            24, num_gens + 3)
        assert self.timeseries.generators_reactive_power.shape == (
            24, num_gens + 3)
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.97, p_nom3*0.98]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.5, p_nom3*0.4]).all()
        # remove added generators
        self.topology.remove_generator(gen_name)
        self.topology.remove_generator(gen_name2)
        self.topology.remove_generator(gen_name3)
        # TEST TIMESERIES IMPORT
        # test import timeseries from dbs
        timeindex = pd.date_range('1/1/2011', periods=24, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'other': [0.775] * 24},
                                           index=timeindex)
        storage_units_active_power.index = timeindex
        timeseries.get_component_timeseries(timeindex=timeindex,
                                edisgo_obj=self,
                                timeseries_generation_fluctuating='oedb',
                                timeseries_generation_dispatchable=ts_gen_dispatchable,
                                timeseries_load='demandlib',
                                timeseries_storage_units=storage_units_active_power)

        # add single mv solar generator
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                               bus="Bus_BranchTee_LVGrid_1_7",
                                               generator_type='solar',
                                               weather_cell_id=1122075)
        timeseries.add_generators_timeseries(self, gen_name)
        assert (self.timeseries.generators_active_power.shape == (
                24, num_gens + 1))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 1))
        #Todo: check values

        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        new_gens_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.97], [p_nom3 * 0.98]])
                  .repeat(len(timeindex), axis=1).T))
        timeseries.add_generators_timeseries(self,
            [gen_name2, gen_name3])
        assert (self.timeseries.generators_active_power.shape == (
            24, num_gens + 3))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 3))
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.775, p_nom3*0.775]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [-tan(acos(0.9))*p_nom2*0.775, -tan(acos(0.95))*p_nom3*0.775]).all()
        # check values when reactive power is inserted as timeseries
        new_gens_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.54], [p_nom3 * 0.45]])
                  .repeat(len(timeindex), axis=1).T))
        timeseries.add_generators_timeseries(self, [gen_name2, gen_name3],
            timeseries_generation_dispatchable=new_gens_active_power,
            generation_reactive_power=new_gens_reactive_power)
        assert (self.timeseries.generators_active_power.shape == (
            24, num_gens + 3))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 3))
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2 * 0.775, p_nom3 * 0.775]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2 * 0.54, p_nom3 * 0.45]).all()
        # remove added generators
        self.topology.remove_generator(gen_name)
        self.topology.remove_generator(gen_name2)
        self.topology.remove_generator(gen_name3)

    def test_add_storage_unit_timeseries(self):
        """Test add_storage_unit_timeseries method"""
        # TEST WORST-CASE
        # add single storage unit
        num_storage_units = len(self.topology.storage_units_df)
        timeseries.get_component_timeseries(edisgo_obj=self, mode='worst-case')
        p_nom = 2.1
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        storage_name = self.topology.add_storage_unit(
            bus='Bus_MVStation_1', p_nom=p_nom)
        timeseries.add_storage_units_timeseries(self, storage_name)
        assert (self.timeseries.storage_units_active_power.index ==
                timeindex).all()
        assert (self.timeseries.storage_units_reactive_power.index ==
                timeindex).all()
        assert (self.timeseries.storage_units_active_power.shape ==
                (len(timeindex), num_storage_units+1))
        assert (self.timeseries.storage_units_reactive_power.shape ==
                (len(timeindex), num_storage_units + 1))
        assert (self.timeseries.storage_units_active_power.loc[
                    timeindex, storage_name].values == [p_nom, -p_nom]).all()
        assert (np.isclose(self.timeseries.storage_units_reactive_power.loc[
                    timeindex, storage_name].values,
                    [-p_nom*tan(acos(0.9)), p_nom*tan(acos(0.9))])).all()
        # add two storage units
        p_nom2 = 1.3
        storage_name2 = self.topology.add_storage_unit(
            bus='Bus_BranchTee_LVGrid_1_13', p_nom=p_nom2)
        p_nom3 = 3.12
        storage_name3 = self.topology.add_storage_unit(
            bus='BusBar_MVGrid_1_LVGrid_6_MV', p_nom=p_nom3)
        timeseries.add_storage_units_timeseries(self,
                                                [storage_name2, storage_name3])
        assert (self.timeseries.storage_units_active_power.shape ==
                (len(timeindex), num_storage_units + 3))
        assert (self.timeseries.storage_units_reactive_power.shape ==
                (len(timeindex), num_storage_units + 3))
        assert np.isclose(
            self.timeseries.storage_units_active_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [[p_nom2, p_nom3], [-p_nom2, -p_nom3]]).all()
        assert np.isclose(
            self.timeseries.storage_units_reactive_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [[-tan(acos(0.95))*p_nom2, -tan(acos(0.9))*p_nom3],
             [tan(acos(0.95))*p_nom2, tan(acos(0.9))*p_nom3]]).all()
        # remove storages
        self.topology.remove_storage(storage_name)
        self.topology.remove_storage(storage_name2)
        self.topology.remove_storage(storage_name3)
        # TEST MANUAL
        timeindex = pd.date_range('1/1/2018', periods=24, freq='H')
        generators_active_power, generators_reactive_power, \
        loads_active_power, loads_reactive_power, \
        storage_units_active_power, storage_units_reactive_power = \
            self.create_random_timeseries_for_topology(timeindex)

        timeseries.get_component_timeseries(
            edisgo_obj=self, mode='manual', timeindex=timeindex,
            loads_active_power=loads_active_power,
            loads_reactive_power=loads_reactive_power,
            generators_active_power=generators_active_power,
            generators_reactive_power=generators_reactive_power,
            storage_units_active_power=storage_units_active_power,
            storage_units_reactive_power=storage_units_reactive_power)
        # add single mv solar generator
        storage_name = self.topology.add_storage_unit(
            bus='Bus_MVStation_1', p_nom=p_nom)
        new_storage_active_power = pd.DataFrame(
            index=timeindex, columns=[storage_name],
            data=([p_nom * 0.97] * len(timeindex)))
        new_storage_reactive_power = pd.DataFrame(
            index=timeindex, columns=[storage_name],
            data=([p_nom * 0.5] * len(timeindex)))
        timeseries.add_storage_units_timeseries(self,
            storage_name, storage_units_active_power=new_storage_active_power,
            storage_units_reactive_power=new_storage_reactive_power)
        # check expected values
        assert self.timeseries.storage_units_active_power.shape == (
            24, num_storage_units + 1)
        assert self.timeseries.storage_units_reactive_power.shape == \
               (24, num_storage_units + 1)
        assert \
            (self.timeseries.storage_units_active_power.index == timeindex).all()
        assert (self.timeseries.storage_units_active_power.loc[
                    timeindex, storage_name].values == 0.97 * p_nom).all()
        assert np.isclose(self.timeseries.storage_units_reactive_power.loc[
                              timeindex, storage_name], p_nom * 0.5).all()
        # add multiple generators and check
        p_nom2 = 1.3
        storage_name2 = self.topology.add_storage_unit(
            bus='Bus_BranchTee_LVGrid_1_13', p_nom=p_nom2)
        p_nom3 = 3.12
        storage_name3 = self.topology.add_storage_unit(
            bus='BusBar_MVGrid_1_LVGrid_6_MV', p_nom=p_nom3)

        new_storages_active_power = pd.DataFrame(
            index=timeindex, columns=[storage_name2, storage_name3],
            data=(np.array([[p_nom2 * 0.97], [p_nom3 * 0.98]])
                  .repeat(len(timeindex), axis=1).T))
        new_storages_reactive_power = pd.DataFrame(
            index=timeindex, columns=[storage_name2, storage_name3],
            data=(np.array([[p_nom2 * 0.5], [p_nom3 * 0.4]])
                  .repeat(len(timeindex), axis=1).T))
        timeseries.add_storage_units_timeseries(self,
            [storage_name2, storage_name3],
            storage_units_active_power=new_storages_active_power,
            storage_units_reactive_power=new_storages_reactive_power)
        # check expected values
        assert self.timeseries.storage_units_active_power.shape == (
            24, num_storage_units + 3)
        assert self.timeseries.storage_units_reactive_power.shape == (
            24, num_storage_units + 3)
        assert np.isclose(
            self.timeseries.storage_units_active_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [p_nom2 * 0.97, p_nom3 * 0.98]).all()
        assert np.isclose(
            self.timeseries.storage_units_reactive_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [p_nom2 * 0.5, p_nom3 * 0.4]).all()
        # remove added generators
        self.topology.remove_storage(storage_name)
        self.topology.remove_storage(storage_name2)
        self.topology.remove_storage(storage_name3)
        # TEST TIMESERIES IMPORT
        # test import timeseries from dbs
        timeindex = pd.date_range('1/1/2011', periods=24, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'other': [0.775] * 24},
                                           index=timeindex)
        # reindex timeseries
        storage_units_active_power = \
            storage_units_active_power.set_index(timeindex)
        new_storage_active_power = \
            new_storage_active_power.set_index(timeindex)
        new_storage_reactive_power = \
            new_storage_reactive_power.set_index(timeindex)
        new_storages_active_power = \
            new_storages_active_power.set_index(timeindex)
        new_storages_reactive_power = \
            new_storages_reactive_power.set_index(timeindex)
        timeseries.get_component_timeseries(
            timeindex=timeindex,
            edisgo_obj=self,
            timeseries_generation_fluctuating='oedb',
            timeseries_generation_dispatchable=ts_gen_dispatchable,
            timeseries_load='demandlib',
            timeseries_storage_units=storage_units_active_power)

        # add single mv solar generator
        storage_name = self.topology.add_storage_unit(
            bus='Bus_MVStation_1', p_nom=p_nom)

        timeseries.add_storage_units_timeseries(self,
            storage_name, timeseries_storage_units=new_storage_active_power,
            timeseries_storage_units_reactive_power=new_storage_reactive_power)
        assert (self.timeseries.storage_units_active_power.shape == (
            24, num_storage_units + 1))
        assert (self.timeseries.storage_units_reactive_power.shape ==
                (24, num_storage_units + 1))
        assert_frame_equal(self.timeseries.storage_units_active_power.loc[
                timeindex, [storage_name]], new_storage_active_power)
        assert_frame_equal(self.timeseries.storage_units_reactive_power.loc[
                timeindex, [storage_name]], new_storage_reactive_power)

        # add multiple generators and check
        p_nom2 = 1.3
        storage_name2 = self.topology.add_storage_unit(
            bus='Bus_BranchTee_LVGrid_1_13', p_nom=p_nom2)
        p_nom3 = 3.12
        storage_name3 = self.topology.add_storage_unit(
            bus='BusBar_MVGrid_1_LVGrid_6_MV', p_nom=p_nom3)

        timeseries.add_storage_units_timeseries(self,
            [storage_name2, storage_name3],
            timeseries_storage_units=new_storages_active_power)

        assert (self.timeseries.storage_units_active_power.shape == (
            24, num_storage_units + 3))
        assert (self.timeseries.storage_units_reactive_power.shape ==
                (24, num_storage_units + 3))
        assert np.isclose(
            self.timeseries.storage_units_active_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [p_nom2 * 0.97, p_nom3 * 0.98]).all()
        assert np.isclose(
            self.timeseries.storage_units_reactive_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [-tan(acos(0.95)) * p_nom2 * 0.97,
             -tan(acos(0.9)) * p_nom3 * 0.98]).all()
        # check values when reactive power is inserted as timeseries
        timeseries.add_storage_units_timeseries(self,
                                                [storage_name2, storage_name3],
                                      timeseries_storage_units=
                                      new_storages_active_power,
                                      timeseries_storage_units_reactive_power=
                                      new_storages_reactive_power)
        assert (self.timeseries.storage_units_active_power.shape == (
            24, num_storage_units + 3))
        assert (self.timeseries.storage_units_reactive_power.shape ==
                (24, num_storage_units + 3))
        assert np.isclose(
            self.timeseries.storage_units_active_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [p_nom2 * 0.97, p_nom3 * 0.98]).all()
        assert np.isclose(
            self.timeseries.storage_units_reactive_power.loc[
                timeindex, [storage_name2, storage_name3]].values,
            [p_nom2 * 0.5, p_nom3 * 0.4]).all()
        # remove added generators
        self.topology.remove_storage(storage_name)
        self.topology.remove_storage(storage_name2)
        self.topology.remove_storage(storage_name3)

    def test_check_timeseries_for_index_and_cols(self):
        """Test check_timeseries_for_index_and_cols method"""
        timeindex = pd.date_range('1/1/2017', periods=13, freq='H')
        timeseries.get_component_timeseries(
            edisgo_obj=self, mode='manual', timeindex=timeindex)
        added_comps = ['Comp_1', 'Comp_2']
        timeseries_with_wrong_timeindex = pd.DataFrame(
            index=timeindex[0:12], columns=added_comps,
            data=np.random.rand(12, len(added_comps)))
        #Todo: check what happens with assertion. Why are strings not the same?
        msg = "Inserted timeseries for the following components have the a " \
              "wrong time index:"
        with pytest.raises(ValueError, match=msg):
            timeseries.check_timeseries_for_index_and_cols(self,
                timeseries_with_wrong_timeindex, added_comps)
        timeseries_with_wrong_comp_names = pd.DataFrame(
            index=timeindex, columns=['Comp_1'],
            data=np.random.rand(13, 1))
        msg = "Columns of inserted timeseries are not the same " \
              "as names of components to be added. Timeseries " \
              "for the following components were tried to be " \
              "added:"
        with pytest.raises(ValueError, match=msg):
            timeseries.check_timeseries_for_index_and_cols(self,
                timeseries_with_wrong_comp_names, added_comps)

    def create_random_timeseries_for_topology(self, timeindex):
        # create random timeseries
        load_names = self.topology.loads_df.index
        loads_active_power = \
            pd.DataFrame(index=timeindex, columns=load_names,
                         data=np.multiply(np.random.rand(len(timeindex),
                                                         len(load_names)),
                                      ([self.topology.loads_df.peak_load] *
                                       len(timeindex))))
        loads_reactive_power = \
            pd.DataFrame(index=timeindex, columns=load_names,
                         data=np.multiply(np.random.rand(len(timeindex),
                                                         len(load_names)),
                                      ([self.topology.loads_df.peak_load] *
                                       len(timeindex))))
        generator_names = self.topology.generators_df.index
        generators_active_power = \
            pd.DataFrame(index=timeindex, columns=generator_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(generator_names)),
                             ([self.topology.generators_df.p_nom] *
                              len(timeindex))))
        generators_reactive_power = \
            pd.DataFrame(index=timeindex, columns=generator_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(generator_names)),
                             ([self.topology.generators_df.p_nom] *
                              len(timeindex))))
        storage_names = self.topology.storage_units_df.index
        storage_units_active_power = \
            pd.DataFrame(index=timeindex, columns=storage_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(storage_names)),
                             ([self.topology.storage_units_df.p_nom] *
                              len(timeindex))))
        storage_units_reactive_power = \
            pd.DataFrame(index=timeindex, columns=storage_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(storage_names)),
                             ([self.topology.storage_units_df.p_nom] *
                              len(timeindex))))
        return generators_active_power, generators_reactive_power, \
               loads_active_power, loads_reactive_power, \
               storage_units_active_power, storage_units_reactive_power

    def test_drop_existing_component_timeseries(self):
        """Test for _drop_existing_timseries_method"""
        storage_1 = self.topology.add_storage_unit(
            'Bus_MVStation_1', 0.3)
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        timeseries.get_component_timeseries(edisgo_obj=self, mode='worst-case')
        # test drop load timeseries
        assert hasattr(self.timeseries.loads_active_power,
                       'Load_agricultural_LVGrid_1_1')
        assert hasattr(self.timeseries.loads_reactive_power,
                       'Load_agricultural_LVGrid_1_1')
        timeseries._drop_existing_component_timeseries(
            self, 'loads', ['Load_agricultural_LVGrid_1_1'])
        with pytest.raises(KeyError):
            self.timeseries.loads_active_power.loc[
                timeindex, 'Load_agricultural_LVGrid_1_1']
        with pytest.raises(KeyError):
            self.timeseries.loads_reactive_power.loc[
                timeindex, 'Load_agricultural_LVGrid_1_1']
        # test drop generators timeseries
        assert hasattr(self.timeseries.generators_active_power,
                       'GeneratorFluctuating_7')
        assert hasattr(self.timeseries.generators_reactive_power,
                       'GeneratorFluctuating_7')
        timeseries._drop_existing_component_timeseries(
            self, 'generators', 'GeneratorFluctuating_7')
        with pytest.raises(KeyError):
            self.timeseries.generators_active_power.loc[
                timeindex, 'GeneratorFluctuating_7']
        with pytest.raises(KeyError):
            self.timeseries.generators_reactive_power.loc[
                timeindex, 'GeneratorFluctuating_7']
        # test drop storage units timeseries
        assert hasattr(self.timeseries.storage_units_active_power,
                       storage_1)
        assert hasattr(self.timeseries.storage_units_reactive_power,
                       storage_1)
        timeseries._drop_existing_component_timeseries(
            self, 'storage_units', storage_1)
        with pytest.raises(KeyError):
            self.timeseries.storage_units_active_power.loc[
                timeindex, storage_1]
        with pytest.raises(KeyError):
            self.timeseries.storage_units_reactive_power.loc[
                timeindex, storage_1]
        self.topology.remove_storage(storage_1)


class TestReactivePowerTimeSeriesFunctions:

    @classmethod
    def setup_class(self):
        self.topology = Topology()
        self.timeseries = timeseries.TimeSeries()
        self.config = Config()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path,
                                       self)
        self.timeseries.timeindex = pd.date_range(
            '1/1/1970', periods=2, freq='H')

    def test_set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self):

        # test for component_type="generators"
        comp_mv_1 = "Generator_1"
        comp_mv_2 = "GeneratorFluctuating_2"
        comp_lv_1 = "GeneratorFluctuating_25"
        comp_lv_2 = "GeneratorFluctuating_26"

        active_power_ts = pd.DataFrame(
            data={
                comp_mv_1: [0.5, 1.5],
                comp_mv_2: [2.5, 3.5],
                comp_lv_1: [0.1, 0.0],
                comp_lv_2: [0.15, 0.07]
            },
            index=self.timeseries.timeindex)
        self.timeseries.generators_active_power = active_power_ts

        timeseries._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self,
            self.topology.generators_df.loc[
                [comp_mv_1, comp_mv_2, comp_lv_1], :],
            "generators"
        )

        assert self.timeseries.generators_reactive_power.shape == (2, 3)
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                :, [comp_mv_1, comp_mv_2]].values,
            active_power_ts.loc[
                :, [comp_mv_1, comp_mv_2]].values * -0.484322).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
            :, comp_lv_1].values,
            active_power_ts.loc[:, comp_lv_1] * -0.328684).all()

        timeseries._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self,
            self.topology.generators_df.loc[[comp_lv_2], :],
            "generators"
        )

        # check new time series and that old reactive power time series
        # remained unchanged
        assert self.timeseries.generators_reactive_power.shape == (2, 4)
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
            :, [comp_mv_1, comp_mv_2]].values,
            active_power_ts.loc[
            :, [comp_mv_1, comp_mv_2]].values * -0.484322).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
            :, [comp_lv_1, comp_lv_2]].values,
            active_power_ts.loc[:, [comp_lv_1, comp_lv_2]] * -0.328684).all()

        # test for component_type="loads"
        comp_mv_1 = "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"
        comp_lv_1 = "Load_residential_LVGrid_7_2"
        comp_lv_2 = "Load_agricultural_LVGrid_8_1"

        active_power_ts = pd.DataFrame(
            data={
                comp_mv_1: [0.5, 1.5],
                comp_lv_1: [0.1, 0.0],
                comp_lv_2: [0.15, 0.07]
            },
            index=self.timeseries.timeindex)
        self.timeseries.loads_active_power = active_power_ts

        timeseries._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self,
            self.topology.loads_df.loc[[comp_mv_1, comp_lv_1], :],
            "loads"
        )

        assert self.timeseries.loads_reactive_power.shape == (2, 2)
        assert np.isclose(
            self.timeseries.loads_reactive_power.loc[
                :, [comp_mv_1]].values,
            active_power_ts.loc[
                :, [comp_mv_1]].values * 0.484322).all()
        assert np.isclose(
            self.timeseries.loads_reactive_power.loc[
            :, comp_lv_1].values,
            active_power_ts.loc[:, comp_lv_1] * 0.328684).all()

        timeseries._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self,
            self.topology.loads_df.loc[[comp_lv_2], :],
            "loads"
        )

        assert self.timeseries.loads_reactive_power.shape == (2, 3)
        assert np.isclose(
            self.timeseries.loads_reactive_power.loc[
            :, comp_lv_2].values,
            active_power_ts.loc[:, comp_lv_2] * 0.328684).all()

        # test for component_type="storage_units"
        comp_mv_1 = "Storage_1"

        active_power_ts = pd.DataFrame(
            data={
                comp_mv_1: [0.5, 1.5]
            },
            index=self.timeseries.timeindex)
        self.timeseries.storage_units_active_power = active_power_ts

        timeseries._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            self,
            self.topology.storage_units_df.loc[[comp_mv_1], :],
            "storage_units"
        )

        assert self.timeseries.storage_units_reactive_power.shape == (2, 1)
        assert np.isclose(
            self.timeseries.storage_units_reactive_power.loc[
            :, [comp_mv_1]].values,
            active_power_ts.loc[
            :, [comp_mv_1]].values * -0.484322).all()
