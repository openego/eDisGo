import logging

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_index_equal

from edisgo import EDisGo
from edisgo.io import timeseries_import
from edisgo.tools.config import Config


class TestTimeseriesImport:
    @classmethod
    def setup_class(self):
        self.config = Config(config_path=None)

    def test__timeindex_helper_func(self):
        # test with timeindex=None and TimeSeries.timeindex not set
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        ind, ind_full = timeseries_import._timeindex_helper_func(edisgo, timeindex=None)
        timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
        assert_index_equal(ind, timeindex)
        assert_index_equal(ind_full, timeindex)

        # test with timeindex=None and TimeSeries.timeindex set
        edisgo_index = pd.date_range("1/1/2010", periods=5, freq="H")
        edisgo.set_timeindex(edisgo_index)
        ind, ind_full = timeseries_import._timeindex_helper_func(edisgo, timeindex=None)
        timeindex = pd.date_range("1/1/2010", periods=8760, freq="H")
        assert_index_equal(ind, edisgo_index)
        assert_index_equal(ind_full, timeindex)

        # test with given timeindex and leap year
        given_index = pd.date_range("1/1/2012", periods=5, freq="H")
        ind, ind_full = timeseries_import._timeindex_helper_func(
            edisgo, timeindex=given_index
        )
        timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
        assert_index_equal(ind, timeindex)
        assert_index_equal(ind_full, timeindex)

        # test with given timeindex and leap year and allowing leap year
        ind, ind_full = timeseries_import._timeindex_helper_func(
            edisgo, timeindex=given_index, allow_leap_year=True
        )
        timeindex = pd.date_range("1/1/2012", periods=8760, freq="H")
        assert_index_equal(ind, given_index)
        assert_index_equal(ind_full, timeindex)

        # test with given timeindex and no leap year
        given_index = pd.date_range("1/1/2013", periods=5, freq="H")
        ind, ind_full = timeseries_import._timeindex_helper_func(
            edisgo,
            timeindex=given_index,
        )
        timeindex = pd.date_range("1/1/2013", periods=8760, freq="H")
        assert_index_equal(ind, given_index)
        assert_index_equal(ind_full, timeindex)

    def test_feedin_oedb_legacy(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        timeindex = pd.date_range("1/1/2010", periods=3000, freq="H")
        feedin = timeseries_import.feedin_oedb_legacy(edisgo, timeindex)
        assert len(feedin["solar"][1122074]) == 3000
        assert len(feedin["solar"][1122075]) == 3000
        assert len(feedin["wind"][1122074]) == 3000
        assert len(feedin["wind"][1122075]) == 3000
        assert np.isclose(feedin["solar"][1122074][timeindex[13]], 0.074941)
        assert np.isclose(feedin["wind"][1122074][timeindex[37]], 0.039784)
        assert np.isclose(feedin["solar"][1122075][timeindex[61]], 0.423823)
        assert np.isclose(feedin["wind"][1122075][timeindex[1356]], 0.106361)

    @pytest.mark.local
    def test_feedin_oedb(self):
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        timeindex = pd.date_range("1/2/2018", periods=6, freq="H")
        edisgo_object.set_timeindex(timeindex)
        feedin_df = timeseries_import.feedin_oedb(
            edisgo_object,
            engine=pytest.engine,
        )
        assert feedin_df.shape == (6, 4)
        assert_index_equal(feedin_df.index, timeindex)

    def test_load_time_series_demandlib(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        timeindex = pd.date_range("1/1/2018", periods=7000, freq="H")
        load = timeseries_import.load_time_series_demandlib(edisgo, timeindex)
        assert (
            load.columns == ["cts", "residential", "agricultural", "industrial"]
        ).all()
        assert len(load) == 7000
        assert np.isclose(load.loc[timeindex[453], "cts"], 8.33507e-05)
        assert np.isclose(load.loc[timeindex[13], "residential"], 1.73151e-04)
        assert np.isclose(load.loc[timeindex[6328], "agricultural"], 1.01346e-04)
        assert np.isclose(load.loc[timeindex[4325], "industrial"], 9.91768e-05)

    @pytest.mark.local
    def test_cop_oedb(self):
        cop_df = timeseries_import.cop_oedb(
            pytest.engine, weather_cell_ids=[11051, 11052]
        )
        assert cop_df.shape == (8760, 2)
        assert (cop_df > 1.0).all().all()
        assert (cop_df < 10.0).all().all()

        # ToDo
        # # test with overwriting time index
        # cop_df = timeseries_import.cop_oedb(
        #     pytest.engine, weather_cell_ids=[11051, 11052], year=2010)
        #
        # # test with leap year
        # cop_df = timeseries_import.cop_oedb(
        #     pytest.engine, weather_cell_ids=[11051, 11052], year=2020)

    def setup_egon_heat_pump_data(self):
        names = [
            "HP_442081",
            "Heat_Pump_LVGrid_1163850014_district_heating_6",
            "HP_448156",
        ]
        building_ids = [431821, None, 448156]
        sector = ["individual_heating", "district_heating", "individual_heating"]
        weather_cell_ids = [11051, 11051, 11052]
        district_heating_ids = [None, 5, None]
        hp_df = pd.DataFrame(
            data={
                "bus": "dummy_bus",
                "p_set": 1.0,
                "building_id": building_ids,
                "type": "heat_pump",
                "sector": sector,
                "weather_cell_id": weather_cell_ids,
                "district_heating_id": district_heating_ids,
            },
            index=names,
        )
        return hp_df

    @pytest.mark.local
    def test_heat_demand_oedb(self, caplog):
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        hp_data_egon = self.setup_egon_heat_pump_data()
        edisgo_object.topology.loads_df = pd.concat(
            [edisgo_object.topology.loads_df, hp_data_egon]
        )
        df = timeseries_import.heat_demand_oedb(
            edisgo_object, "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 3)
        assert df.index[0].year == 2035

        # test for leap year
        with caplog.at_level(logging.WARNING):
            df = timeseries_import.heat_demand_oedb(
                edisgo_object, "eGon100RE", pytest.engine, year=2020
            )
        assert "A leap year was given to 'heat_demand_oedb' function." in caplog.text
        assert df.shape == (8760, 3)
        assert df.index[0].year == 2045

        # ToDo add further tests

    @pytest.mark.local
    def test_electricity_demand_oedb(self, caplog):
        # test with one load each and without year
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        df = timeseries_import.electricity_demand_oedb(
            edisgo_object,
            "eGon2035",
            pytest.engine,
            load_names=[
                "Load_mvgd_33532_1_industrial",
                "Load_mvgd_33532_lvgd_1140900000_1_residential",
                "Load_mvgd_33532_lvgd_1163850001_47_cts",
            ],
        )
        assert df.shape == (8760, 3)
        assert df.index[0].year == 2035

        # test without CTS and residential and given year
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        df = timeseries_import.electricity_demand_oedb(
            edisgo_object,
            "eGon2035",
            pytest.engine,
            load_names=["Load_mvgd_33532_1_industrial"],
            year=2011,
        )
        assert df.shape == (8760, 1)
        assert df.index[0].year == 2011

        # test for leap year and all loads in the grid
        with caplog.at_level(logging.WARNING):
            df = timeseries_import.electricity_demand_oedb(
                edisgo_object, "eGon100RE", pytest.engine, year=2020
            )
        assert (
            "A leap year was given to 'electricity_demand_oedb' function."
            in caplog.text
        )
        assert df.shape == (8760, 2463)
        assert df.index[0].year == 2045

        # ToDo add further tests to check values

    @pytest.mark.local
    def test_get_residential_heat_profiles_per_building(self):
        df = timeseries_import.get_residential_heat_profiles_per_building(
            [442081, 448156], "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 2)
        # ToDo add further tests

    @pytest.mark.local
    def test_get_district_heating_heat_demand_profiles(self):
        df = timeseries_import.get_district_heating_heat_demand_profiles(
            [6], "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 1)
        # ToDo add further tests

    @pytest.mark.local
    def test_get_cts_profiles_per_building(self):
        df = timeseries_import.get_cts_profiles_per_building(
            33532, "eGon2035", "heat", pytest.engine
        )
        assert df.shape == (8760, 85)
        df = timeseries_import.get_cts_profiles_per_building(
            33532, "eGon2035", "electricity", pytest.engine
        )
        assert df.shape == (8760, 85)
        # ToDo add further tests

    @pytest.mark.local
    def test_get_residential_electricity_profiles_per_building(self):
        df = timeseries_import.get_residential_electricity_profiles_per_building(
            [-1, 442081], "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 1)
        assert np.isclose(df.loc[:, 442081].sum(), 7.799, atol=1e-3)

    @pytest.mark.local
    def test_get_industrial_electricity_profiles_per_site(self):
        # test with one site and one OSM area
        df = timeseries_import.get_industrial_electricity_profiles_per_site(
            [1, 541658], "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 2)
        assert np.isclose(df.loc[:, 1].sum(), 32417.233, atol=1e-3)
        assert np.isclose(df.loc[:, 541658].sum(), 2554.944, atol=1e-3)

        # test without site and only OSM area
        df = timeseries_import.get_industrial_electricity_profiles_per_site(
            [541658], "eGon2035", pytest.engine
        )
        assert df.shape == (8760, 1)
