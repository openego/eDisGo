import logging

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_index_equal

from edisgo import EDisGo
from edisgo.io import timeseries_import
from edisgo.tools.config import Config


@pytest.fixture
def mock_oep_demand_retrieval(monkeypatch):
    """
    Replace OEP heat and electricity demand retrieval with local profiles.
    This is a shared fixture. Hence a dispatcher as an internal helper is
    required.
    """

    profiles = {
        "residential_heat": pd.DataFrame(
            {
                431821: np.full(8760, 0.1),
                430859: np.full(8760, 0.2),
            }
        ),
        "cts_heat": pd.DataFrame(
            {
                431821: np.full(8760, 0.3),
                999999: np.full(8760, 0.5),
            }
        ),
        "district_heating": pd.DataFrame({5: np.full(8760, 0.4)}),
        "residential_electricity": pd.DataFrame({444312: np.full(8760, 0.1)}),
        "cts_electricity": pd.DataFrame(
            {
                441688: np.full(8760, 0.2),
                999999: np.full(8760, 0.5),
            }
        ),
        "industrial_electricity": pd.DataFrame({541658: np.full(8760, 0.3)}),
    }

    residential_heat_mock = Mock(
        side_effect=lambda *args, **kwargs: profiles["residential_heat"].copy()
    )
    district_heating_mock = Mock(
        side_effect=lambda *args, **kwargs: profiles["district_heating"].copy()
    )
    residential_electricity_mock = Mock(
        side_effect=lambda *args, **kwargs: profiles["residential_electricity"].copy()
    )
    industrial_electricity_mock = Mock(
        side_effect=lambda *args, **kwargs: profiles["industrial_electricity"].copy()
    )

    def _get_cts_profiles(*args, **kwargs):
        # sector is the third positional argument in:
        # get_cts_profiles_per_building(edisgo_obj, scenario, sector, engine)
        # This is shared by two production functions (heat_demand_oedb(),
        # electricity_demand_oedb()). Hence a dispatcher is required:
        sector = kwargs.get("sector", args[2] if len(args) > 2 else None)

        if sector == "heat":
            return profiles["cts_heat"].copy()
        if sector == "electricity":
            return profiles["cts_electricity"].copy()

        raise ValueError(f"Unexpected CTS sector: {sector}")

    cts_mock = Mock(side_effect=_get_cts_profiles)

    monkeypatch.setattr(
        timeseries_import,
        "get_residential_heat_profiles_per_building",
        residential_heat_mock,
    )
    monkeypatch.setattr(
        timeseries_import,
        "get_district_heating_heat_demand_profiles",
        district_heating_mock,
    )
    monkeypatch.setattr(
        timeseries_import,
        "get_residential_electricity_profiles_per_building",
        residential_electricity_mock,
    )
    monkeypatch.setattr(
        timeseries_import,
        "get_industrial_electricity_profiles_per_site",
        industrial_electricity_mock,
    )
    monkeypatch.setattr(
        timeseries_import,
        "get_cts_profiles_per_building",
        cts_mock,
    )

    return {
        "residential_heat": residential_heat_mock,
        "district_heating": district_heating_mock,
        "residential_electricity": residential_electricity_mock,
        "industrial_electricity": industrial_electricity_mock,
        "cts": cts_mock,
    }


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

    @pytest.mark.oep
    def test_query_feedin_oedb_legacy_live(self, oep_engine):
        """Test the legacy OEP query contract for one weather cell."""

        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

        database_data = timeseries_import._query_feedin_oedb_legacy(
            edisgo,
            weather_cell_ids={1122074},
            engine=oep_engine,
        )

        assert set(database_data.columns) == {
            "weather_cell_id",
            "carrier",
            "feedin",
        }
        assert set(database_data.weather_cell_id) == {1122074}
        assert set(database_data.carrier) == {"solar", "wind_onshore"}
        assert database_data.feedin.map(len).eq(8760).all()

    def test_feedin_oedb_legacy_offline(self, monkeypatch):
        """Test legacy feed-in transformation with synthetic query data."""

        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        requested_timeindex = pd.date_range(
            "1/1/2011 01:00",
            periods=2,
            freq="H",
        )
        database_data = pd.DataFrame(
            {
                "weather_cell_id": [1122074, 1122074],
                "carrier": ["solar", "wind_onshore"],
                "feedin": [
                    np.arange(8760, dtype=float) / 100,
                    np.arange(8760, dtype=float) / 200,
                ],
            }
        )

        weather_cells_mock = Mock(return_value={1122074})
        query_mock = Mock(return_value=database_data)
        monkeypatch.setattr(
            timeseries_import.tools,
            "get_weather_cells_intersecting_with_grid_district",
            weather_cells_mock,
        )
        monkeypatch.setattr(
            timeseries_import,
            "_query_feedin_oedb_legacy",
            query_mock,
        )

        feedin = timeseries_import.feedin_oedb_legacy(
            edisgo,
            timeindex=requested_timeindex,
            engine=None,
        )

        assert feedin.shape == (2, 2)
        assert_index_equal(feedin.index, requested_timeindex)
        assert feedin.columns.tolist() == [
            ("solar", 1122074),
            ("wind", 1122074),
        ]
        assert feedin[("solar", 1122074)].tolist() == [0.01, 0.02]
        assert feedin[("wind", 1122074)].tolist() == [0.005, 0.01]
        query_mock.assert_called_once_with(
            edisgo,
            {1122074},
            engine=None,
        )

    @pytest.mark.oep
    def test_feedin_oedb_live(self, oep_engine):
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        timeindex = pd.date_range("1/2/2018", periods=6, freq="H")
        edisgo_object.set_timeindex(timeindex)
        feedin_df = timeseries_import.feedin_oedb(
            edisgo_object,
            engine=oep_engine,
        )
        assert feedin_df.shape == (6, 4)
        assert_index_equal(feedin_df.index, timeindex)

    def test_feedin_oedb_offline(self, monkeypatch):
        """Test feed-in reshaping and slicing with synthetic query data."""

        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )
        requested_timeindex = pd.date_range(
            "1/1/2011 01:00",
            periods=2,
            freq="H",
        )
        database_data = pd.DataFrame(
            {
                "weather_cell_id": [11051, 11051],
                "carrier": ["pv", "wind_onshore"],
                "feedin": [
                    np.arange(8760, dtype=float) / 100,
                    np.arange(8760, dtype=float) / 200,
                ],
            }
        )

        weather_cells_mock = Mock(return_value=[11051])
        query_mock = Mock(return_value=database_data)
        monkeypatch.setattr(
            timeseries_import.tools,
            "get_weather_cells_intersecting_with_grid_district",
            weather_cells_mock,
        )
        monkeypatch.setattr(
            timeseries_import,
            "_query_feedin_oedb",
            query_mock,
        )

        feedin_df = timeseries_import.feedin_oedb(
            edisgo_object,
            engine=None,
            timeindex=requested_timeindex,
        )

        assert feedin_df.shape == (2, 2)
        assert_index_equal(feedin_df.index, requested_timeindex)
        assert feedin_df.columns.tolist() == [
            ("solar", 11051),
            ("wind", 11051),
        ]
        assert feedin_df[("solar", 11051)].tolist() == [0.01, 0.02]
        assert feedin_df[("wind", 11051)].tolist() == [0.005, 0.01]
        query_mock.assert_called_once_with(None, [11051])

    def test_load_time_series_demandlib(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        timeindex = pd.date_range("1/1/2018", periods=8760, freq="H")
        load = timeseries_import.load_time_series_demandlib(edisgo, timeindex)
        assert (
            load.columns == ["cts", "residential", "agricultural", "industrial"]
        ).all()
        assert len(load) == 8760
        assert np.isclose(load.loc[timeindex[453], "cts"], 8.33507e-05)
        assert np.isclose(load.loc[timeindex[13], "residential"], 1.73151e-04)
        assert np.isclose(load.loc[timeindex[6328], "agricultural"], 1.01346e-04)
        assert np.isclose(load.loc[timeindex[4325], "industrial"], 9.87654320e-05)
        assert np.isclose(load.sum()["cts"], 1.0)
        assert np.isclose(load.sum()["residential"], 1.0)
        assert np.isclose(load.sum()["agricultural"], 1.0)
        assert np.isclose(load.sum()["industrial"], 1.0)

    @pytest.mark.oep
    def test_cop_oedb_live(self, oep_engine):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        timeindex = pd.date_range("1/1/2011", periods=2, freq="H")
        cop_df = timeseries_import.cop_oedb(
            edisgo_object=edisgo,
            engine=oep_engine,
            weather_cell_ids=[11051],
            timeindex=timeindex,
        )
        assert cop_df.shape == (2, 1)
        assert_index_equal(cop_df.index, timeindex)
        assert (cop_df > 1.0).all().all()
        assert (cop_df < 10.0).all().all()

    def test_cop_oedb_offline(self, monkeypatch):
        """Test COP array conversion and slicing with synthetic query data."""

        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        requested_timeindex = pd.date_range(
            "1/1/2011 02:00",
            periods=2,
            freq="H",
        )
        cop_values = np.arange(8760, dtype=float) / 10 + 2
        database_data = pd.DataFrame(
            {"cop": [cop_values]},
            index=pd.Index([11051], name="w_id"),
        )

        query_mock = Mock(return_value=database_data)
        monkeypatch.setattr(
            timeseries_import,
            "_query_cop_oedb",
            query_mock,
        )

        cop_df = timeseries_import.cop_oedb(
            edisgo_object=edisgo,
            engine=None,
            weather_cell_ids=[11051],
            timeindex=requested_timeindex,
        )

        assert cop_df.shape == (2, 1)
        assert_index_equal(cop_df.index, requested_timeindex)
        assert cop_df[11051].tolist() == [2.2, 2.3]
        query_mock.assert_called_once_with(None, [11051])

    def setup_egon_heat_pump_data(self):
        names = [
            "HP_442081",
            "Heat_Pump_LVGrid_1163850014_district_heating_6",
            "HP_448156",
        ]
        building_ids = [431821, None, 430859]
        sector = ["individual_heating", "district_heating", "individual_heating"]
        weather_cell_ids = [11051, 11051, 11052]
        district_heating_ids = [None, 5, None]
        area_ids = [None, 5, None]
        hp_df = pd.DataFrame(
            data={
                "bus": "dummy_bus",
                "p_set": 1.0,
                "building_id": building_ids,
                "type": "heat_pump",
                "sector": sector,
                "weather_cell_id": weather_cell_ids,
                "district_heating_id": district_heating_ids,
                "area_id": area_ids,
            },
            index=names,
        )
        return hp_df

    def test_heat_demand_oedb(
        self,
        caplog,
        mock_oep_demand_retrieval,
    ):
        """Test heat-demand processing with mocked OEP retrieval."""

        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )
        hp_data_egon = self.setup_egon_heat_pump_data()
        edisgo_object.topology.loads_df = pd.concat(
            [edisgo_object.topology.loads_df, hp_data_egon]
        )

        mocks = mock_oep_demand_retrieval

        df = timeseries_import.heat_demand_oedb(
            edisgo_object,
            "eGon2035",
            engine=None,
        )

        assert df.shape == (8760, 3)
        assert df.index[0].year == 2035
        assert set(df.columns) == set(hp_data_egon.index)

        # Residential and CTS profiles for building 431821 are added together.
        assert np.isclose(df["HP_442081"].iloc[0], 0.4)

        # Only a residential profile exists for building 430859.
        assert np.isclose(df["HP_448156"].iloc[0], 0.2)

        # District-heating profile belonging to area 5.
        assert np.isclose(
            df["Heat_Pump_LVGrid_1163850014_district_heating_6"].iloc[0],
            0.4,
        )

        # Test handling of an unsupported leap-year index.
        with caplog.at_level(logging.WARNING):
            df = timeseries_import.heat_demand_oedb(
                edisgo_object,
                "eGon100RE",
                engine=None,
                timeindex=pd.date_range(
                    "1/1/2020",
                    periods=8760,
                    freq="H",
                ),
            )

        assert "A leap year was given." in caplog.text
        assert df.shape == (8760, 3)
        assert df.index[0].year == 2045

        assert mocks["residential_heat"].call_count == 2
        assert mocks["district_heating"].call_count == 2
        assert mocks["cts"].call_count == 2

    def test_electricity_demand_oedb(
        self,
        caplog,
        mock_oep_demand_retrieval,
    ):
        """Test electricity-demand processing with mocked OEP retrieval."""

        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )

        industrial_load = "Load_mvgd_33535_1_industrial"
        residential_load = "Load_mvgd_33535_lvgd_1141170000_1_residential"
        cts_load = "Load_mvgd_33535_lvgd_1164120005_60_cts"
        selected_loads = [
            industrial_load,
            residential_load,
            cts_load,
        ]

        mocks = mock_oep_demand_retrieval

        # Test one selected load from each sector.
        df = timeseries_import.electricity_demand_oedb(
            edisgo_object,
            "eGon2035",
            engine=None,
            load_names=selected_loads,
        )

        assert df.shape == (8760, 3)
        assert df.index[0].year == 2035
        assert set(df.columns) == set(selected_loads)
        assert np.isclose(df[residential_load].iloc[0], 0.1)
        assert np.isclose(df[cts_load].iloc[0], 0.2)
        assert np.isclose(df[industrial_load].iloc[0], 0.3)

        # Test an explicitly provided short time index.
        requested_timeindex = pd.date_range(
            "1/1/2011",
            periods=4,
            freq="H",
        )
        df = timeseries_import.electricity_demand_oedb(
            edisgo_object,
            "eGon2035",
            engine=None,
            load_names=[industrial_load],
            timeindex=requested_timeindex,
        )

        assert df.shape == (4, 1)
        assert df.index.equals(requested_timeindex)

        # Keep the all-loads case small by retaining only the three test loads.
        edisgo_object.topology.loads_df = edisgo_object.topology.loads_df.loc[
            selected_loads
        ].copy()

        # Test automatic replacement of a leap-year index.
        with caplog.at_level(logging.WARNING):
            df = timeseries_import.electricity_demand_oedb(
                edisgo_object,
                "eGon100RE",
                engine=None,
                timeindex=pd.date_range(
                    "1/1/2020",
                    periods=4,
                    freq="H",
                ),
            )

        assert "A leap year was given." in caplog.text
        assert df.shape == (8760, 3)
        assert df.index[0].year == 2045
        assert set(df.columns) == set(selected_loads)

        assert mocks["residential_electricity"].call_count == 2
        assert mocks["industrial_electricity"].call_count == 3
        assert mocks["cts"].call_count == 2

    @pytest.mark.oep
    def test_query_residential_heat_profile_data_live(self, oep_engine):
        """Check the live schemas needed to construct residential heat profiles."""

        peta_demand, profile_ids, daily_profiles, daily_demand_share = (
            timeseries_import._query_residential_heat_profile_data(
                [442081],
                "eGon2035",
                oep_engine,
            )
        )

        assert peta_demand.columns.tolist() == ["zensus_id", "demand"]
        assert not peta_demand.empty
        assert {
            "zensus_id",
            "building_id",
            "selected_idp_profiles",
            "buildings",
            "day_of_year",
        }.issubset(profile_ids.columns)
        assert not profile_ids.empty
        assert {"idp", "hour"}.issubset(daily_profiles.columns)
        assert not daily_profiles.empty
        assert daily_demand_share.columns.tolist() == [
            "zensus_id",
            "day_of_year",
            "daily_demand_share",
        ]
        assert not daily_demand_share.empty

    def test_get_residential_heat_profiles_per_building_offline(
        self,
        monkeypatch,
    ):
        """Test residential heat-profile construction with synthetic query data."""

        peta_demand = pd.DataFrame({"zensus_id": [1], "demand": [120.0]})
        profile_ids = pd.DataFrame(
            {
                "zensus_id": [1, 1, 1, 1],
                "building_id": [101, 101, 102, 102],
                "selected_idp_profiles": [11, 11, 12, 12],
                "buildings": [2, 2, 2, 2],
                "day_of_year": [1, 2, 1, 2],
            }
        )
        daily_profiles = pd.DataFrame(
            {
                "idp": [0.25, 0.75, 0.5, 0.5],
                "hour": [1, 2, 1, 2],
            },
            index=pd.Index([11, 11, 12, 12], name="index"),
        )
        daily_demand_share = pd.DataFrame(
            {
                "zensus_id": [1, 1],
                "day_of_year": [1, 2],
                "daily_demand_share": [0.4, 0.6],
            }
        )
        query_mock = Mock(
            return_value=(
                peta_demand,
                profile_ids,
                daily_profiles,
                daily_demand_share,
            )
        )
        monkeypatch.setattr(
            timeseries_import,
            "_query_residential_heat_profile_data",
            query_mock,
        )

        profiles = timeseries_import.get_residential_heat_profiles_per_building(
            building_ids=[101, 102],
            scenario="eGon2035",
            engine=None,
        )

        assert profiles.shape == (4, 2)
        assert profiles.columns.tolist() == [101, 102]
        assert profiles[101].tolist() == [6.0, 18.0, 9.0, 27.0]
        assert profiles[102].tolist() == [12.0, 12.0, 18.0, 18.0]
        query_mock.assert_called_once_with([101, 102], "eGon2035", None)

    @pytest.mark.oep
    def test_get_district_heating_heat_demand_profiles_live(self, oep_engine):
        df = timeseries_import.get_district_heating_heat_demand_profiles(
            [6], "eGon2035", oep_engine
        )
        assert df.shape == (8760, 1)
        assert df.columns.tolist() == [6]
        assert df.index.tolist() == list(range(1, 8761))

    def test_get_district_heating_heat_demand_profiles_offline(
        self,
        monkeypatch,
    ):
        """Test district-heating array expansion with synthetic query data."""

        database_data = pd.DataFrame(
            {
                "area_id": [5, 6],
                "dist_aggregated_mw": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            }
        )
        query_mock = Mock(return_value=database_data)
        monkeypatch.setattr(
            timeseries_import,
            "_query_district_heating_heat_demand_profiles",
            query_mock,
        )

        profiles = timeseries_import.get_district_heating_heat_demand_profiles(
            district_heating_ids=[5, 6],
            scenario="eGon2035",
            engine=None,
        )

        assert profiles.shape == (3, 2)
        assert profiles.columns.tolist() == [5, 6]
        assert profiles.index.tolist() == [1, 2, 3]
        assert profiles[5].tolist() == [0.1, 0.2, 0.3]
        assert profiles[6].tolist() == [0.4, 0.5, 0.6]
        query_mock.assert_called_once_with(None, [5, 6], "eGon2035")

    def test_get_cts_profiles_per_building_offline(self, monkeypatch):
        """Test cross-grid CTS profile collection with synthetic data."""

        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )
        cts_loads = edisgo_object.topology.loads_df[
            (edisgo_object.topology.loads_df.type == "conventional_load")
            & (edisgo_object.topology.loads_df.sector == "cts")
        ].iloc[:2]
        edisgo_object.topology.loads_df = cts_loads.copy()
        building_ids = cts_loads.building_id.astype(int).tolist()

        building_grid_map = pd.DataFrame(
            {"bus_id": [33535, 99999]},
            index=pd.Index(building_ids, name="building_id"),
        )

        def _get_profiles_per_grid(bus_id, scenario, sector, engine):
            building_id = building_ids[0] if bus_id == 33535 else building_ids[1]
            values = [0.1, 0.2] if bus_id == 33535 else [0.3, 0.4]
            return pd.DataFrame({building_id: values})

        mapping_query_mock = Mock(return_value=building_grid_map)
        per_grid_mock = Mock(side_effect=_get_profiles_per_grid)
        monkeypatch.setattr(
            timeseries_import,
            "_query_cts_building_grid_map",
            mapping_query_mock,
        )
        monkeypatch.setattr(
            timeseries_import,
            "get_cts_profiles_per_grid",
            per_grid_mock,
        )

        profiles = timeseries_import.get_cts_profiles_per_building(
            edisgo_object,
            scenario="eGon2035",
            sector="electricity",
            engine=None,
        )

        assert profiles.shape == (2, 2)
        assert profiles.columns.tolist() == building_ids
        assert profiles[building_ids[0]].tolist() == [0.1, 0.2]
        assert profiles[building_ids[1]].tolist() == [0.3, 0.4]
        assert per_grid_mock.call_count == 2
        assert set(call.kwargs["bus_id"] for call in per_grid_mock.call_args_list) == {
            33535,
            99999,
        }

    @pytest.mark.oep
    def test_query_cts_profiles_per_grid_live(self, oep_engine):
        """Test the live query contracts for heat and electricity CTS data."""

        for sector in ["heat", "electricity"]:
            raw_profile, demand_share, total_heat_demand = (
                timeseries_import._query_cts_profiles_per_grid_data(
                    bus_id=33535,
                    scenario="eGon2035",
                    sector=sector,
                    engine=oep_engine,
                )
            )

            assert raw_profile.shape[0] == 1
            assert raw_profile.columns.tolist() == ["bus_id", "p_set"]
            assert len(raw_profile.iloc[0].p_set) == 8760
            assert demand_share.columns.tolist() == ["profile_share"]
            assert not demand_share.empty
            if sector == "heat":
                assert total_heat_demand > 0
            else:
                assert total_heat_demand is None

    def test_get_cts_profiles_per_grid_offline(self, monkeypatch):
        """Test CTS disaggregation and heat scaling with synthetic data."""

        def _query_profiles(bus_id, scenario, sector, engine):
            if sector == "electricity":
                raw_profile = pd.DataFrame({"bus_id": [bus_id], "p_set": [[4.0, 8.0]]})
                total_heat_demand = None
            else:
                raw_profile = pd.DataFrame({"bus_id": [bus_id], "p_set": [[2.0, 2.0]]})
                total_heat_demand = 8.0

            demand_share = pd.DataFrame(
                {"profile_share": [0.25, 0.75]},
                index=pd.Index([101, 102], name="building_id"),
            )
            return raw_profile, demand_share, total_heat_demand

        query_mock = Mock(side_effect=_query_profiles)
        monkeypatch.setattr(
            timeseries_import,
            "_query_cts_profiles_per_grid_data",
            query_mock,
        )

        electricity = timeseries_import.get_cts_profiles_per_grid(
            bus_id=33535,
            scenario="eGon2035",
            sector="electricity",
            engine=None,
        )
        heat = timeseries_import.get_cts_profiles_per_grid(
            bus_id=33535,
            scenario="eGon2035",
            sector="heat",
            engine=None,
        )

        assert electricity.shape == (2, 2)
        assert electricity[101].tolist() == [1.0, 2.0]
        assert electricity[102].tolist() == [3.0, 6.0]
        assert heat.shape == (2, 2)
        assert heat[101].tolist() == [1.0, 1.0]
        assert heat[102].tolist() == [3.0, 3.0]
        assert query_mock.call_count == 2

    @pytest.mark.oep
    def test_query_residential_electricity_profile_data_live(self, oep_engine):
        """Check the live schemas needed for residential electricity profiles."""

        scaling_factors, profile_ids, profiles = (
            timeseries_import._query_residential_electricity_profile_data(
                [-1, 442081],
                "eGon2035",
                oep_engine,
            )
        )

        assert scaling_factors.columns.tolist() == ["factor"]
        assert not scaling_factors.empty
        assert profile_ids.columns.tolist() == [
            "building_id",
            "cell_id",
            "profile_id",
        ]
        assert not profile_ids.empty
        assert profiles.shape[0] == 8760
        assert not profiles.empty

    def test_get_residential_electricity_profiles_per_building_offline(
        self,
        monkeypatch,
    ):
        """Test mapping, aggregation and scaling with synthetic query data."""

        scaling_factors = pd.DataFrame(
            {"factor": [2.0, 0.5]},
            index=pd.Index([10, 20], name="cell_id"),
        )
        profile_ids = pd.DataFrame(
            {
                "building_id": [101, 101, 102],
                "cell_id": [10, 10, 20],
                "profile_id": ["p1", "p2", "p2"],
            }
        )
        database_profiles = pd.DataFrame(
            {
                "p1": [1_000_000.0, 2_000_000.0, 3_000_000.0],
                "p2": [500_000.0, 1_000_000.0, 1_500_000.0],
            }
        )
        query_mock = Mock(
            return_value=(scaling_factors, profile_ids, database_profiles)
        )
        monkeypatch.setattr(
            timeseries_import,
            "_query_residential_electricity_profile_data",
            query_mock,
        )

        profiles = timeseries_import.get_residential_electricity_profiles_per_building(
            [-1, 101, 102],
            "eGon2035",
            engine=None,
        )

        assert profiles.shape == (3, 2)
        assert profiles.columns.tolist() == [101, 102]
        assert profiles[101].tolist() == [3.0, 6.0, 9.0]
        assert profiles[102].tolist() == [0.25, 0.5, 0.75]
        query_mock.assert_called_once_with([-1, 101, 102], "eGon2035", None)

    @pytest.mark.oep
    def test_query_industrial_electricity_profile_data_live(self, oep_engine):
        """Check one live profile from each industrial source table."""

        sites, areas = timeseries_import._query_industrial_electricity_profile_data(
            [1, 541658],
            "eGon2035",
            oep_engine,
        )

        for result in [sites, areas]:
            assert result.shape[0] == 1
            assert result.columns.tolist() == ["site_id", "p_set"]
            assert len(result.iloc[0].p_set) == 8760

    def test_get_industrial_electricity_profiles_per_site_offline(
        self,
        monkeypatch,
    ):
        """Test merging and expanding industrial profiles with synthetic data."""

        sites = pd.DataFrame({"site_id": [1], "p_set": [[1.0, 2.0, 3.0]]})
        areas = pd.DataFrame({"site_id": [541658], "p_set": [[4.0, 5.0, 6.0]]})
        query_mock = Mock(return_value=(sites, areas))
        monkeypatch.setattr(
            timeseries_import,
            "_query_industrial_electricity_profile_data",
            query_mock,
        )

        profiles = timeseries_import.get_industrial_electricity_profiles_per_site(
            [1, 541658],
            "eGon2035",
            engine=None,
        )

        assert profiles.shape == (3, 2)
        assert profiles.columns.tolist() == [1, 541658]
        assert profiles[1].tolist() == [1.0, 2.0, 3.0]
        assert profiles[541658].tolist() == [4.0, 5.0, 6.0]
        query_mock.assert_called_once_with([1, 541658], "eGon2035", None)
