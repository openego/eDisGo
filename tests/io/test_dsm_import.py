from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.io import dsm_import


@pytest.fixture
def mock_dsm_oedb_retrieval(monkeypatch):
    """Replace external DSM retrieval with local profiles.

    The returned ``profile_values`` entry accepts scalars or hourly arrays for
    each profile and sector, allowing tests to customize the synthetic inputs.
    """

    profile_values = {
        "e_min": -0.4,
        "e_max": 0.4,
        "p_min": -0.2,
        "p_max": 0.2,
    }
    sector_values = {
        "cts": profile_values.copy(),
        "industrial": profile_values.copy(),
    }

    def _create_profiles(columns, values):
        return {
            profile: pd.DataFrame(
                {column: value for column in columns},
                index=range(8760),
            )
            for profile, value in values.items()
        }

    def _get_cts_profiles(edisgo_obj, scenario, engine):
        cts_loads = edisgo_obj.topology.loads_df[
            (edisgo_obj.topology.loads_df.type == "conventional_load")
            & (edisgo_obj.topology.loads_df.sector == "cts")
        ]
        return _create_profiles(cts_loads.index, sector_values["cts"])

    def _get_industrial_profiles(load_ids, scenario, engine):
        return _create_profiles(load_ids, sector_values["industrial"])

    cts_mock = Mock(side_effect=_get_cts_profiles)
    industrial_mock = Mock(side_effect=_get_industrial_profiles)

    monkeypatch.setattr(dsm_import, "get_profile_cts", cts_mock)
    monkeypatch.setattr(
        dsm_import,
        "get_profiles_per_industrial_load",
        industrial_mock,
    )

    return {
        "profile_values": sector_values,
        "cts": cts_mock,
        "industrial": industrial_mock,
    }


class TestDSMImport:
    def test_oedb_offline(self, mock_dsm_oedb_retrieval):
        """Test combining CTS and industrial DSM profiles offline."""

        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )

        cts_loads = edisgo_object.topology.loads_df[
            (edisgo_object.topology.loads_df.type == "conventional_load")
            & (edisgo_object.topology.loads_df.sector == "cts")
        ].index[:2]
        edisgo_object.topology.loads_df = edisgo_object.topology.loads_df.loc[
            cts_loads
        ].copy()

        mocks = mock_dsm_oedb_retrieval

        # Initially both loads are CTS loads.
        dsm_profiles = dsm_import.oedb(
            edisgo_object,
            scenario="eGon2035",
            engine=None,
        )

        for profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[profile].shape == (8760, 2)
            assert set(dsm_profiles[profile].columns) == set(cts_loads)
            assert dsm_profiles[profile].index[0].year == 2035

        assert (dsm_profiles["p_min"] == -0.2).all().all()
        assert (dsm_profiles["e_min"] == -0.4).all().all()
        assert (dsm_profiles["p_max"] == 0.2).all().all()
        assert (dsm_profiles["e_max"] == 0.4).all().all()

        # Convert one CTS load into an industrial load. The real oedb function
        # must rename its building ID back to the eDisGo load name when merging.
        industrial_load = cts_loads[0]
        edisgo_object.topology.loads_df.at[industrial_load, "sector"] = "industrial"
        edisgo_object.topology.loads_df.at[industrial_load, "building_id"] = 1

        dsm_profiles = dsm_import.oedb(
            edisgo_object,
            scenario="eGon2035",
            engine=None,
        )

        for profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[profile].shape == (8760, 2)
            assert set(dsm_profiles[profile].columns) == set(cts_loads)
            assert industrial_load in dsm_profiles[profile].columns

        assert mocks["cts"].call_count == 2
        assert mocks["industrial"].call_count == 2

    def test_oedb_partial_timeindex_offline(self, mock_dsm_oedb_retrieval):
        """Slice all four DSM profiles and preserve CTS/industrial load mapping."""
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        names = edisgo_object.topology.loads_df.query(
            "type == 'conventional_load' and sector == 'cts'"
        ).index[:2]
        cts_load, industrial_load = names
        edisgo_object.topology.loads_df = edisgo_object.topology.loads_df.loc[
            names
        ].copy()
        edisgo_object.topology.loads_df.at[industrial_load, "sector"] = "industrial"
        edisgo_object.topology.loads_df.at[industrial_load, "building_id"] = 1

        # Annual values vary by hour; industrial values differ from CTS values.
        hourly_factor = 1 + np.arange(8760, dtype=float) / 10000
        values = mock_dsm_oedb_retrieval["profile_values"]
        for profile, base_value in values["cts"].items():
            values["cts"][profile] = base_value * hourly_factor
            values["industrial"][profile] = 2 * base_value * hourly_factor
        requested_timeindex = pd.date_range("2035-02-01 12:00", periods=3, freq="h")

        result = dsm_import.oedb(
            edisgo_object, "eGon2035", engine=None, timeindex=requested_timeindex
        )

        expected_values = {
            "p_min": [-0.21512, -0.21514, -0.21516],
            "p_max": [0.21512, 0.21514, 0.21516],
            "e_min": [-0.43024, -0.43028, -0.43032],
            "e_max": [0.43024, 0.43028, 0.43032],
        }
        assert set(result) == set(expected_values)
        for profile, cts_values in expected_values.items():
            expected = pd.DataFrame(
                {cts_load: cts_values, industrial_load: np.array(cts_values) * 2},
                index=requested_timeindex,
            )
            pd.testing.assert_frame_equal(result[profile], expected, check_like=True)

    def test_pivot_helper(self):
        """Test transformation of nested database arrays into time series."""

        database_data = pd.DataFrame(
            {
                "site_id": [1, 2],
                "p_min": [[-0.1, -0.2], [-0.3, -0.4]],
                "time_step": [[0, 1], [0, 1]],
            }
        )

        result = dsm_import._pivot_helper(database_data, "p_min")

        assert result.shape == (2, 2)
        assert result.columns.tolist() == [1, 2]
        assert result[1].tolist() == [-0.1, -0.2]
        assert result[2].tolist() == [-0.3, -0.4]

    def test_distribute_dsm_profiles_to_cts_loads(self):
        """Test proportional distribution of aggregated DSM profiles."""

        dsm_profiles = {
            "p_min": pd.DataFrame({"aggregate": [-4.0, -8.0]}),
            "p_max": pd.DataFrame({"aggregate": [4.0, 8.0]}),
            "e_min": pd.DataFrame({"aggregate": [-12.0, -16.0]}),
            "e_max": pd.DataFrame({"aggregate": [12.0, 16.0]}),
        }
        cts_loads = pd.DataFrame(
            {"p_set": [1.0, 3.0]},
            index=["cts_1", "cts_2"],
        )

        result = dsm_import._distribute_dsm_profiles_to_cts_loads(
            dsm_profiles,
            cts_loads,
        )

        assert result["p_min"]["cts_1"].tolist() == [-1.0, -2.0]
        assert result["p_min"]["cts_2"].tolist() == [-3.0, -6.0]
        assert result["e_max"]["cts_1"].tolist() == [3.0, 4.0]
        assert result["e_max"]["cts_2"].tolist() == [9.0, 12.0]

    def test_get_profiles_per_industrial_load_empty(self):
        """Test that an empty ID selection requires no database access."""

        dsm_profiles = dsm_import.get_profiles_per_industrial_load(
            load_ids=[],
            scenario="eGon2035",
            engine=None,
        )

        for profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[profile].empty

    @pytest.mark.oep
    def test_get_profiles_per_industrial_load_live(self, oep_engine):
        dsm_profiles = dsm_import.get_profiles_per_industrial_load(
            load_ids=[15388, 241, 1], scenario="eGon2035", engine=oep_engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 3)
            assert sorted(dsm_profiles[dsm_profile].columns) == [1, 241, 15388]
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()

    @pytest.mark.oep
    def test_get_profile_cts_live(self, oep_engine):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
        )

        cts_load = edisgo.topology.loads_df[
            (edisgo.topology.loads_df.type == "conventional_load")
            & (edisgo.topology.loads_df.sector == "cts")
        ].index[0]
        edisgo.topology.loads_df = edisgo.topology.loads_df.loc[[cts_load]].copy()

        dsm_profiles = dsm_import.get_profile_cts(
            edisgo_obj=edisgo, scenario="eGon2035", engine=oep_engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 1)
            assert dsm_profiles[dsm_profile].columns.tolist() == [cts_load]
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()
