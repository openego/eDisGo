import pytest

from edisgo import EDisGo
from edisgo.io import dsm_import


class TestDSMImport:
    @pytest.mark.local
    def test_oedb(self):
        # test without industrial load
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        dsm_profiles = dsm_import.oedb(
            edisgo_object, scenario="eGon2035", engine=pytest.engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 85)
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()

        # test with one industrial load
        dsm_load = edisgo_object.topology.loads_df[
            (edisgo_object.topology.loads_df.type == "conventional_load")
            & (edisgo_object.topology.loads_df.sector == "cts")
        ].index[0]
        edisgo_object.topology.loads_df.at[dsm_load, "sector"] = "industry"
        edisgo_object.topology.loads_df.at[dsm_load, "building_id"] = 1

        dsm_profiles = dsm_import.oedb(
            edisgo_object, scenario="eGon2035", engine=pytest.engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 85)
            assert dsm_load in dsm_profiles[dsm_profile].columns
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()

    @pytest.mark.local
    def test_get_profiles_per_industrial_load(self):
        dsm_profiles = dsm_import.get_profiles_per_industrial_load(
            load_ids=[15388, 241, 1], scenario="eGon2035", engine=pytest.engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 3)
            assert sorted(dsm_profiles[dsm_profile].columns) == [1, 241, 15388]
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()

        dsm_profiles = dsm_import.get_profiles_per_industrial_load(
            load_ids=[], scenario="eGon2035", engine=pytest.engine
        )
        assert dsm_profiles["p_min"].empty

    @pytest.mark.local
    def test_get_profile_cts(self):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        dsm_profiles = dsm_import.get_profile_cts(
            edisgo_obj=edisgo, scenario="eGon2035", engine=pytest.engine
        )
        for dsm_profile in ["e_max", "e_min", "p_max", "p_min"]:
            assert dsm_profiles[dsm_profile].shape == (8760, 85)
        assert (dsm_profiles["p_min"] <= 0.0).all().all()
        assert (dsm_profiles["e_min"] <= 0.0).all().all()
        assert (dsm_profiles["p_max"] >= 0.0).all().all()
        assert (dsm_profiles["e_max"] >= 0.0).all().all()
