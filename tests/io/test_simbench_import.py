import shutil

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo


def copy_with_changed_table(tmp_path, table, change):
    """Copy the SimBench test grid and change one of its csv tables."""
    path = tmp_path / "simbench_grid"
    shutil.copytree(pytest.simbench_test_network_path, path)
    file = path / f"{table}.csv"
    df = pd.read_csv(file, sep=";")
    change(df).to_csv(file, sep=";", index=False)
    return str(path)


class TestImportSimbenchGrid:
    @pytest.fixture(scope="class")
    @classmethod
    def edisgo(cls):
        edisgo = EDisGo()
        edisgo.import_simbench_grid(pytest.simbench_test_network_path)
        return edisgo

    def test_grids(self, edisgo):
        topology = edisgo.topology
        assert topology.mv_grid.id == 1101
        assert topology._lv_grid_ids == [1108]
        lv_buses = topology.buses_df[topology.buses_df.lv_grid_id == 1108]
        assert (lv_buses.v_nom == 0.4).all()
        assert (topology.transformers_df.bus1.isin(lv_buses.index)).all()
        # the HV side of the substation is not part of the eDisGo grid
        assert not topology.transformers_hvmv_df.bus0.isin(
            topology.buses_df.index
        ).any()

    def test_grid_is_connected_and_rings_are_open(self, edisgo):
        assert nx.is_connected(edisgo.topology.to_graph())
        switches = list(edisgo.topology.mv_grid.switch_disconnectors)
        assert len(switches) == 6
        assert {s.state for s in switches} == {"open"}
        # closing a switch closes its ring
        switches[0].close()
        assert len(nx.cycle_basis(edisgo.topology.to_graph())) == 1
        switches[0].open()

    def test_ding0_names(self, edisgo):
        topology = edisgo.topology
        assert topology.mv_grid.station.index.tolist() == ["Bus_MVStation_1101"]
        assert topology.get_lv_grid(1108).station.index.tolist() == [
            "BusBar_MVGrid_1101_LVGrid_1108_LV"
        ]
        assert topology.transformers_df.index.tolist() == [
            "LVStation_1108_transformer_1"
        ]
        assert topology.buses_df.index.is_unique
        switches = topology.switches_df
        virtual = switches.bus_open == "virtual_" + switches.bus_closed
        # one open switch per closed side is named after it
        assert virtual.sum() == switches.bus_closed.nunique()

    def test_load_types(self, edisgo):
        loads = edisgo.topology.loads_df
        profiles = pd.read_csv(
            f"{pytest.simbench_test_network_path}/Load.csv", sep=";", index_col="id"
        ).profile
        heat_pumps = profiles.str.startswith(("Air_", "Soil_"))
        home_charging = profiles.str.startswith("HLS_")
        assert set(loads.index[loads.type == "heat_pump"]) == set(
            profiles.index[heat_pumps]
        )
        assert set(loads.index[loads.type == "charging_point"]) == set(
            profiles.index[home_charging]
        )
        assert (loads.type == "conventional_load").sum() == (
            ~heat_pumps & ~home_charging
        ).sum()

    def test_time_series_are_scaled_profiles(self, edisgo):
        path = pytest.simbench_test_network_path
        load = pd.read_csv(f"{path}/Load.csv", sep=";", index_col="id").iloc[0]
        profiles = pd.read_csv(f"{path}/LoadProfile.csv", sep=";")
        ts = edisgo.timeseries
        assert len(ts.timeindex) == 2 * 96
        assert ts.timeindex[0] == pd.Timestamp("2011-01-01")
        np.testing.assert_allclose(
            ts.loads_active_power[load.name].values,
            load.pLoad * profiles[f"{load.profile}_pload"].values,
        )
        np.testing.assert_allclose(
            ts.loads_reactive_power[load.name].values,
            load.qLoad * profiles[f"{load.profile}_qload"].values,
        )

    def test_pv_with_battery_is_feed_in(self, edisgo):
        # SimBench PV storage units are PV systems with a home battery; their
        # profile is the feed-in of both, given in the load sign convention
        path = pytest.simbench_test_network_path
        storage = pd.read_csv(f"{path}/Storage.csv", sep=";", index_col="id")
        profiles = pd.read_csv(f"{path}/StorageProfile.csv", sep=";")
        assert edisgo.topology.storage_units_df.empty
        generators = edisgo.topology.generators_df.loc[storage.index]
        assert (generators.type == "solar").all()
        assert (generators.subtype == storage.type).all()
        unit = storage.iloc[0]
        feed_in = edisgo.timeseries.generators_active_power[unit.name]
        np.testing.assert_allclose(
            feed_in.values, -unit.pStor * profiles[unit.profile].values
        )
        assert (feed_in >= 0).all()

    def test_power_flow(self, edisgo):
        edisgo.analyze()
        assert edisgo.results.v_res.notna().all().all()

    def test_reinforce(self):
        edisgo = EDisGo()
        timeindex = pd.date_range("2011-01-01 18:00", periods=2, freq="15min")
        edisgo.import_simbench_grid(
            pytest.simbench_test_network_path, timeindex=timeindex
        )
        # overload the MV/LV station
        lv_loads = edisgo.topology.get_lv_grid(1108).loads_df.index
        loads_p = edisgo.timeseries.loads_active_power
        loads_p[lv_loads] *= 10
        edisgo.timeseries.loads_active_power = loads_p
        edisgo.reinforce()
        transformers = edisgo.topology.transformers_df.index
        assert len(transformers) > 1
        assert transformers.is_unique
        assert transformers.str.startswith("LVStation_1108_transformer").all()

    def test_timeindex(self):
        edisgo = EDisGo()
        timeindex = pd.date_range("2011-01-02", periods=4, freq="15min")
        edisgo.import_simbench_grid(
            pytest.simbench_test_network_path, timeindex=timeindex
        )
        assert (edisgo.timeseries.timeindex == timeindex).all()
        assert (edisgo.timeseries.loads_active_power.index == timeindex).all()

    def test_without_time_series(self):
        edisgo = EDisGo()
        edisgo.import_simbench_grid(
            pytest.simbench_test_network_path, time_series=False
        )
        assert edisgo.timeseries.timeindex.empty
        assert edisgo.timeseries.loads_active_power.empty

    def test_separately_operated_busbars_are_connected(self, tmp_path):
        # open the busbar coupler, so that half of the feeders are only connected
        # to the MV station via the coupler
        def open_coupler(switches):
            switches.loc[switches.id == "MV1.101 MV Sectionalizer1", "cond"] = 0
            return switches

        path = copy_with_changed_table(tmp_path, "Switch", open_coupler)
        edisgo = EDisGo()
        edisgo.import_simbench_grid(path, time_series=False)
        assert nx.is_connected(edisgo.topology.to_graph())

    def test_grid_without_storage(self, tmp_path):
        # SimBench grids of scenario 0 have no storage units
        path = tmp_path / "simbench_grid"
        shutil.copytree(pytest.simbench_test_network_path, path)
        (path / "Storage.csv").unlink()
        (path / "StorageProfile.csv").unlink()
        edisgo = EDisGo()
        edisgo.import_simbench_grid(str(path))
        generators = edisgo.topology.generators_df
        assert pd.api.types.is_float_dtype(generators.p_nom)
        assert set(edisgo.timeseries.generators_active_power.columns) == set(
            generators.index
        )
        assert edisgo.topology.storage_units_df.empty

    def test_path_without_grid(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No SimBench grid found"):
            EDisGo().import_simbench_grid(str(tmp_path))

    def test_grid_without_mv_grid(self, tmp_path):
        def drop_hvmv_transformers(trafos):
            return trafos[trafos.voltLvl != 4]

        path = copy_with_changed_table(tmp_path, "Transformer", drop_hvmv_transformers)
        with pytest.raises(ValueError, match="no HV/MV transformer"):
            EDisGo().import_simbench_grid(path, time_series=False)
