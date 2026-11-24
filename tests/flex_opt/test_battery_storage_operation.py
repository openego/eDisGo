import copy

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.battery_storage_operation import apply_reference_operation
from edisgo.io import storage_import


class TestStorageOperation:
    @classmethod
    def setup_class(self):
        self.timeindex = pd.date_range("1/1/2011 12:00", periods=5, freq="H")
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path, timeindex=self.timeindex
        )
        self.edisgo.topology.storage_units_df = pd.DataFrame(
            data={
                "bus": [
                    "Bus_BranchTee_LVGrid_2_4",
                    "Bus_BranchTee_LVGrid_2_4",
                    "Bus_BranchTee_LVGrid_2_4",
                    "Bus_BranchTee_LVGrid_2_4",
                    "Bus_BranchTee_LVGrid_2_4",
                ],
                "control": ["PQ", "PQ", "PQ", "PQ", "PQ"],
                "p_nom": [0.2, 2.0, 0.4, 0.5, 0.6],
                "max_hours": [6, 6, 1, 6, 6],
                "efficiency_store": [0.9, 1.0, 0.9, 1.0, 0.8],
                "efficiency_dispatch": [0.9, 1.0, 0.9, 1.0, 0.8],
                "building_id": [1, 2, 3, 4, 5],
            },
            index=["stor1", "stor2", "stor3", "stor4", "stor5"],
        )
        # set building IDs
        self.edisgo.topology.loads_df.at[
            "Load_residential_LVGrid_8_2", "building_id"
        ] = 2
        self.edisgo.topology.loads_df.at[
            "Load_residential_LVGrid_8_3", "building_id"
        ] = 2
        self.edisgo.topology.generators_df.at[
            "GeneratorFluctuating_25", "building_id"
        ] = 2
        self.edisgo.topology.generators_df.at[
            "GeneratorFluctuating_26", "building_id"
        ] = 2
        self.edisgo.topology.loads_df.at[
            "Load_residential_LVGrid_3_2", "building_id"
        ] = 3.0
        self.edisgo.topology.generators_df.at[
            "GeneratorFluctuating_17", "building_id"
        ] = 3.0
        self.edisgo.topology.loads_df.at[
            "Load_residential_LVGrid_1_6", "building_id"
        ] = 4
        self.edisgo.topology.loads_df.at[
            "Load_residential_LVGrid_1_4", "building_id"
        ] = 5.0
        self.edisgo.topology.generators_df.at[
            "GeneratorFluctuating_27", "building_id"
        ] = 5.0
        # set time series
        self.edisgo.timeseries.loads_active_power = pd.DataFrame(
            data={
                "Load_residential_LVGrid_8_2": [0.5, 1.0, 1.5, 0.0, 0.5],
                "Load_residential_LVGrid_8_3": [0.5, 1.0, 1.5, 0.0, 0.5],
                "Load_residential_LVGrid_3_2": [0.5, 0.0, 1.0, 0.5, 0.5],
                "Load_residential_LVGrid_1_4": [0.0, 1.0, 1.5, 0.0, 0.5],
            },
            index=self.timeindex,
        )
        self.edisgo.timeseries.generators_active_power = pd.DataFrame(
            data={
                "GeneratorFluctuating_25": [1.5, 3.0, 4.5, 0.0, 0.0],
                "GeneratorFluctuating_26": [0.5, 1.0, 1.5, 0.0, 0.5],
                "GeneratorFluctuating_17": [0.0, 1.0, 1.5, 1.0, 0.0],
                "GeneratorFluctuating_27": [0.5, 0.0, 0.5, 0.0, 0.0],
            },
            index=self.timeindex,
        )

    def test_operating_strategy(self):
        # test without load (stor1)
        # test with several loads and several PV systems (stor2)
        # test with one load and one PV system (stor3)
        # test without PV system (stor4)
        # test with one value not numeric (stor5)

        # test with providing storage name
        apply_reference_operation(edisgo_obj=self.edisgo, storage_units_names=["stor1"])

        check_ts = pd.DataFrame(
            data={
                "stor1": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_active_power,
            check_ts,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_reactive_power,
            check_ts,
        )

        # test without providing storage names
        soe_df = apply_reference_operation(edisgo_obj=self.edisgo)

        assert soe_df.shape == (5, 3)
        assert self.edisgo.timeseries.storage_units_active_power.shape == (5, 5)
        assert self.edisgo.timeseries.storage_units_reactive_power.shape == (5, 5)

        # check stor2
        s = "stor2"
        check_ts = pd.DataFrame(
            data={
                s: [-1.0, -2.0, -2.0, 0.0, 0.5],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_active_power.loc[:, [s]],
            check_ts,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, [s]],
            check_ts * -np.tan(np.arccos(0.95)),
        )
        check_ts = pd.DataFrame(
            data={
                s: [1.0, 3.0, 5.0, 5.0, 4.5],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            soe_df.loc[:, [s]],
            check_ts,
        )

        # check stor3
        s = "stor3"
        check_ts = pd.DataFrame(
            data={
                s: [0.0, -0.4, -0.044444, 0.0, 0.36],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_active_power.loc[:, [s]],
            check_ts,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, [s]],
            check_ts * -np.tan(np.arccos(0.95)),
        )
        check_ts = pd.DataFrame(
            data={
                s: [0.0, 0.36, 0.4, 0.4, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            soe_df.loc[:, [s]],
            check_ts,
        )

        # check stor4 - all zeros
        s = "stor4"
        check_ts = pd.DataFrame(
            data={
                s: [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_active_power.loc[:, [s]],
            check_ts,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, [s]],
            check_ts,
        )
        # check stor5
        s = "stor5"
        check_ts = pd.DataFrame(
            data={
                s: [-0.5, 0.32, 0.0, 0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_active_power.loc[:, [s]],
            check_ts,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.storage_units_reactive_power.loc[:, [s]],
            check_ts * -np.tan(np.arccos(0.95)),
        )
        check_ts = pd.DataFrame(
            data={
                s: [0.4, 0.0, 0.0, 0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            soe_df.loc[:, [s]],
            check_ts,
        )

        # test error raising
        self.edisgo.topology.storage_units_df.at["stor5", "max_hours"] = np.nan
        msg = (
            "Parameter max_hours for storage unit stor5 is not a number. It needs "
            "to be set in Topology.storage_units_df."
        )
        with pytest.raises(ValueError, match=msg):
            apply_reference_operation(self.edisgo)

    def test_default_efficiencies(self):
        # reference run with efficiencies set in storage_units_df (stor3: 0.9)
        edisgo_ref = copy.deepcopy(self.edisgo)
        soe_ref = apply_reference_operation(edisgo_ref, storage_units_names=["stor3"])

        # eGon home batteries come without efficiencies
        edisgo = copy.deepcopy(self.edisgo)
        stor_df = edisgo.topology.storage_units_df
        stor_df.loc["stor3", ["efficiency_store", "efficiency_dispatch"]] = np.nan
        soe_df = apply_reference_operation(
            edisgo,
            storage_units_names=["stor3"],
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
        )
        pd.testing.assert_frame_equal(soe_df, soe_ref)
        # filled values are not written to the topology
        assert pd.isna(stor_df.at["stor3", "efficiency_store"])

        # default of 0.95 (stor3 is charged with p_nom of 0.4 MW in time step 1)
        soe_df = apply_reference_operation(edisgo, storage_units_names=["stor3"])
        assert soe_df.iat[1, 0] == pytest.approx(0.4 * 0.95)

    def test_check_inputs(self):
        # no time index set
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.timeseries.timeindex = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="No time index is set."):
            apply_reference_operation(edisgo, storage_units_names=["stor3"])

        # building IDs missing, e.g. generators not imported from eGon database
        for df_name, hint in [
            ("storage_units_df", "import_home_batteries"),
            ("generators_df", "import_generators"),
            ("loads_df", "ding0 grid created from eGon data"),
        ]:
            edisgo = copy.deepcopy(self.edisgo)
            df = getattr(edisgo.topology, df_name).drop(columns=["building_id"])
            setattr(edisgo.topology, df_name, df)
            msg = f"Topology.{df_name} has no column 'building_id'.*{hint}"
            with pytest.raises(ValueError, match=msg):
                apply_reference_operation(edisgo, storage_units_names=["stor3"])

        # time series of PV plant or load missing
        for comp_type, comp in [
            ("generators", "GeneratorFluctuating_17"),
            ("loads", "Load_residential_LVGrid_3_2"),
        ]:
            edisgo = copy.deepcopy(self.edisgo)
            ts_name = f"{comp_type}_active_power"
            ts_df = getattr(edisgo.timeseries, ts_name).drop(columns=[comp])
            setattr(edisgo.timeseries, ts_name, ts_df)
            msg = f"No active power time series set for {comp_type} {comp}."
            with pytest.raises(ValueError, match=msg):
                apply_reference_operation(edisgo, storage_units_names=["stor3"])

        # missing time series of loads in buildings without PV plant are not needed
        # (stor4 has a load without time series but no PV plant)
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.set_time_series_manual(
            storage_units_p=pd.DataFrame({"stor4": 1.0}, index=self.timeindex)
        )
        apply_reference_operation(edisgo, storage_units_names=["stor4"])
        assert (edisgo.timeseries.storage_units_active_power["stor4"] == 0.0).all()

    def test_default_storage_units(self):
        edisgo = copy.deepcopy(self.edisgo)
        stor_df = edisgo.topology.storage_units_df
        # undo NaN set in test_operating_strategy
        stor_df.at["stor5", "max_hours"] = 6
        # storage unit without building ID (e.g. large-scale storage) keeps its
        # time series when no storage names are given
        stor_df.loc["stor_large"] = stor_df.loc["stor2"]
        stor_df.at["stor_large", "building_id"] = np.nan
        ts_large = pd.DataFrame({"stor_large": 1.0}, index=self.timeindex)
        edisgo.set_time_series_manual(storage_units_p=ts_large)

        soe_df = apply_reference_operation(edisgo)
        assert sorted(soe_df.columns) == ["stor2", "stor3", "stor5"]
        pd.testing.assert_frame_equal(
            edisgo.timeseries.storage_units_active_power.loc[:, ["stor_large"]],
            ts_large,
        )

        # no storage unit with building ID
        stor_df["building_id"] = np.nan
        soe_df = apply_reference_operation(edisgo)
        assert soe_df.empty
        pd.testing.assert_index_equal(soe_df.index, self.timeindex)

    def test_freq(self):
        # frequency is derived from time index (15 min -> 0.25 h)
        edisgo = copy.deepcopy(self.edisgo)
        timeindex = pd.date_range("1/1/2011 12:00", periods=5, freq="15min")
        ts_names = ["generators_active_power", "loads_active_power"]
        ts_dfs = {_: getattr(edisgo.timeseries, _) for _ in ts_names}
        edisgo.timeseries.timeindex = timeindex
        for ts_name, ts_df in ts_dfs.items():
            setattr(edisgo.timeseries, ts_name, ts_df.set_axis(timeindex))

        soe_df = apply_reference_operation(edisgo, storage_units_names=["stor3"])
        # stor3 is charged with p_nom of 0.4 MW for a quarter of an hour
        assert soe_df.at[timeindex[1], "stor3"] == pytest.approx(0.4 * 0.9 * 0.25)


class TestStorageOperationEgon:
    """
    Tests the reference operation with home batteries as they are integrated from the
    eGon database, i.e. without efficiencies.

    """

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.timeindex = pd.date_range("1/1/2011 12:00", periods=4, freq="H")
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path,
            legacy_ding0_grids=False,
            timeindex=self.timeindex,
        )
        # home batteries as retrieved from the eGon database
        batteries_df = pd.DataFrame(
            data={
                "p_nom": [0.005, 0.005],
                "capacity": [0.01, 0.01],
                "building_id": [445710, 446933],
            },
            index=[1, 2],
        )
        storage_import._home_batteries_grid_integration(self.edisgo, batteries_df)
        self.load_pv = "Load_mvgd_33535_lvgd_1164120011_19_residential"
        self.load_no_pv = "Load_mvgd_33535_lvgd_1164210000_230_residential"

    def test_apply_home_storage_operating_strategy(self):
        storage_df = self.edisgo.topology.storage_units_df
        assert storage_df.efficiency_store.isna().all()
        assert storage_df.efficiency_dispatch.isna().all()

        # generators not imported from eGon database yet, so building IDs are missing
        msg = "Topology.generators_df has no column 'building_id'.*import_generators"
        with pytest.raises(ValueError, match=msg):
            self.edisgo.apply_home_storage_operating_strategy()

        # assign PV plant to building 445710, as done by EDisGo.import_generators()
        pv = self.edisgo.topology.generators_df[
            self.edisgo.topology.generators_df.type == "solar"
        ].index[0]
        self.edisgo.topology.generators_df["building_id"] = np.nan
        self.edisgo.topology.generators_df.at[pv, "building_id"] = 445710
        self.edisgo.set_time_series_manual(
            generators_p=pd.DataFrame(
                {pv: [0.0, 0.007, 0.007, 0.0]}, index=self.timeindex
            ),
            loads_p=pd.DataFrame(
                {self.load_pv: 0.002, self.load_no_pv: 0.002}, index=self.timeindex
            ),
        )

        soe_df = self.edisgo.apply_home_storage_operating_strategy()

        # default efficiencies of 0.95 are used: no charge in time step 0, charging
        # with p_nom in time steps 1 and 2, discharging to cover demand in time step 3
        pd.testing.assert_frame_equal(
            soe_df,
            pd.DataFrame(
                {"Storage_445710": [0.0, 0.00475, 0.0095, 0.007395]},
                index=self.timeindex,
            ),
        )
        storage_ts = self.edisgo.timeseries.storage_units_active_power
        assert (storage_ts["Storage_446933"] == 0.0).all()
        assert storage_ts.loc[self.timeindex[1], "Storage_445710"] == -0.005
        assert not storage_ts.isna().any().any()
        # filled efficiencies are not written to the topology
        assert storage_df.efficiency_store.isna().all()

        # parameters are passed on to apply_reference_operation
        soe_df = self.edisgo.apply_home_storage_operating_strategy(efficiency_store=1.0)
        assert soe_df.at[self.timeindex[1], "Storage_445710"] == 0.005

        msg = "Operating strategy 'dumb' for home storage units is not valid."
        with pytest.raises(ValueError, match=msg):
            self.edisgo.apply_home_storage_operating_strategy(strategy="dumb")
