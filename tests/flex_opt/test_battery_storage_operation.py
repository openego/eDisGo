import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.battery_storage_operation import apply_reference_operation


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
