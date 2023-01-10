import os
import shutil

import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.network.heat import HeatPump


class TestHeatPump:
    @classmethod
    def setup_class(self):
        self.timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        self.cop = pd.DataFrame(
            data={
                "hp1": [5.0, 6.0],
                "hp2": [7.0, 8.0],
            },
            index=self.timeindex,
        )
        self.heat_demand = pd.DataFrame(
            data={
                "hp1": [1.0, 2.0],
                "hp2": [3.0, 4.0],
            },
            index=self.timeindex,
        )
        self.tes = pd.DataFrame(
            data={
                "capacity": [9.5, 10.5],
                "efficiency": [0.9, 0.8],
                "state_of_charge_initial": [0.5, 0.4],
            },
            index=["hp1", "hp2"],
        )
        self.building_ids = pd.DataFrame(
            data={
                "building_ids": [[1], [2, 3]],
            },
            index=["hp1", "hp2"],
        )

    @pytest.fixture(autouse=True)
    def setup_heat_pump(self):
        self.heatpump = HeatPump()
        self.heatpump.cop_df = self.cop
        self.heatpump.heat_demand_df = self.heat_demand
        self.heatpump.thermal_storage_units_df = self.tes
        self.heatpump.building_ids_df = self.building_ids

    def test_set_cop(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        # test with dataframe
        cop = pd.DataFrame(
            data={
                "hp3": [5.0, 6.0],
            },
            index=self.timeindex,
        )
        self.heatpump.set_cop(self.edisgo, cop)
        pd.testing.assert_frame_equal(
            self.heatpump.cop_df,
            cop,
            check_freq=False,
        )
        # ToDo: test with oedb

    def test_set_heat_demand(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        # test with dataframe
        heat_demand = pd.DataFrame(
            data={
                "hp3": [1.0, 2.0],
            },
            index=self.timeindex,
        )
        self.heatpump.set_heat_demand(self.edisgo, heat_demand)
        pd.testing.assert_frame_equal(
            self.heatpump.heat_demand_df,
            heat_demand,
            check_freq=False,
        )
        # ToDo: test with oedb

    def test_reduce_memory(self):

        # check with default value
        assert (self.heatpump.cop_df.dtypes == "float64").all()
        assert (self.heatpump.heat_demand_df.dtypes == "float64").all()

        self.heatpump.reduce_memory()

        assert (self.heatpump.cop_df.dtypes == "float32").all()
        assert (self.heatpump.heat_demand_df.dtypes == "float32").all()

        # check arguments
        self.heatpump.reduce_memory(to_type="float16", attr_to_reduce=["cop_df"])

        assert (self.heatpump.cop_df.dtypes == "float16").all()
        assert (self.heatpump.heat_demand_df.dtypes == "float32").all()

        # check with empty dataframes
        self.heatpump.heat_demand_df = pd.DataFrame()
        self.heatpump.reduce_memory()

    def test_to_csv(self):

        # test with default values
        save_dir = os.path.join(os.getcwd(), "heat_pump_csv")
        self.heatpump.to_csv(save_dir)

        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 4
        assert "cop.csv" in files_in_dir
        assert "heat_demand.csv" in files_in_dir
        assert "thermal_storage_units.csv" in files_in_dir
        assert "building_ids.csv" in files_in_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16
        self.heatpump.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (self.heatpump.cop_df.dtypes == "float16").all()
        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 4

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):

        # write to csv
        save_dir = os.path.join(os.getcwd(), "heat_pump_csv")
        self.heatpump.to_csv(save_dir)

        # reset HeatPump object
        self.heatpump = HeatPump()

        self.heatpump.from_csv(save_dir)

        pd.testing.assert_frame_equal(
            self.heatpump.cop_df,
            self.cop,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.heatpump.heat_demand_df,
            self.heat_demand,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.heatpump.thermal_storage_units_df,
            self.tes,
        )
        pd.testing.assert_frame_equal(
            self.heatpump.building_ids_df,
            self.building_ids,
        )

        shutil.rmtree(save_dir)
