import logging
import os
import shutil

import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.network.heat import HeatPump


class TestHeatPump:
    @classmethod
    def setup_class(cls):
        # set up data used in tests
        cls.heatpump = HeatPump()

        cls.timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        cls.cop = pd.DataFrame(
            data={
                "hp1": [5.0, 6.0],
                "hp2": [7.0, 8.0],
            },
            index=cls.timeindex,
        )
        cls.heat_demand = pd.DataFrame(
            data={
                "hp1": [1.0, 2.0],
                "hp2": [3.0, 4.0],
            },
            index=cls.timeindex,
        )
        cls.tes = pd.DataFrame(
            data={
                "capacity": [9.5, 10.5],
                "efficiency": [0.9, 0.8],
                "state_of_charge_initial": [0.5, 0.4],
            },
            index=["hp1", "hp2"],
        )

    def setup_egon_heat_pump_data(self):
        names = [
            "HP_442081",
            "Heat_Pump_LVGrid_1163850014_district_heating_6",
            "HP_448156",
        ]
        building_ids = [442081, None, 448156]
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

    def test_set_cop(self):

        # ################### test with dataframe ###################
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
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

        # test if concatenating works correctly
        cop = pd.DataFrame(
            data={
                "hp3": [1.0, 2.0],
                "hp2": [3.0, 4.0],
            },
            index=self.timeindex,
        )
        self.heatpump.set_cop(self.edisgo, cop)
        pd.testing.assert_frame_equal(
            self.heatpump.cop_df,
            cop,
            check_freq=False,
        )

    @pytest.mark.local
    def test_set_cop_oedb(self, caplog):

        # ################### test with oedb ###################
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

        # test with missing weather cell information (column does not exist) - raises
        # ValueError
        msg = "In order to obtain COP time series data from database"
        with pytest.raises(ValueError, match=msg):
            edisgo_object.heat_pump.set_cop(
                edisgo_object,
                "oedb",
                engine=pytest.engine,
                heat_pump_names=edisgo_object.topology.loads_df.index[0:4],
            )

        # test with missing weather cell information (column exists but all values
        # None) - raises ValueError
        with pytest.raises(ValueError, match=msg):
            edisgo_object.topology.loads_df["weather_cell_id"] = None
            edisgo_object.heat_pump.set_cop(
                edisgo_object,
                "oedb",
                engine=pytest.engine,
                heat_pump_names=edisgo_object.topology.loads_df.index[0:4],
            )

        # test with missing weather cell information (some values None) - raises
        # warning
        hp_data_egon = self.setup_egon_heat_pump_data()
        edisgo_object.topology.loads_df = pd.concat(
            [edisgo_object.topology.loads_df, hp_data_egon]
        )
        heat_pump_names = hp_data_egon.index.append(
            edisgo_object.topology.loads_df.index[0:1]
        )
        with caplog.at_level(logging.WARNING):
            self.heatpump.set_cop(
                edisgo_object,
                "oedb",
                engine=pytest.engine,
                heat_pump_names=heat_pump_names,
            )
        assert "There are heat pumps with no weather cell ID." in caplog.text
        assert self.heatpump.cop_df.shape == (8760, 4)

    def test_set_heat_demand(self):
        # test with dataframe
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        heat_demand = pd.DataFrame(
            data={
                "hp3": [1.0, 2.0],
                "hp4": [1.0, 2.0],
            },
            index=self.timeindex,
        )
        self.heatpump.set_heat_demand(self.edisgo, heat_demand)
        pd.testing.assert_frame_equal(
            self.heatpump.heat_demand_df,
            heat_demand,
            check_freq=False,
        )

        # test if concatenating works correctly
        heat_demand = pd.DataFrame(
            data={
                "hp3": [1.0, 2.0],
                "hp2": [3.0, 4.0],
            },
            index=self.timeindex,
        )
        self.heatpump.set_heat_demand(self.edisgo, heat_demand)
        heat_demand["hp4"] = pd.Series([1.0, 2.0], index=self.timeindex)
        pd.testing.assert_frame_equal(
            self.heatpump.heat_demand_df[sorted(self.heatpump.heat_demand_df)],
            heat_demand[sorted(heat_demand.columns)],
            check_freq=False,
        )

    @pytest.mark.local
    def test_set_heat_demand_oedb(self):
        # test with oedb
        edisgo_object = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        hp_data_egon = self.setup_egon_heat_pump_data()
        edisgo_object.topology.loads_df = pd.concat(
            [edisgo_object.topology.loads_df, hp_data_egon]
        )

        # ################# test with no timeindex to get year from #############
        self.heatpump.set_heat_demand(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            scenario="eGon2035",
        )
        assert self.heatpump.heat_demand_df.shape == (8760, 3)
        assert self.heatpump.heat_demand_df.index[0].year == 2035

        # ###### test with timeindex to get year from and invalid heat pump name #####
        edisgo_object.set_timeindex(
            pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        )
        self.heatpump.set_heat_demand(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            scenario="eGon2035",
            heat_pump_names=["HP_442081", "HP_dummy"],
        )
        assert self.heatpump.heat_demand_df.shape == (8760, 1)
        assert self.heatpump.heat_demand_df.index[0].year == 2011

    def test_reduce_memory(self):

        self.heatpump.cop_df = self.cop
        self.heatpump.heat_demand_df = self.heat_demand

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

        self.heatpump.cop_df = self.cop
        self.heatpump.heat_demand_df = self.heat_demand
        self.heatpump.thermal_storage_units_df = self.tes

        # test with default values
        save_dir = os.path.join(os.getcwd(), "heat_pump_csv")
        self.heatpump.to_csv(save_dir)

        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 3
        assert "cop.csv" in files_in_dir
        assert "heat_demand.csv" in files_in_dir
        assert "thermal_storage_units.csv" in files_in_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16
        self.heatpump.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (self.heatpump.cop_df.dtypes == "float16").all()
        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 3

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):

        self.heatpump.cop_df = self.cop
        self.heatpump.heat_demand_df = self.heat_demand
        self.heatpump.thermal_storage_units_df = self.tes

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

        shutil.rmtree(save_dir)
