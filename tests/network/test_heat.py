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
            "Heat_Pump_LVGrid_1163850014_district_heating_6_2",
            "Heat_Pump_LVGrid_1163850014_district_heating_6_3",
        ]
        building_ids = [442081, None, 430859, None, None]
        sector = [
            "individual_heating",
            "district_heating",
            "individual_heating",
            "district_heating_resistive_heater",
            "district_heating_resistive_heater",
        ]
        weather_cell_ids = [11051, 11051, 11052, 11051, 11052]
        district_heating_ids = [None, 5, None, 5, 6]
        area_ids = [None, 4, None, 4, 5]
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

    def test_set_cop(self):

        # ################### test with dataframe ###################
        heat_pump = HeatPump()
        cop = pd.DataFrame(
            data={
                "hp3": [5.0, 6.0],
            },
            index=self.timeindex,
        )
        heat_pump.set_cop(None, cop)
        pd.testing.assert_frame_equal(
            heat_pump.cop_df,
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
        heat_pump.set_cop(None, cop)
        pd.testing.assert_frame_equal(
            heat_pump.cop_df,
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

        # test with heat_pump_names empty
        edisgo_object.heat_pump.set_cop(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            heat_pump_names=[],
        )
        assert edisgo_object.heat_pump.cop_df.empty

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
            edisgo_object.heat_pump.set_cop(
                edisgo_object,
                "oedb",
                engine=pytest.engine,
                heat_pump_names=heat_pump_names,
            )
        assert "There are heat pumps with no weather cell ID." in caplog.text
        assert edisgo_object.heat_pump.cop_df.shape == (8760, 6)
        assert (
            edisgo_object.heat_pump.cop_df.loc[
                :, "Heat_Pump_LVGrid_1163850014_district_heating_6_2"
            ]
            == 0.99
        ).all()
        assert (
            edisgo_object.heat_pump.cop_df.loc[
                :, "Heat_Pump_LVGrid_1163850014_district_heating_6_3"
            ]
            == 0.99
        ).all()
        assert (
            edisgo_object.heat_pump.cop_df.loc[:, "HP_442081"]
            == edisgo_object.heat_pump.cop_df.loc[
                :, "Heat_Pump_LVGrid_1163850014_district_heating_6"
            ]
        ).all()

    def test_set_heat_demand(self):
        # test with dataframe
        heat_pump = HeatPump()
        heat_demand = pd.DataFrame(
            data={
                "hp3": [1.0, 2.0],
                "hp4": [1.0, 2.0],
            },
            index=self.timeindex,
        )
        heat_pump.set_heat_demand(None, heat_demand)
        pd.testing.assert_frame_equal(
            heat_pump.heat_demand_df,
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
        heat_pump.set_heat_demand(None, heat_demand)
        heat_demand["hp4"] = pd.Series([1.0, 2.0], index=self.timeindex)
        pd.testing.assert_frame_equal(
            heat_pump.heat_demand_df[sorted(heat_pump.heat_demand_df)],
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
        edisgo_object.heat_pump.set_heat_demand(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            scenario="eGon2035",
        )
        assert edisgo_object.heat_pump.heat_demand_df.shape == (8760, 5)
        assert edisgo_object.heat_pump.heat_demand_df.index[0].year == 2035

        # ###### test with timeindex to get year from and invalid heat pump name #####
        # reset heat_demand_df
        edisgo_object.heat_pump.heat_demand_df = pd.DataFrame()
        edisgo_object.set_timeindex(
            pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        )
        edisgo_object.heat_pump.set_heat_demand(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            scenario="eGon2035",
            heat_pump_names=["HP_442081", "HP_dummy"],
        )
        assert edisgo_object.heat_pump.heat_demand_df.shape == (2, 1)
        assert edisgo_object.heat_pump.heat_demand_df.index[0].year == 2011

        # ###### test with empty list for heat pump names #####
        edisgo_object.heat_pump.set_heat_demand(
            edisgo_object,
            "oedb",
            engine=pytest.engine,
            scenario="eGon2035",
            heat_pump_names=[],
        )
        assert edisgo_object.heat_pump.heat_demand_df.shape == (2, 1)
        assert edisgo_object.heat_pump.heat_demand_df.index[0].year == 2011

    def test_reduce_memory(self):

        heatpump = HeatPump()
        heatpump.cop_df = self.cop
        heatpump.heat_demand_df = self.heat_demand

        # check with default value
        assert (heatpump.cop_df.dtypes == "float64").all()
        assert (heatpump.heat_demand_df.dtypes == "float64").all()

        heatpump.reduce_memory()

        assert (heatpump.cop_df.dtypes == "float32").all()
        assert (heatpump.heat_demand_df.dtypes == "float32").all()

        # check arguments
        heatpump.reduce_memory(to_type="float16", attr_to_reduce=["cop_df"])

        assert (heatpump.cop_df.dtypes == "float16").all()
        assert (heatpump.heat_demand_df.dtypes == "float32").all()

        # check with empty dataframes
        heatpump.heat_demand_df = pd.DataFrame()
        heatpump.reduce_memory()

    def test_to_csv(self):

        heatpump = HeatPump()
        heatpump.cop_df = self.cop
        heatpump.heat_demand_df = self.heat_demand
        heatpump.thermal_storage_units_df = self.tes

        # test with default values
        save_dir = os.path.join(os.getcwd(), "heat_pump_csv")
        heatpump.to_csv(save_dir)

        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 3
        assert "cop.csv" in files_in_dir
        assert "heat_demand.csv" in files_in_dir
        assert "thermal_storage_units.csv" in files_in_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16
        heatpump.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (heatpump.cop_df.dtypes == "float16").all()
        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 3

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):

        heatpump = HeatPump()
        heatpump.cop_df = self.cop
        heatpump.heat_demand_df = self.heat_demand
        heatpump.thermal_storage_units_df = self.tes

        # write to csv
        save_dir = os.path.join(os.getcwd(), "heat_pump_csv")
        heatpump.to_csv(save_dir)

        # reset HeatPump object
        heatpump = HeatPump()

        heatpump.from_csv(save_dir)

        pd.testing.assert_frame_equal(
            heatpump.cop_df,
            self.cop,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            heatpump.heat_demand_df,
            self.heat_demand,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            heatpump.thermal_storage_units_df,
            self.tes,
        )

        shutil.rmtree(save_dir)

    def test_resample_timeseries(self):
        heatpump = HeatPump()
        heatpump.cop_df = self.cop
        heatpump.thermal_storage_units_df = self.tes

        # test up-sampling of COP with default parameters
        heatpump.resample_timeseries()
        assert len(heatpump.cop_df) == 8
        assert (heatpump.cop_df.iloc[0:4, 0] == 5).all()
        assert (heatpump.cop_df.iloc[4:8, 1] == 8).all()

        # test up-sampling of heat demand with default parameters
        heatpump.heat_demand_df = self.heat_demand
        heatpump.resample_timeseries(method="bfill")
        assert len(heatpump.heat_demand_df) == 8
        assert heatpump.heat_demand_df.iloc[0, 0] == 1
        assert (heatpump.heat_demand_df.iloc[1:8, 1] == 4).all()
        assert (heatpump.cop_df.iloc[0:4, 0] == 5).all()
        assert (heatpump.cop_df.iloc[4:8, 1] == 8).all()

        # test down-sampling
        heatpump.resample_timeseries(freq="1H")
        assert len(heatpump.heat_demand_df) == 2
        assert len(heatpump.cop_df) == 2

    def test_check_integrity(self, caplog):
        # check for empty HeatPump class
        heatpump = HeatPump()
        with caplog.at_level(logging.WARNING):
            heatpump.check_integrity()
        assert len(caplog.text) == 0

        caplog.clear()

        # create duplicate entries and loads that do not appear in each DSM dataframe
        heat_demand_df = pd.concat(
            [
                self.heat_demand,
                pd.DataFrame(
                    data={
                        "hp1": [1.0, 2.0],
                        "hp3": [3.0, 4.0],
                    },
                    index=self.timeindex,
                ),
            ],
            axis=1,
        )
        heatpump.cop_df = self.cop
        heatpump.heat_demand_df = heat_demand_df

        with caplog.at_level(logging.WARNING):
            heatpump.check_integrity()
        assert len(caplog.messages) == 2
        assert (
            "HeatPump timeseries heat_demand_df contains the following duplicates:"
            in caplog.text
        )
        assert (
            "HeatPump timeseries cop_df is missing the following entries:"
            in caplog.text
        )
