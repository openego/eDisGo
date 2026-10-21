import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.heat_pump_operation import operating_strategy


class TestHeatPumpOperation:
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
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path, timeindex=self.timeindex
        )
        self.edisgo.heat_pump.cop_df = self.cop
        self.edisgo.heat_pump.heat_demand_df = self.heat_demand

    def test_operating_strategy(self):
        # test with default parameters
        operating_strategy(self.edisgo)

        hp_ts = pd.DataFrame(
            data={
                "hp1": [0.2, 1 / 3],
                "hp2": [3 / 7, 0.5],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.loads_active_power,
            hp_ts,
        )
        hp_ts = pd.DataFrame(
            data={
                "hp1": [0.0, 0.0],
                "hp2": [0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.loads_reactive_power,
            hp_ts,
        )

        # test with providing heat pump names
        timestep = self.timeindex[0]
        self.edisgo.heat_pump.heat_demand_df.at[timestep, "hp1"] = 0.0
        self.edisgo.heat_pump.heat_demand_df.at[timestep, "hp2"] = 0.0

        operating_strategy(self.edisgo, heat_pump_names=["hp1"])

        assert self.edisgo.timeseries.loads_active_power.at[timestep, "hp1"] == 0.0
        assert self.edisgo.timeseries.loads_active_power.at[timestep, "hp2"] == 3 / 7

        # test error raising
        msg = "Heat pump operating strategy dummy is not a valid option."
        with pytest.raises(ValueError, match=msg):
            operating_strategy(self.edisgo, strategy="dummy")
