import pandas as pd
import pytest

from numpy.random import default_rng

from edisgo import EDisGo
from edisgo.flex_opt.heat_pump_operation import operating_strategy


class TestHeatPumpOperation:
    @classmethod
    def setup_class(self):
        self.timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")

        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path, timeindex=self.timeindex
        )

        # insert heat pumps into topology
        rng = default_rng(1)
        number_of_heat_pumps = 2
        self.name = []
        for i in range(number_of_heat_pumps):
            self.name.append(
                self.edisgo.add_component(
                    comp_type="load",
                    bus=rng.choice(self.edisgo.topology.buses_df.index, size=1)[0],
                    p_set=0.43,
                    type="heat_pump",
                )
            )

        # add one addtional hp that is not in the topology
        self.name.append("hp" + str(len(self.name) + 1))

        self.cop = pd.DataFrame(
            data={
                self.name[0]: [5.0, 6.0],
                self.name[1]: [7.0, 8.0],
                self.name[2]: [7.2, 6.7],
            },
            index=self.timeindex,
        )
        self.heat_demand = pd.DataFrame(
            data={
                self.name[0]: [1.0, 4.0],
                self.name[1]: [3.0, 4.0],
                self.name[2]: [2.0, 3.0],
            },
            index=self.timeindex,
        )

        self.edisgo.heat_pump.cop_df = self.cop
        self.edisgo.heat_pump.heat_demand_df = self.heat_demand

    def test_operating_strategy(self):
        # test with default parameters
        operating_strategy(self.edisgo)

        hp_ts = pd.DataFrame(
            data={
                self.name[0]: [0.2, 0.43],
                self.name[1]: [3 / 7, 0.43],
                self.name[2]: [2 / 7.2, 3 / 6.7],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.loads_active_power,
            hp_ts,
        )
        hp_ts = pd.DataFrame(
            data={
                self.name[0]: [0.0, 0.0],
                self.name[1]: [0.0, 0.0],
                self.name[2]: [0.0, 0.0],
            },
            index=self.timeindex,
        )
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.loads_reactive_power,
            hp_ts,
        )

        # test with providing heat pump names
        timestep = self.timeindex[0]
        self.edisgo.heat_pump.heat_demand_df.at[timestep, self.name[0]] = 0.0
        self.edisgo.heat_pump.heat_demand_df.at[timestep, self.name[1]] = 0.0

        operating_strategy(self.edisgo, heat_pump_names=[self.name[0]])

        assert (
            self.edisgo.timeseries.loads_active_power.at[timestep, self.name[0]] == 0.0
        )
        assert (
            self.edisgo.timeseries.loads_active_power.at[timestep, self.name[1]]
            == 3 / 7
        )

        # test error raising
        msg = "Heat pump operating strategy dummy is not a valid option."
        with pytest.raises(ValueError, match=msg):
            operating_strategy(self.edisgo, strategy="dummy")
