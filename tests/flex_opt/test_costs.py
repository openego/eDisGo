import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.costs import grid_expansion_costs, line_expansion_costs


class TestCosts:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.edisgo.analyze()

    def test_costs(self):
        # manually add one reinforced HV/MV and MV/LV transformer
        hv_mv_trafo = self.edisgo.topology.transformers_hvmv_df.loc[
            "MVStation_1_transformer_1"
        ]
        hv_mv_trafo.name = "MVStation_1_transformer_reinforced_2"
        self.edisgo.topology.transformers_hvmv_df = pd.concat(
            [
                self.edisgo.topology.transformers_hvmv_df,
                hv_mv_trafo.to_frame().T,
            ]
        )
        mv_lv_trafo = self.edisgo.topology.transformers_df.loc[
            "LVStation_1_transformer_1"
        ]
        mv_lv_trafo.name = "LVStation_1_transformer_reinforced_1"
        self.edisgo.topology.transformers_df.drop(
            "LVStation_1_transformer_1", inplace=True
        )
        self.edisgo.topology.transformers_df = pd.concat(
            [
                self.edisgo.topology.transformers_df,
                mv_lv_trafo.to_frame().T,
            ]
        )

        self.edisgo.results.equipment_changes = pd.DataFrame(
            {
                "iteration_step": [1, 1, 1, 1, 1, 2, 4, 0],
                "change": [
                    "added",
                    "added",
                    "removed",
                    "changed",
                    "changed",
                    "changed",
                    "changed",
                    "added",
                ],
                "equipment": [
                    "MVStation_1_transformer_reinforced_2",
                    "LVStation_1_transformer_reinforced_1",
                    "LVStation_1_transformer_1",
                    "NA2XS2Y 3x1x185 RM/25",
                    "48-AL1/8-ST1A",
                    "NA2XS2Y 3x1x185 RM/25",
                    "NAYY 4x1x35",
                    "dummy_gen",
                ],
                "quantity": [1, 1, 1, 2, 1, 1, 3, 1],
            },
            index=[
                "MVGrid_1_station",
                "LVGrid_1_station",
                "LVGrid_1_station",
                "Line_10006",
                "Line_10019",
                "Line_10019",
                "Line_50000002",
                "dummy_gen",
            ],
        )

        costs = grid_expansion_costs(self.edisgo)

        assert len(costs) == 4
        assert (
            costs.loc["MVStation_1_transformer_reinforced_2", "voltage_level"]
            == "hv/mv"
        )
        assert costs.loc["MVStation_1_transformer_reinforced_2", "quantity"] == 1
        assert costs.loc["MVStation_1_transformer_reinforced_2", "total_costs"] == 1000
        assert (
            costs.loc["LVStation_1_transformer_reinforced_1", "voltage_level"]
            == "mv/lv"
        )
        assert costs.loc["LVStation_1_transformer_reinforced_1", "quantity"] == 1
        assert costs.loc["LVStation_1_transformer_reinforced_1", "total_costs"] == 10
        assert np.isclose(costs.loc["Line_10019", "total_costs"], 32.3082)
        assert np.isclose(costs.loc["Line_10019", "length"], 0.40385)
        assert costs.loc["Line_10019", "quantity"] == 1
        assert costs.loc["Line_10019", "type"] == "48-AL1/8-ST1A"
        assert costs.loc["Line_10019", "voltage_level"] == "mv"
        assert np.isclose(costs.loc["Line_50000002", "total_costs"], 2.34)
        assert np.isclose(costs.loc["Line_50000002", "length"], 0.09)
        assert costs.loc["Line_50000002", "quantity"] == 3
        assert costs.loc["Line_50000002", "type"] == "NAYY 4x1x35"
        assert costs.loc["Line_50000002", "voltage_level"] == "lv"

    def test_line_expansion_costs(self):
        costs = line_expansion_costs(self.edisgo)
        assert len(costs) == len(self.edisgo.topology.lines_df)
        assert (costs.index == self.edisgo.topology.lines_df.index).all()
        assert len(costs[costs.voltage_level == "mv"]) == len(
            self.edisgo.topology.mv_grid.lines_df
        )
        assert np.isclose(costs.at["Line_10003", "costs_earthworks"], 0.083904 * 60)
        assert np.isclose(costs.at["Line_10003", "costs_cable"], 0.083904 * 20)
        assert costs.at["Line_10003", "voltage_level"] == "mv"
        assert np.isclose(costs.at["Line_10000015", "costs_earthworks"], 1.53)
        assert np.isclose(costs.at["Line_10000015", "costs_cable"], 0.27)
        assert costs.at["Line_10000015", "voltage_level"] == "lv"

        costs = line_expansion_costs(self.edisgo, ["Line_10003", "Line_10000015"])
        assert len(costs) == 2
        assert (costs.index.values == ["Line_10003", "Line_10000015"]).all()
        assert np.isclose(costs.at["Line_10003", "costs_earthworks"], 0.083904 * 60)
        assert np.isclose(costs.at["Line_10003", "costs_cable"], 0.083904 * 20)
        assert costs.at["Line_10003", "voltage_level"] == "mv"
        assert np.isclose(costs.at["Line_10000015", "costs_earthworks"], 1.53)
        assert np.isclose(costs.at["Line_10000015", "costs_cable"], 0.27)
        assert costs.at["Line_10000015", "voltage_level"] == "lv"
