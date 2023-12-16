import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.io import pypsa_io
from edisgo.network.results import Results


class TestPypsaIO:
    def test_to_pypsa(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        timeindex = self.edisgo.timeseries.timeindex

        # test mode None
        pypsa_network = pypsa_io.to_pypsa(self.edisgo, timesteps=timeindex)
        slack_df = pypsa_network.generators[pypsa_network.generators.control == "Slack"]
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == "Bus_MVStation_1"
        # ToDo: Check further things

        # test mode "lv" and single time step
        lv_grid = self.edisgo.topology.get_lv_grid(1)
        pypsa_network = pypsa_io.to_pypsa(
            self.edisgo, timesteps=timeindex[0], mode="lv", lv_grid_id=lv_grid.id
        )
        slack_df = pypsa_network.generators[pypsa_network.generators.control == "Slack"]
        assert len(slack_df) == 1
        assert slack_df.bus.values[0] == lv_grid.station.index[0]
        assert len(pypsa_network.buses) == 15
        # ToDo: Check further things and parameter options

    def test_append_lv_components(self):
        lv_components = {
            "Load": pd.DataFrame(),
            "Generator": pd.DataFrame(),
            "StorageUnit": pd.DataFrame(),
        }
        comps = pd.DataFrame({"bus": []})
        # check if returns when comps is empty
        pypsa_io._append_lv_components("Unkown", comps, lv_components, "TestGrid")
        # check exceptions for wrong input parameters
        comps = pd.DataFrame({"bus": ["bus1"], "p_set": [0.1]}, index=["dummy"])
        msg = "Component type not defined."
        with pytest.raises(ValueError, match=msg):
            pypsa_io._append_lv_components("Unkown", comps, lv_components, "TestGrid")
        msg = "Aggregation type for loads invalid."
        with pytest.raises(ValueError, match=msg):
            pypsa_io._append_lv_components(
                "Load",
                comps,
                lv_components,
                "TestGrid",
                aggregate_loads="unknown",
            )
        msg = "Aggregation type for generators invalid."
        with pytest.raises(ValueError, match=msg):
            pypsa_io._append_lv_components(
                "Generator",
                comps,
                lv_components,
                "TestGrid",
                aggregate_generators="unknown",
            )
        msg = "Aggregation type for storages invalid."
        with pytest.raises(ValueError, match=msg):
            pypsa_io._append_lv_components(
                "StorageUnit",
                comps,
                lv_components,
                "TestGrid",
                aggregate_storages="unknown",
            )
        # check appending aggregated elements to lv_components in different
        # modes
        # CHECK GENERATORS
        gens = pd.DataFrame(
            {
                "bus": ["LVStation"] * 6,
                "control": ["PQ"] * 6,
                "p_nom": [0.05, 0.23, 0.04, 0.2, 0.1, 0.4],
                "type": ["solar", "wind", "solar", "solar", "gas", "wind"],
            },
            index=[
                "Solar_1",
                "Wind_1",
                "Solar_2",
                "Solar_3",
                "Gas_1",
                "Wind_2",
            ],
        )
        # check not aggregated generators
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens,
            lv_components,
            "TestGrid",
            aggregate_generators=None,
        )
        assert len(aggr_dict) == 0
        assert len(lv_components["Generator"]) == 6
        assert_frame_equal(
            gens.loc[:, ["bus", "control", "p_nom"]],
            lv_components["Generator"].loc[:, ["bus", "control", "p_nom"]],
        )
        assert (
            lv_components["Generator"].fluctuating
            == [True, True, True, True, False, True]
        ).all()
        # check aggregation of generators by type
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens,
            lv_components,
            "TestGrid",
            aggregate_generators="type",
        )
        assert len(aggr_dict) == 3
        assert (aggr_dict["TestGrid_gas"] == ["Gas_1"]).all()
        assert (aggr_dict["TestGrid_solar"] == ["Solar_1", "Solar_2", "Solar_3"]).all()
        assert (aggr_dict["TestGrid_wind"] == ["Wind_1", "Wind_2"]).all()
        assert len(lv_components["Generator"]) == 3
        assert (lv_components["Generator"].control == "PQ").all()
        assert (lv_components["Generator"].bus == "LVStation").all()
        assert (
            lv_components["Generator"].index.values
            == ["TestGrid_gas", "TestGrid_solar", "TestGrid_wind"]
        ).all()
        assert np.isclose(lv_components["Generator"].p_nom, [0.1, 0.29, 0.63]).all()
        assert (lv_components["Generator"].fluctuating == [False, True, True]).all()
        # check if only one type is existing
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens.loc[gens.type == "solar"],
            lv_components,
            "TestGrid",
            aggregate_generators="type",
        )
        assert len(aggr_dict) == 1
        assert (aggr_dict["TestGrid_solar"] == ["Solar_1", "Solar_2", "Solar_3"]).all()
        assert len(lv_components["Generator"]) == 1
        assert lv_components["Generator"].index.values == ["TestGrid_solar"]
        assert np.isclose(lv_components["Generator"].p_nom, 0.29)
        assert (lv_components["Generator"].fluctuating == [True]).all()
        # check aggregation of generators by fluctuating or dispatchable
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens,
            lv_components,
            "TestGrid",
            aggregate_generators="curtailable",
        )
        assert len(aggr_dict) == 2
        assert (
            aggr_dict["TestGrid_fluctuating"]
            == ["Solar_1", "Wind_1", "Solar_2", "Solar_3", "Wind_2"]
        ).all()
        assert (aggr_dict["TestGrid_dispatchable"] == ["Gas_1"]).all()
        assert len(lv_components["Generator"]) == 2
        assert (lv_components["Generator"].control == "PQ").all()
        assert (lv_components["Generator"].bus == "LVStation").all()
        assert (
            lv_components["Generator"].index.values
            == ["TestGrid_fluctuating", "TestGrid_dispatchable"]
        ).all()
        assert np.isclose(lv_components["Generator"].p_nom, [0.92, 0.1]).all()
        assert (lv_components["Generator"].fluctuating == [True, False]).all()
        # check if only dispatchable gens are given
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens.loc[gens.type == "gas"],
            lv_components,
            "TestGrid",
            aggregate_generators="curtailable",
        )
        assert len(aggr_dict) == 1
        assert (aggr_dict["TestGrid_dispatchable"] == ["Gas_1"]).all()
        assert len(lv_components["Generator"]) == 1
        assert lv_components["Generator"].index.values == ["TestGrid_dispatchable"]
        assert np.isclose(lv_components["Generator"].p_nom, 0.1)
        assert (lv_components["Generator"].fluctuating == [False]).all()
        # check if only fluctuating gens are given
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens.drop(gens.loc[gens.type == "gas"].index),
            lv_components,
            "TestGrid",
            aggregate_generators="curtailable",
        )
        assert len(aggr_dict) == 1
        assert (
            aggr_dict["TestGrid_fluctuating"]
            == ["Solar_1", "Wind_1", "Solar_2", "Solar_3", "Wind_2"]
        ).all()
        assert len(lv_components["Generator"]) == 1
        assert lv_components["Generator"].index.values == ["TestGrid_fluctuating"]
        assert np.isclose(lv_components["Generator"].p_nom, 0.92)
        assert (lv_components["Generator"].fluctuating == [True]).all()
        # check aggregation of all generators
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens,
            lv_components,
            "TestGrid",
            aggregate_generators="all",
        )
        assert len(aggr_dict) == 1
        assert (
            aggr_dict["TestGrid_generators"]
            == ["Solar_1", "Wind_1", "Solar_2", "Solar_3", "Gas_1", "Wind_2"]
        ).all()
        assert len(lv_components["Generator"]) == 1
        assert (lv_components["Generator"].control == "PQ").all()
        assert (lv_components["Generator"].bus == "LVStation").all()
        assert (
            lv_components["Generator"].index.values == ["TestGrid_generators"]
        ).all()
        assert np.isclose(lv_components["Generator"].p_nom, 1.02)
        assert (lv_components["Generator"].fluctuating == ["Mixed"]).all()
        # check only fluctuating
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens.drop(gens.loc[gens.type == "gas"].index),
            lv_components,
            "TestGrid",
            aggregate_generators="all",
        )
        assert len(aggr_dict) == 1
        assert (
            aggr_dict["TestGrid_generators"]
            == ["Solar_1", "Wind_1", "Solar_2", "Solar_3", "Wind_2"]
        ).all()
        assert len(lv_components["Generator"]) == 1
        assert (lv_components["Generator"].control == "PQ").all()
        assert (lv_components["Generator"].bus == "LVStation").all()
        assert (
            lv_components["Generator"].index.values == ["TestGrid_generators"]
        ).all()
        assert np.isclose(lv_components["Generator"].p_nom, 0.92)
        assert (lv_components["Generator"].fluctuating == [True]).all()
        # check only dispatchable
        lv_components["Generator"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Generator",
            gens.loc[gens.type == "gas"],
            lv_components,
            "TestGrid",
            aggregate_generators="all",
        )
        assert len(aggr_dict) == 1
        assert (aggr_dict["TestGrid_generators"] == ["Gas_1"]).all()
        assert len(lv_components["Generator"]) == 1
        assert (lv_components["Generator"].control == "PQ").all()
        assert (lv_components["Generator"].bus == "LVStation").all()
        assert (
            lv_components["Generator"].index.values == ["TestGrid_generators"]
        ).all()
        assert np.isclose(lv_components["Generator"].p_nom, 0.1)
        assert (lv_components["Generator"].fluctuating == [False]).all()
        lv_components["Generator"] = pd.DataFrame()
        # CHECK LOADS
        loads = pd.DataFrame(
            {
                "bus": ["LVStation"] * 6,
                "p_set": [0.05, 0.23, 0.04, 0.2, 0.1, 0.4],
                "sector": [
                    "retail",
                    "agricultural",
                    "retail",
                    "retail",
                    "industrial",
                    "agricultural",
                ],
            },
            index=[
                "Retail_1",
                "Agricultural_1",
                "Retail_2",
                "Retail_3",
                "Industrial_1",
                "Agricultural_2",
            ],
        )
        # check not aggregated loads
        aggr_dict = pypsa_io._append_lv_components(
            "Load", loads, lv_components, "TestGrid", aggregate_loads=None
        )
        assert len(aggr_dict) == 0
        assert len(lv_components["Load"]) == 6
        assert (loads.p_set.values == lv_components["Load"].p_set.values).all()
        assert (lv_components["Load"].bus == "LVStation").all()
        assert (lv_components["Load"].index == loads.index).all()
        # check aggregate loads by sector
        lv_components["Load"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Load",
            loads,
            lv_components,
            "TestGrid",
            aggregate_loads="sectoral",
        )
        assert len(aggr_dict) == 3
        assert (
            aggr_dict["TestGrid_agricultural"] == ["Agricultural_1", "Agricultural_2"]
        ).all()
        assert (aggr_dict["TestGrid_industrial"] == ["Industrial_1"]).all()
        assert (
            aggr_dict["TestGrid_retail"] == ["Retail_1", "Retail_2", "Retail_3"]
        ).all()
        assert len(lv_components["Load"]) == 3
        assert (lv_components["Load"].bus == "LVStation").all()
        assert (
            lv_components["Load"].index.values
            == [
                "TestGrid_agricultural",
                "TestGrid_industrial",
                "TestGrid_retail",
            ]
        ).all()
        assert np.isclose(lv_components["Load"].p_set, [0.63, 0.1, 0.29]).all()
        # check if only one sector exists
        lv_components["Load"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Load",
            loads.loc[loads.sector == "industrial"],
            lv_components,
            "TestGrid",
            aggregate_loads="sectoral",
        )
        assert len(aggr_dict) == 1
        assert (aggr_dict["TestGrid_industrial"] == ["Industrial_1"]).all()
        assert len(lv_components["Load"]) == 1
        assert (lv_components["Load"].bus == "LVStation").all()
        assert (lv_components["Load"].index.values == ["TestGrid_industrial"]).all()
        assert np.isclose(lv_components["Load"].p_set, 0.1).all()
        # check aggregation of all loads
        lv_components["Load"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "Load", loads, lv_components, "TestGrid", aggregate_loads="all"
        )
        assert len(aggr_dict) == 1
        assert (
            aggr_dict["TestGrid_loads"]
            == [
                "Retail_1",
                "Agricultural_1",
                "Retail_2",
                "Retail_3",
                "Industrial_1",
                "Agricultural_2",
            ]
        ).all()
        assert len(lv_components["Load"]) == 1
        assert (lv_components["Load"].bus == "LVStation").all()
        assert (lv_components["Load"].index.values == ["TestGrid_loads"]).all()
        assert np.isclose(lv_components["Load"].p_set, 1.02).all()
        lv_components["Load"] = pd.DataFrame()
        # CHECK STORAGES
        storages = pd.DataFrame(
            {"bus": ["LVStation"] * 2, "control": ["PQ"] * 2},
            index=["Storage_1", "Storage_2"],
        )
        # check appending without aggregation
        aggr_dict = pypsa_io._append_lv_components(
            "StorageUnit",
            storages,
            lv_components,
            "TestGrid",
            aggregate_storages=None,
        )
        assert len(aggr_dict) == 0
        assert len(lv_components["StorageUnit"]) == 2
        assert (lv_components["StorageUnit"].bus == "LVStation").all()
        assert (lv_components["StorageUnit"].control == "PQ").all()
        assert (
            lv_components["StorageUnit"].index.values == ["Storage_1", "Storage_2"]
        ).all()
        # check aggregration of all storages
        lv_components["StorageUnit"] = pd.DataFrame()
        aggr_dict = pypsa_io._append_lv_components(
            "StorageUnit",
            storages,
            lv_components,
            "TestGrid",
            aggregate_storages="all",
        )
        assert len(aggr_dict) == 1
        assert (aggr_dict["TestGrid_storages"] == ["Storage_1", "Storage_2"]).all()
        assert len(lv_components["StorageUnit"]) == 1
        assert (lv_components["StorageUnit"].bus == "LVStation").all()
        assert (lv_components["StorageUnit"].control == "PQ").all()
        assert lv_components["StorageUnit"].index.values == "TestGrid_storages"

    def test_get_generators_timeseries_with_aggregated_elements(self):
        pass

    def test_set_seed(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        timeindex = self.edisgo.timeseries.timeindex

        # test with missing busses

        # set up results for first time step and MV busses
        self.edisgo.analyze(timesteps=timeindex[0], mode="mv")
        # create pypsa network for first time step and all busses
        pypsa_network = self.edisgo.to_pypsa(timesteps=timeindex[0])
        pypsa_io.set_seed(self.edisgo, pypsa_network)

        # check that for LV busses default values are used and for MV busses
        # results from previous power flow
        lv_bus = "Bus_BranchTee_LVGrid_1_10"
        mv_bus = "BusBar_MVGrid_1_LVGrid_3_MV"
        assert pypsa_network.buses_t.v_mag_pu.loc[timeindex[0], lv_bus] == 1.0
        assert pypsa_network.buses_t.v_ang.loc[timeindex[0], lv_bus] == 0.0
        assert (
            pypsa_network.buses_t.v_mag_pu.loc[timeindex[0], mv_bus]
            == self.edisgo.results.pfa_v_mag_pu_seed.loc[timeindex[0], mv_bus]
        )
        assert (
            pypsa_network.buses_t.v_ang.loc[timeindex[0], mv_bus]
            == self.edisgo.results.pfa_v_ang_seed.loc[timeindex[0], mv_bus]
        )
        # run power flow to check if it converges
        pypsa_network.pf(use_seed=True)
        # write results to edisgo object
        pypsa_io.process_pfa_results(self.edisgo, pypsa_network, timeindex)

        # test with missing time steps
        pypsa_network = self.edisgo.to_pypsa()
        pypsa_io.set_seed(self.edisgo, pypsa_network)

        # check that second time step default values are used and for first
        # time steps results from previous power flow
        assert (
            pypsa_network.buses_t.v_mag_pu.loc[timeindex[0], lv_bus]
            == self.edisgo.results.pfa_v_mag_pu_seed.loc[timeindex[0], lv_bus]
        )
        assert (
            pypsa_network.buses_t.v_ang.loc[timeindex[0], lv_bus]
            == self.edisgo.results.pfa_v_ang_seed.loc[timeindex[0], lv_bus]
        )
        assert (
            pypsa_network.buses_t.v_mag_pu.loc[timeindex[0], mv_bus]
            == self.edisgo.results.pfa_v_mag_pu_seed.loc[timeindex[0], mv_bus]
        )
        assert (
            pypsa_network.buses_t.v_ang.loc[timeindex[0], mv_bus]
            == self.edisgo.results.pfa_v_ang_seed.loc[timeindex[0], mv_bus]
        )
        assert pypsa_network.buses_t.v_mag_pu.loc[timeindex[1], lv_bus] == 1.0
        assert pypsa_network.buses_t.v_ang.loc[timeindex[1], lv_bus] == 0.0
        assert pypsa_network.buses_t.v_mag_pu.loc[timeindex[1], mv_bus] == 1.0
        assert pypsa_network.buses_t.v_ang.loc[timeindex[1], mv_bus] == 0.0
        # run power flow to check if it converges
        pypsa_network.pf(use_seed=True)

        # test with seed for all busses and time steps available from previous
        # power flow analyses (and at the same time check, if results from
        # different power flow analyses are appended correctly)

        # reset results
        self.edisgo.results = Results(self.edisgo)
        # run power flow separately for both time steps
        self.edisgo.analyze(timesteps=timeindex[0])
        self.edisgo.analyze(timesteps=timeindex[1])
        pypsa_network = self.edisgo.to_pypsa()
        pypsa_io.set_seed(self.edisgo, pypsa_network)

        # check that for both time steps results from previous power flow
        # analyses are used
        assert (
            pypsa_network.buses_t.v_mag_pu.loc[timeindex[0], lv_bus]
            == self.edisgo.results.pfa_v_mag_pu_seed.loc[timeindex[0], lv_bus]
        )
        assert (
            pypsa_network.buses_t.v_ang.loc[timeindex[1], lv_bus]
            == self.edisgo.results.pfa_v_ang_seed.loc[timeindex[1], lv_bus]
        )
        assert (
            pypsa_network.buses_t.v_mag_pu.loc[timeindex[1], mv_bus]
            == self.edisgo.results.pfa_v_mag_pu_seed.loc[timeindex[1], mv_bus]
        )
        assert (
            pypsa_network.buses_t.v_ang.loc[timeindex[0], mv_bus]
            == self.edisgo.results.pfa_v_ang_seed.loc[timeindex[0], mv_bus]
        )
        # run power flow to check if it converges
        pypsa_network.pf(use_seed=True)

    def test_process_pfa_results(self):
        # test update of seed
        pass
