import pytest
import shapely

from edisgo.io import ding0_import
from edisgo.network.grids import LVGrid, MVGrid
from edisgo.network.topology import Topology


class TestImportFromDing0:
    @classmethod
    def setup_class(self):
        self.topology = Topology()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

    def test_import_ding0_grid(self):
        """Test successful import of ding0 network."""

        # buses, generators, loads, lines, transformers dataframes
        # check number of imported components
        assert self.topology.buses_df.shape[0] == 140
        assert self.topology.generators_df.shape[0] == 28
        assert self.topology.loads_df.shape[0] == 50
        assert self.topology.lines_df.shape[0] == 129
        assert self.topology.transformers_df.shape[0] == 14
        assert self.topology.transformers_hvmv_df.shape[0] == 1
        assert self.topology.switches_df.shape[0] == 2
        assert self.topology.storage_units_df.shape[0] == 1

        # grid district
        assert self.topology.grid_district["population"] == 23358
        assert isinstance(self.topology.grid_district["geom"], shapely.geometry.Polygon)

        # grids
        assert isinstance(self.topology.mv_grid, MVGrid)
        assert len(self.topology._grids) == 11
        lv_grid = [_ for _ in self.topology.mv_grid.lv_grids if _.id == 3][0]
        assert isinstance(lv_grid, LVGrid)

    def test_path_error(self):
        """Test catching error when path to network does not exist."""
        msg = "Directory wrong_directory does not exist."
        with pytest.raises(AssertionError, match=msg):
            ding0_import.import_ding0_grid("wrong_directory", self.topology)

    def test_validate_ding0_grid_import(self):
        """Test of validation of grids."""
        comps_dict = {
            "buses": "BusBar_MVGrid_1_LVGrid_2_MV",
            "generators": "GeneratorFluctuating_14",
            "loads": "Load_residential_LVGrid_3_2",
            "transformers": "LVStation_5_transformer_1",
            "lines": "Line_10014",
            "switches": "circuit_breaker_1",
        }
        # check duplicate node
        for comp, name in comps_dict.items():
            new_comp = getattr(self.topology, "_{}_df".format(comp)).loc[name]
            comps = getattr(self.topology, "_{}_df".format(comp))
            setattr(self.topology, "_{}_df".format(comp), comps.append(new_comp))
            try:
                ding0_import._validate_ding0_grid_import(self.topology)
                raise Exception(
                    "Appending components {} in check duplicate "
                    "did not work properly.".format(comp)
                )
            except ValueError as e:
                assert e.args[
                    0
                ] == "{} have duplicate entry in one " "of the components dataframes.".format(
                    name
                )
            # reset dataframe
            setattr(self.topology, "_{}_df".format(comp), comps)
            ding0_import._validate_ding0_grid_import(self.topology)

        # check not connected generator and load
        for nodal_component in ["loads", "generators"]:
            comps = getattr(self.topology, "_{}_df".format(nodal_component))
            new_comp = comps.loc[comps_dict[nodal_component]]
            new_comp.name = "new_nodal_component"
            new_comp.bus = "Non_existent_bus_" + nodal_component
            setattr(
                self.topology, "_{}_df".format(nodal_component), comps.append(new_comp)
            )
            try:
                ding0_import._validate_ding0_grid_import(self.topology)
                raise Exception(
                    "Appending components {} did not work "
                    "properly.".format(nodal_component)
                )
            except ValueError as e:
                assert e.args[
                    0
                ] == "The following {} have buses which are " "not defined: {}.".format(
                    nodal_component, new_comp.name
                )
            # reset dataframe
            setattr(self.topology, "_{}_df".format(nodal_component), comps)
            ding0_import._validate_ding0_grid_import(self.topology)

        # check branch components
        i = 0
        for branch_component in ["lines", "transformers"]:
            comps = getattr(self.topology, "_{}_df".format(branch_component))
            new_comp = comps.loc[comps_dict[branch_component]]
            new_comp.name = "new_branch_component"
            setattr(new_comp, "bus" + str(i), "Non_existent_bus_" + branch_component)
            setattr(
                self.topology, "_{}_df".format(branch_component), comps.append(new_comp)
            )
            try:
                ding0_import._validate_ding0_grid_import(self.topology)
                raise Exception(
                    "Appending components {} did not work "
                    "properly.".format(branch_component)
                )
            except ValueError as e:
                assert e.args[
                    0
                ] == "The following {} have bus{} which are " "not defined: {}.".format(
                    branch_component, i, new_comp.name
                )
            # reset dataframe
            setattr(self.topology, "_{}_df".format(branch_component), comps)
            ding0_import._validate_ding0_grid_import(self.topology)
            i += 1

        # check switches
        comps = self.topology.switches_df
        for attr in ["bus_open", "bus_closed"]:
            new_comp = comps.loc[comps_dict["switches"]]
            new_comp.name = "new_switch"
            new_comps = comps.append(new_comp)
            new_comps.at[new_comp.name, attr] = "Non_existent_" + attr
            self.topology.switches_df = new_comps
            try:
                ding0_import._validate_ding0_grid_import(self.topology)
                raise Exception(
                    "Appending components switches did not work " "properly."
                )
            except ValueError as e:
                assert e.args[
                    0
                ] == "The following switches have {} which " "are not defined: {}.".format(
                    attr, new_comp.name
                )
            self.topology.switches_df = comps
            ding0_import._validate_ding0_grid_import(self.topology)

        # check isolated node
        bus = self.topology.buses_df.loc[comps_dict["buses"]]
        bus.name = "New_bus"
        self.topology.buses_df = self.topology.buses_df.append(bus)
        try:
            ding0_import._validate_ding0_grid_import(self.topology)
            raise Exception("Appending components buses did not work " "properly.")
        except ValueError as e:
            assert e.args[0] == "The following buses are isolated: " "{}.".format(
                bus.name
            )

    def test_transformer_buses(self):
        assert (
            self.topology.buses_df.loc[self.topology.transformers_df.bus1].v_nom.values
            < self.topology.buses_df.loc[
                self.topology.transformers_df.bus0
            ].v_nom.values
        ).all()
        self.topology.transformers_df.loc[
            "LVStation_7_transformer_1", "bus0"
        ] = "Bus_secondary_LVStation_7"
        self.topology.transformers_df.loc[
            "LVStation_7_transformer_1", "bus1"
        ] = "Bus_primary_LVStation_7"
        with pytest.raises(AssertionError):
            assert (
                self.topology.buses_df.reindex(
                    index=self.topology.transformers_df.bus1
                ).v_nom.values
                < self.topology.buses_df.reindex(
                    index=self.topology.transformers_df.bus0
                ).v_nom.values
            ).all()
