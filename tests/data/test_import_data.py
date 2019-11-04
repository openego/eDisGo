import pytest
import shapely
import os

from edisgo.network.topology import Topology
from edisgo.network.grids import MVGrid, LVGrid
from edisgo.data import import_data


class TestImportFromDing0:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.topology = Topology()
        import_data.import_ding0_grid(test_network_directory, self)

    def test_import_ding0_grid(self):
        """Test successful import of ding0 topology."""

        # buses, generators, loads, lines, transformers dataframes
        # check number of imported components
        assert self.topology.buses_df.shape[0] == 208
        assert self.topology.generators_df.shape[0] == 28
        assert self.topology.loads_df.shape[0] == 50
        assert self.topology.lines_df.shape[0] == 198
        assert self.topology.transformers_df.shape[0] == 9
        assert self.topology.switches_df.shape[0] == 2
        assert self.topology.storages_df.shape[0] == 0
        # check necessary columns
        assert all([col in self.topology.buses_df.columns for col in
                    import_data.COLUMNS['buses_df']])
        assert all([col in self.topology.generators_df.columns
                    for col in import_data.COLUMNS['generators_df']])
        assert all([col in self.topology.loads_df.columns for col in
                    import_data.COLUMNS['loads_df']])
        assert all([col in self.topology.transformers_df.columns
                    for col in import_data.COLUMNS['transformers_df']])
        assert all([col in self.topology.lines_df.columns for col in
                    import_data.COLUMNS['lines_df']])
        assert all([col in self.topology.switches_df.columns for col in
                    import_data.COLUMNS['switches_df']])
        assert all([col in self.topology.storages_df.columns for col in
                    import_data.COLUMNS['storages_df']])

        # topology district
        assert self.topology.grid_district['population'] == 23358
        assert isinstance(self.topology.grid_district['geom'],
                          shapely.geometry.Polygon)

        # grids
        assert isinstance(self.topology.mv_grid, MVGrid)
        assert len(self.topology._grids) == 10
        lv_grid = [_ for _ in self.topology.mv_grid.lv_grids if _.id == 3][0]
        assert isinstance(lv_grid, LVGrid)

    def test_path_error(self):
        """Test catching error when path to topology does not exist."""
        msg = "Specified directory containing ding0 topology data does not " \
              "exist or does not contain topology data."
        with pytest.raises(AttributeError, match=msg):
            import_data.import_ding0_grid('wrong_directory', self.topology)

    def test_validate_ding0_grid_import(self):
        """Test of validation of grids."""
        comps_dict = {'buses': 'Bus_primary_LVStation_2',
                      'generators': 'GeneratorFluctuating_14',
                      'loads': 'Load_residential_LVGrid_3_2',
                      'transformers': 'LVStation_5_transformer_1',
                      'lines': 'Line_10014',
                      'switches': 'circuit_breaker_1'}
        # check duplicate node
        for comp, name in comps_dict.items():
            new_comp = getattr(self.topology, '_{}_df'.format(comp)).loc[name]
            comps = getattr(self.topology, '_{}_df'.format(comp))
            setattr(self.topology, '_{}_df'.format(comp),
                    comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.topology)
                raise Exception('Appending components {} in check duplicate '
                                'did not work properly.'.format(comp))
            except ValueError as e:
                assert e.args[0] == '{} have duplicate entry in one ' \
                                    'of the components dataframes.'.format(
                    name)
            # reset dataframe
            setattr(self.topology, '_{}_df'.format(comp), comps)
            import_data._validate_ding0_grid_import(self.topology)

        # check not connected generator and load
        for nodal_component in ["loads", "generators"]:
            comps = getattr(self.topology, '_{}_df'.format(nodal_component))
            new_comp = comps.loc[comps_dict[nodal_component]]
            new_comp.name = 'new_nodal_component'
            new_comp.bus = 'Non_existent_bus_' + nodal_component
            setattr(self.topology, '_{}_df'.format(nodal_component),
                    comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.topology)
                raise Exception('Appending components {} did not work '
                                'properly.'.format(nodal_component))
            except ValueError as e:
                assert e.args[0] == 'The following {} have buses which are ' \
                                    'not defined: {}.'.format(
                    nodal_component, new_comp.name)
            # reset dataframe
            setattr(self.topology, '_{}_df'.format(nodal_component), comps)
            import_data._validate_ding0_grid_import(self.topology)

        # check branch components
        i = 0
        for branch_component in ["lines", "transformers"]:
            comps = getattr(self.topology, '_{}_df'.format(branch_component))
            new_comp = comps.loc[comps_dict[branch_component]]
            new_comp.name = 'new_branch_component'
            setattr(new_comp, 'bus' + str(i),
                    'Non_existent_bus_' + branch_component)
            setattr(self.topology, '_{}_df'.format(branch_component),
                    comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.topology)
                raise Exception('Appending components {} did not work '
                                'properly.'.format(branch_component))
            except ValueError as e:
                assert e.args[0] == 'The following {} have bus{} which are ' \
                                    'not defined: {}.'.format(
                    branch_component, i, new_comp.name)
            # reset dataframe
            setattr(self.topology, '_{}_df'.format(branch_component), comps)
            import_data._validate_ding0_grid_import(self.topology)
            i += 1

        # check switches
        comps = self.topology.switches_df
        for attr in ["bus_open", "bus_closed"]:
            new_comp = comps.loc[comps_dict['switches']]
            new_comp.name = 'new_switch'
            new_comps = comps.append(new_comp)
            new_comps.at[new_comp.name, attr] = 'Non_existent_' + attr
            self.topology.switches_df = new_comps
            try:
                import_data._validate_ding0_grid_import(self.topology)
                raise Exception('Appending components switches did not work '
                                'properly.')
            except ValueError as e:
                assert e.args[0] == 'The following switches have {} which ' \
                                    'are not defined: {}.'.format(
                    attr, new_comp.name)
            self.topology.switches_df = comps
            import_data._validate_ding0_grid_import(self.topology)

        # check isolated node
        bus = self.topology.buses_df.loc[comps_dict['buses']]
        bus.name = 'New_bus'
        self.topology.buses_df = self.topology.buses_df.append(bus)
        try:
            import_data._validate_ding0_grid_import(self.topology)
            raise Exception('Appending components buses did not work '
                            'properly.')
        except ValueError as e:
            assert e.args[0] == 'The following buses are isolated: ' \
                                '{}.'.format(bus.name)
