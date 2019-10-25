import pytest
import shapely
import os

from edisgo.grid.network import Network
from edisgo.grid.grids import MVGrid, LVGrid
from edisgo.data import import_data


class TestImportFromDing0:
    #ToDo add tests for switches_df and storages_df

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.network = Network()
        import_data.import_ding0_grid(test_network_directory, self.network)

    def test_import_ding0_grid_network(self):
        """Test successful import of ding0 grid - Network"""

        # buses, generators, loads, lines, transformers dataframes
        # check number of imported components
        assert self.network.buses_df.shape[0] == 208
        assert self.network.generators_df.shape[0] == 28
        assert self.network.loads_df.shape[0] == 50
        assert self.network.lines_df.shape[0] == 198
        assert self.network.transformers_df.shape[0] == 9
        # check necessary columns
        assert all([col in self.network.buses_df.columns for col in
                    import_data.COLUMNS['buses_df']])
        assert all([col in self.network.generators_df.columns
                    for col in import_data.COLUMNS['generators_df']])
        assert all([col in self.network.loads_df.columns for col in
                    import_data.COLUMNS['loads_df']])
        assert all([col in self.network.transformers_df.columns
                    for col in import_data.COLUMNS['transformers_df']])
        assert all([col in self.network.lines_df.columns for col in
                    import_data.COLUMNS['lines_df']])

        # grid district
        assert self.network.grid_district['population'] == 23358
        assert isinstance(self.network.grid_district['geom'],
                          shapely.geometry.Polygon)

        # grids
        assert isinstance(self.network.mv_grid, MVGrid)
        assert len(self.network._grids) == 10

    def test_import_ding0_grid_mv_grid(self):
        """Test successful import of ding0 grid - MVGrid"""

        #ToDo test generator_objects?

        mv_grid = self.network.mv_grid

        assert mv_grid.id == 1
        assert mv_grid.nominal_voltage == 20
        assert len(list(mv_grid.lv_grids)) == 9

        assert len(mv_grid.buses_df.index) == 33
        assert 'Bus_BranchTee_MVGrid_1_7' in mv_grid.buses_df.index
        assert len(mv_grid.generators_df.index) == 9
        assert 'Generator_slack' not in mv_grid.generators_df.index
        assert 'Generator_1' in mv_grid.generators_df.index
        assert len(mv_grid.loads_df.index) == 1
        assert 'Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1' in \
               mv_grid.loads_df.index

        assert sorted(mv_grid.weather_cells) == [1122074, 1122075]
        assert mv_grid.peak_generation_capacity == 22.075
        assert mv_grid.peak_generation_capacity_per_technology['solar'] == 4.6
        assert mv_grid.peak_load == 0.31
        assert mv_grid.peak_load_per_sector['retail'] == 0.31

    def test_import_ding0_grid_lv_grids(self):
        """Test successful import of ding0 grid - LVGrid"""

        lv_grid = [_ for _ in self.network.mv_grid.lv_grids if _.id == 3][0]

        assert isinstance(lv_grid, LVGrid)
        assert lv_grid.nominal_voltage == 0.4

        assert len(lv_grid.buses_df) == 13
        assert 'Bus_BranchTee_LVGrid_3_2' in lv_grid.buses_df.index
        assert len(lv_grid.generators_df.index) == 0
        assert len(lv_grid.loads_df.index) == 4
        assert 'Load_residential_LVGrid_3_3' in lv_grid.loads_df.index

        assert sorted(lv_grid.weather_cells) == []
        assert lv_grid.peak_generation_capacity == 0
        assert lv_grid.peak_generation_capacity_per_technology.empty
        assert lv_grid.peak_load == 0.054627
        assert lv_grid.peak_load_per_sector['agricultural'] == 0.051

    def test_path_error(self):
        """Test catching error when path to network does not exist."""
        msg = "Specified directory containing ding0 grid data does not " \
              "exist or does not contain grid data."
        with pytest.raises(AttributeError, match=msg):
            import_data.import_ding0_grid('wrong_directory', self.network)

    def test_validate_ding0_grid_import(self):
        """
        Test of validation of grids
        """
        comps_dict = {'buses': 'Bus_primary_LVStation_2',
                      'generators': 'GeneratorFluctuating_14',
                      'loads': 'Load_residential_LVGrid_3_2',
                      'transformers':'LVStation_5_transformer_1',
                      'lines':'Line_10014',
                      'switches':'circuit_breaker_1'}
        # check duplicate node
        for comp, name in comps_dict.items():
            new_comp = getattr(self.network, comp + '_df').loc[name]
            comps = getattr(self.network, comp + '_df')
            setattr(self.network, comp + '_df', comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.network)
                raise Exception('Appending components {} in check duplicate '
                                'did not work properly.'.format(comp))
            except ValueError as e:
                assert e.args[0] == '{} have duplicate entry in one ' \
                                    'of the components dataframes'.format(
                    name)
            setattr(self.network, comp + '_df', comps)
            import_data._validate_ding0_grid_import(self.network)
        print('Check duplicates finished.')

        # check not connected generator and load
        for nodal_component in ["loads", "generators"]:
            comps = getattr(self.network, nodal_component + '_df')
            new_comp = comps.loc[comps_dict[nodal_component]]
            new_comp.name = 'new_nodal_component'
            new_comp.bus = 'Non_existant_bus_'+nodal_component
            setattr(self.network, nodal_component + '_df', comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.network)
                raise Exception('Appending components {} did not work properly.'.format(nodal_component))
            except ValueError as e:
                assert e.args[0] == 'The following {} have buses which are ' \
                                    'not defined: {}'.format(
                    nodal_component, new_comp.name)
            setattr(self.network, nodal_component + '_df', comps)
            import_data._validate_ding0_grid_import(self.network)
        print('Check nodal components finished.')

        # check branch components
        i = 0
        for branch_component in ["lines", "transformers"]:
            comps = getattr(self.network, branch_component + '_df')
            new_comp = comps.loc[comps_dict[branch_component]]
            new_comp.name = 'new_branch_component'
            setattr(new_comp, 'bus'+str(i),'Non_existant_bus_' + branch_component)
            setattr(self.network, branch_component + '_df',
                    comps.append(new_comp))
            try:
                import_data._validate_ding0_grid_import(self.network)
                raise Exception('Appending components {} did not work properly.'.format(branch_component))
            except ValueError as e:
                assert e.args[0] == 'The following {} have bus{} which are ' \
                                    'not defined: {}'.format(
                    branch_component, i, new_comp.name)
            setattr(self.network, branch_component + '_df', comps)
            import_data._validate_ding0_grid_import(self.network)
            i=+1
        print('Check branch elements finished.')

        # check switches
        comps = self.network.switches_df
        for attr in ["bus_open", "bus_closed"]:
            new_comp = comps.loc[comps_dict['switches']]
            new_comp.name = 'new_switch'
            new_comps = comps.append(new_comp)
            new_comps.at[new_comp.name,attr] = 'Non_existant_'+attr
            self.network.switches_df = new_comps
            try:
                import_data._validate_ding0_grid_import(self.network)
                raise Exception('Appending components switches did not work properly.')
            except ValueError as e:
                assert e.args[0] == 'The following switches have {} which are ' \
                                    'not defined: {}'.format(
                    attr, new_comp.name)
            self.network.switches_df = comps
            import_data._validate_ding0_grid_import(self.network)
        print('Check switches finished')

        # check isolated node
        bus = self.network.buses_df.loc[comps_dict['buses']]
        bus.name = 'New_bus'
        self.network.buses_df = self.network.buses_df.append(bus)
        try:
            import_data._validate_ding0_grid_import(self.network)
            raise Exception('Appending components buses did not work properly.')
        except ValueError as e:
            assert e.args[0] == 'The following buses are isolated nodes: ' \
                                '{}'.format(bus.name)
        print('Check isolated nodes finished.')

        print('OK')



