import pandas as pd
import pyproj
from functools import partial
import os
if not 'READTHEDOCS' in os.environ:
    from shapely.ops import transform

from edisgo.grid.components import Transformer, Line, LVStation
from edisgo.grid.grids import LVGrid, MVGrid
from edisgo.grid.tools import get_mv_feeder_from_line


def grid_expansion_costs(network, without_generator_import=False):
    """
    Calculates grid expansion costs for each reinforced transformer and line
    in kEUR.

    Attributes
    ----------
    network : :class:`~.grid.network.Network`
    without_generator_import : Boolean
        If True excludes lines that were added in the generator import to
        connect new generators to the grid from calculation of grid expansion
        costs. Default: False.

    Returns
    -------
    `pandas.DataFrame<dataframe>`
        DataFrame containing type and costs plus in the case of lines the
        line length and number of parallel lines of each reinforced
        transformer and line. Index of the DataFrame is the respective object
        that can either be a :class:`~.grid.components.Line` or a
        :class:`~.grid.components.Transformer`. Columns are the following:

        type: String
            Transformer size or cable name

        total_costs: float
            Costs of equipment in kEUR. For lines the line length and number of
            parallel lines is already included in the total costs.

        quantity: int
            For transformers quantity is always one, for lines it specifies the
            number of parallel lines.

        line_length: float
            Length of line or in case of parallel lines all lines in km.

        voltage_level : :obj:`str` {'lv' | 'mv' | 'mv/lv'}
            Specifies voltage level the equipment is in.

        mv_feeder : :class:`~.grid.components.Line`
            First line segment of half-ring used to identify in which
            feeder the grid expansion was conducted in.

    Notes
    -------
    Total grid expansion costs can be obtained through
    self.grid_expansion_costs.total_costs.sum().

    """
    def _get_transformer_costs(transformer):
        if isinstance(transformer.grid, LVGrid):
            return float(network.config['costs_transformers']['lv'])
        elif isinstance(transformer.grid, MVGrid):
            return float(network.config['costs_transformers']['mv'])

    def _get_line_costs(line, quantity):
        # get voltage level
        if isinstance(line.grid, LVGrid):
            voltage_level = 'lv'
        elif isinstance(line.grid, MVGrid):
            voltage_level = 'mv'
        else:
            raise KeyError("Grid must be LVGrid or MVGrid.")
        # get population density in people/km^2
        # transform area to calculate area in km^2
        projection = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:{}'.format(
                int(network.config['geo']['srid']))),
            pyproj.Proj(init='epsg:3035'))
        sqm2sqkm = 1e6
        population_density = (line.grid.grid_district['population'] /
                              (transform(projection,
                               line.grid.grid_district['geom']).area /
                               sqm2sqkm))
        if population_density <= 500:
            population_density = 'rural'
        else:
            population_density = 'urban'
        # get costs from config
        costs_cable = float(network.config['costs_cables']['{}_cable'.format(
            voltage_level)])
        costs_cable_earthwork = float(network.config['costs_cables'][
                                          '{}_cable_incl_earthwork_{}'.format(
                                              voltage_level,
                                              population_density)])
        return (costs_cable_earthwork * l.length +
                costs_cable * l.length * (quantity - 1))

    costs = pd.DataFrame()

    if without_generator_import:
        equipment_changes = network.results.equipment_changes.loc[
            network.results.equipment_changes.iteration_step > 0]
    else:
        equipment_changes = network.results.equipment_changes

    # costs for transformers
    if not equipment_changes.empty:
        transformers = equipment_changes[equipment_changes['equipment'].apply(
            isinstance, args=(Transformer,))]
        added_transformers = transformers[transformers['change'] == 'added']
        removed_transformers = transformers[
            transformers['change'] == 'removed']
        # check if any of the added transformers were later removed
        added_removed_transformers = added_transformers.loc[
            added_transformers['equipment'].isin(
                removed_transformers['equipment'])]
        added_transformers = added_transformers[
            ~added_transformers['equipment'].isin(
                added_removed_transformers.equipment)]
        # calculate costs for each transformer
        for t in added_transformers['equipment']:
            costs = costs.append(pd.DataFrame(
                {'type': t.type.name,
                 'total_costs': _get_transformer_costs(t),
                 'quantity': 1,
                 'voltage_level': 'mv/lv',
                 'mv_feeder': t.grid.station.mv_feeder if isinstance(
                     t.grid, LVGrid) else None},
                index=[t]))

        # costs for lines
        # get changed lines
        lines = equipment_changes.loc[equipment_changes.index[
            equipment_changes.reset_index()['index'].apply(
                isinstance, args=(Line,))]]
        # calculate costs for each reinforced line
        for l in list(lines.index.unique()):
            # check if line connects aggregated units
            aggr_lines = []
            aggr_lines_generator = l.grid.graph.lines_by_attribute('line_aggr')
            for aggr_line in aggr_lines_generator:
                aggr_lines.append(repr(aggr_line['line']))
            if not repr(l) in aggr_lines:
                number_lines_added = equipment_changes[
                    (equipment_changes.index == l) &
                    (equipment_changes.equipment ==
                     l.type.name)]['quantity'].sum()
                costs = costs.append(pd.DataFrame(
                    {'type': l.type.name,
                     'total_costs': _get_line_costs(l, number_lines_added),
                     'length': l.length * number_lines_added,
                     'quantity': number_lines_added,
                     'voltage_level': ('lv' if isinstance(l.grid, LVGrid)
                                       else 'mv'),
                     'mv_feeder': get_mv_feeder_from_line(l)},
                    index=[l]))

    # if no costs incurred write zero costs to DataFrame
    if costs.empty:
        costs = costs.append(pd.DataFrame(
            {'type': ['N/A'],
             'total_costs': [0],
             'length': [0],
             'quantity': [0],
             'voltage_level': '',
             'mv_feeder': ''
             },
            index=['No reinforced equipment.']))

    return costs
