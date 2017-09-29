import sys
import pandas as pd
import pyproj
from functools import partial
from shapely.ops import transform

from edisgo.grid.components import Transformer, Line
from edisgo.grid.grids import LVGrid, MVGrid


def grid_expansion_costs(network):
    """
    Calculates grid expansion costs for each reinforced transformer and line
    in kEUR.

    Attributes
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    `pandas.DataFrame<dataframe>`
        DataFrame containing type and costs plus in the case of lines the
        line length and number of parallel lines of each reinforced
        transformer and line. The DataFrame has the following columns:

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

    def _get_line_costs(line):
        # get voltage level
        if isinstance(line.grid, LVGrid):
            voltage_level = 'lv'
        elif isinstance(line.grid, MVGrid):
            voltage_level = 'mv'
        else:
            print("Voltage level for line must be lv or mv.")
            sys.exit()
        # get population density in people/km^2
        # transform area to calculate area in km^2
        projection = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:{}'.format(network.config['geo']['srid'])),
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
        return (float(network.config['costs_cables']['{} {}'.format(
            voltage_level, population_density)]))

    costs = pd.DataFrame()

    # costs for transformers
    transformers = network.results.equipment_changes[
        network.results.equipment_changes['equipment'].apply(
            isinstance, args=(Transformer,))]
    added_transformers = transformers[transformers['change'] == 'added']
    removed_transformers = transformers[transformers['change'] == 'removed']
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
             'quantity': 1},
            index=[repr(t)]))

    # costs for lines
    # get changed lines
    lines = network.results.equipment_changes.loc[
        network.results.equipment_changes.index[
            network.results.equipment_changes.reset_index()['index'].apply(
                isinstance, args=(Line,))]]
    # calculate costs for each reinforced line
    for l in list(lines.index.unique()):
        costs = costs.append(pd.DataFrame(
            {'type': l.type.name,
             'total_costs': _get_line_costs(l) * l.length * l.quantity,
             'length': l.length * l.quantity,
             'quantity': l.quantity},
            index=[repr(l)]))

    return costs
