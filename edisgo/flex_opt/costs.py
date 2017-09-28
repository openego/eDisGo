import sys
import pandas as pd

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
        Dataframe containing type and costs plus in the case of lines the
        line length and number of parallel lines of each reinforced
        transformer and line. The DataFrame has the following columns:

        type: String
            Transformer size or cable name

        total_costs: float
            Costs of equipment in kEUR. For lines the line length and number of
            parallel lines is already included in the total costs.

        quantity: int
            Number of parallel lines.

        line_length: float
            Length of one line in km.

    Notes
    -------
    Total grid expansion costs can be obtained through
    self.grid_expansion_costs.total_costs.sum().

    """
    def _get_transformer_costs(transformer):
        try:
            # try to get costs for transformer
            cost = float(
                network.config['lv_transformers'][transformer.type.name])
        except:
            # ToDo How to deal with those?
            try:
                # try to get costs for transformer with same nominal power
                cost = float(network.config['costs_lv_transformers'][
                    str(int(transformer.type.S_nom)) + ' kVA'])
            except:
                # use costs of standard transformer
                cost = float(
                    network.config['costs_lv_transformers'][
                        network.config['grid_expansion'][
                            'std_mv_lv_transformer']])
        return cost

    def _get_line_costs(line):
        # get voltage level
        if isinstance(line.grid, LVGrid):
            voltage_level = 'lv'
        elif isinstance(line.grid, MVGrid):
            voltage_level = 'mv'
        else:
            print("Voltage level for line must be lv or mv.")
            sys.exit()

        try:
            # try to get costs for line
            cost = float(
                network.config['costs_{}_cables'.format(voltage_level)][
                    line.type.name])
        except:
            # ToDo How to deal with those?
            # use costs of standard line
            cost = float(
                    network.config['costs_{}_cables'.format(voltage_level)][
                        network.config['grid_expansion'][
                            'std_{}_line'.format(voltage_level)]])
        return cost

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
    for transformer in added_transformers['equipment']:
        costs = costs.append(pd.DataFrame(
            {'type': transformer.type.name,
             'total_costs': _get_transformer_costs(transformer)},
            index=[repr(transformer)]))

    # costs for lines
    # get changed lines
    lines = network.results.equipment_changes.loc[
        network.results.equipment_changes.index[
            network.results.equipment_changes.reset_index()['index'].apply(
                isinstance, args=(Line,))]]
    # calculate costs for each reinforced line
    # ToDo: include costs for groundwork
    for line in list(lines.index.unique()):
        costs = costs.append(pd.DataFrame(
            {'type': line.type.name,
             'total_costs': (_get_line_costs(line) * line.length *
                             line.quantity),
             'length': line.length,
             'quantity': line.quantity},
             index=[repr(line)]))

    return costs