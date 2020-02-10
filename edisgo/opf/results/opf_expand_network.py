import numpy as np
import pandas as pd


def expand_network(edisgo, tolerance=1e-6):
    """
    Apply network expansion factors that were obtained by optimization
    to eDisGo MVGrid

    Parameters
    ----------
    edisgo : :class:`~.edisgo.EDisGo`
    tolerance : float
        The acceptable margin with which an expansion factor can deviate
        from the nearest Integer before it gets rounded up
    """
    if edisgo.opf_results is None:
        raise ValueError("OPF results not found. Run optimization first.")

    nep_factor = edisgo.opf_results.lines.nep.values.astype('float')

    # Only round up numbers that are reasonably far away from the nearest
    # Integer
    nep_factor = np.ceil(nep_factor - tolerance)

    # Get the names of all MV grid lines
    mv_lines = edisgo.topology.mv_grid.lines_df.index

    # Increase number of parallel lines, shrink respective resistance
    edisgo.topology.lines_df.loc[mv_lines, 'num_parallel'] *= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, 'r'] /= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, 'x'] /= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, 's_nom'] *= nep_factor


def grid_expansion_costs(opf_results, tolerance=1e-6):
    """
    Calculates grid expansion costs from OPF.

    As grid expansion is conducted continuously number of expanded lines is
    determined by simply rounding up (including some tolerance).

    Parameters
    ---------
    opf_results : OPFResults class
    tolerance : float

    Returns
    --------
    float
        Grid expansion costs determined by OPF

    """
    lines = opf_results.lines.index

    num_new_lines = np.ceil(
        opf_results.lines.nep *
        opf_results.pypsa.lines.loc[lines, 'num_parallel'] -
        tolerance) - opf_results.pypsa.lines.loc[lines, 'num_parallel']
    costs_cable = opf_results.pypsa.lines.loc[lines, 'costs_cable'] * \
                  num_new_lines

    earthworks = [1 if num_new_lines[l] > 0 else 0 for l in lines]
    costs_earthwork = \
        opf_results.pypsa.lines.loc[lines, 'costs_earthworks'] * earthworks

    total_costs = costs_cable + costs_earthwork
    extended_lines = total_costs[total_costs > 0].index
    costs_df = pd.DataFrame(
        data={'total_costs': total_costs.loc[extended_lines],
              'type': ['line'] * len(extended_lines),
              'length': opf_results.pypsa.lines.loc[extended_lines, 'length'],
              'quantity': num_new_lines.loc[extended_lines],
              'voltage_level': ['mv'] * len(extended_lines)},
        index=extended_lines)

    return costs_df
