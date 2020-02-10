import numpy as np


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
