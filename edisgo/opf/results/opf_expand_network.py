import numpy as np
import pandas as pd
from edisgo.network.timeseries import add_storage_units_timeseries
import logging

logger = logging.getLogger('edisgo')


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
    #ToDo: fix! if there was more than 1 line before the optimization this ceil
    #will overestimate the number of added lines (np.ceil(nep_factor*lines.num_parallel - tolerance))
    #this will give number of added lines
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


def integrate_storage_units(edisgo, min_storage_size=0.3, timeseries=True,
                            as_load=False):
    """
    Integrates storage units from OPF into edisgo grid topology.

    Storage units that are too small to be connected to the MV grid or that
    are not used (time series contains only zeros) are discarded.

    Parameters
    ----------
    edisgo : EDisGo object
    min_storage_size : float
        Minimal storage size in MW needed to connect storage unit to MV grid.
        Smaller storage units are ignored.
    timeseries : bool
        If True time series is added to component.
    as_load : bool
        If True, storage is added as load to the edisgo topology. This
        is temporarily needed as the OPF cannot handle storage units from
        edisgo yet. This way, storage units with fixed position and time
        series can be considered in OPF.

    Returns
    -------
    list(str), float
        First return value contains the names of the added storage units and
        the second return value the capacity of storage units that were too
        small to connect to the MV grid or not used.

    """
    storage_cap_discarded = 0
    added_storage_units = []

    if timeseries:
        storage_ts = (edisgo.opf_results.storage_units_t.ud -
                      edisgo.opf_results.storage_units_t.uc).apply(
            pd.to_numeric)
        reactive_power_ts = pd.DataFrame(0.0,
                                         columns=storage_ts.columns,
                                         index=storage_ts.index)

    # ToDo adding timeseries will only work if timeseries.mode is None
    # ToDo @Anya why is for mode manual kwarg called 'storage_units_reactive_power'
    # and for mode None kwarg called 'timeseries_storage_units'
    for st in edisgo.opf_results.storage_units.index:
        storage_cap = edisgo.opf_results.storage_units.at[st, 'emax']
        if storage_cap >= min_storage_size and \
                (storage_ts.loc[:, st] > 0.001).any():
            if not as_load:
                storage = edisgo.topology.add_storage_unit(
                    bus=st,
                    p_nom=storage_cap)  # as C-rate is currently always 1
            else:
                storage = edisgo.topology.add_load(
                    load_id=1,
                    bus=st,
                    peak_load=storage_cap,
                    annual_consumption=0.0,
                    sector='storage')
            if timeseries:
                ts_active = storage_ts.loc[:, [st]].rename(
                    columns={st: storage})
                ts_reactive = reactive_power_ts.loc[:, [st]].rename(
                    columns={st: storage})
                if not as_load:
                    add_storage_units_timeseries(
                        edisgo_obj=edisgo,
                        storage_unit_names=storage,
                        timeseries_storage_units=ts_active,
                        timeseries_storage_units_reactive_power=ts_reactive)
                else:
                    #ToDo change once fixed in timeseries
                    edisgo.timeseries.loads_active_power = \
                        pd.concat(
                            [edisgo.timeseries.loads_active_power,
                             -ts_active], axis=1)
                    edisgo.timeseries.generators_reactive_power = \
                        pd.concat(
                            [edisgo.timeseries.generators_reactive_power,
                             ts_reactive], axis=1)

            added_storage_units.append(storage)
        else:
            logger.info(
                "Storage size of storage unit at bus {} is too small and "
                "therefore discarded.".format(st))
            storage_cap_discarded += storage_cap
    return added_storage_units, storage_cap_discarded


