import numpy as np
import pandas as pd
from edisgo.network.timeseries import add_storage_units_timeseries
import logging

logger = logging.getLogger("edisgo")


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

    nep_factor = edisgo.opf_results.lines.nep.values.astype("float")

    # Only round up numbers that are reasonably far away from the nearest
    # Integer
    # ToDo: fix! if there was more than 1 line before the optimization this ceil
    # will overestimate the number of added lines (np.ceil(nep_factor*lines.num_parallel - tolerance))
    # this will give number of added lines
    nep_factor = np.ceil(nep_factor - tolerance)

    # Get the names of all MV grid lines
    mv_lines = edisgo.topology.mv_grid.lines_df.index

    # Increase number of parallel lines, shrink respective resistance
    # ToDo: use function Topology.update_number_of_parallel_lines
    edisgo.topology.lines_df.loc[mv_lines, "num_parallel"] *= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, "r"] /= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, "x"] /= nep_factor
    edisgo.topology.lines_df.loc[mv_lines, "s_nom"] *= nep_factor


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
    # ToDo maybe choose differenct default tolerance
    lines = opf_results.lines.index

    num_new_lines = (
        np.ceil(
            opf_results.lines.nep
            * opf_results.pypsa.lines.loc[lines, "num_parallel"]
            - tolerance
        )
        - opf_results.pypsa.lines.loc[lines, "num_parallel"]
    )
    costs_cable = (
        opf_results.pypsa.lines.loc[lines, "costs_cable"] * num_new_lines
    )

    earthworks = [1 if num_new_lines[l] > 0 else 0 for l in lines]
    costs_earthwork = (
        opf_results.pypsa.lines.loc[lines, "costs_earthworks"] * earthworks
    )

    total_costs = costs_cable + costs_earthwork
    extended_lines = total_costs[total_costs > 0].index
    costs_df = pd.DataFrame(
        data={
            "total_costs": total_costs.loc[extended_lines],
            "type": ["line"] * len(extended_lines),
            "length": opf_results.pypsa.lines.loc[extended_lines, "length"],
            "quantity": num_new_lines.loc[extended_lines],
            "voltage_level": ["mv"] * len(extended_lines),
        },
        index=extended_lines,
    )

    return costs_df


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
    # ToDo maybe choose differenct default tolerance
    lines = opf_results.lines.index

    num_new_lines = (
        np.ceil(
            opf_results.lines.nep
            * opf_results.pypsa.lines.loc[lines, "num_parallel"]
            - tolerance
        )
        - opf_results.pypsa.lines.loc[lines, "num_parallel"]
    )
    costs_cable = (
        opf_results.pypsa.lines.loc[lines, "costs_cable"] * num_new_lines
    )

    earthworks = [1 if num_new_lines[l] > 0 else 0 for l in lines]
    costs_earthwork = (
        opf_results.pypsa.lines.loc[lines, "costs_earthworks"] * earthworks
    )

    total_costs = costs_cable + costs_earthwork
    extended_lines = total_costs[total_costs > 0].index
    costs_df = pd.DataFrame(
        data={
            "total_costs": total_costs.loc[extended_lines],
            "type": ["line"] * len(extended_lines),
            "length": opf_results.pypsa.lines.loc[extended_lines, "length"],
            "quantity": num_new_lines.loc[extended_lines],
            "voltage_level": ["mv"] * len(extended_lines),
        },
        index=extended_lines,
    )

    return costs_df


def integrate_storage_units(
    edisgo, min_storage_size=0.3, timeseries=True, as_load=False
):
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
        storage_ts = (
            edisgo.opf_results.storage_units_t.ud
            - edisgo.opf_results.storage_units_t.uc
        ).apply(pd.to_numeric)
        reactive_power_ts = pd.DataFrame(
            0.0, columns=storage_ts.columns, index=storage_ts.index
        )

    # ToDo adding timeseries will only work if timeseries.mode is None
    # ToDo @Anya why is for mode manual kwarg called 'storage_units_reactive_power'
    # and for mode None kwarg called 'timeseries_storage_units'
    for st in edisgo.opf_results.storage_units.index:
        storage_cap = edisgo.opf_results.storage_units.at[st, "emax"]
        if (
            storage_cap >= min_storage_size
            and (storage_ts.loc[:, st] > 0.001).any()
        ):
            if not as_load:
                storage = edisgo.topology.add_storage_unit(
                    bus=st, p_nom=storage_cap
                )  # as C-rate is currently always 1
            else:
                storage = edisgo.topology.add_load(
                    load_id=1,
                    bus=st,
                    peak_load=storage_cap,
                    annual_consumption=0.0,
                    sector="storage",
                )
            if timeseries:
                ts_active = storage_ts.loc[:, [st]].rename(
                    columns={st: storage}
                )
                ts_reactive = reactive_power_ts.loc[:, [st]].rename(
                    columns={st: storage}
                )
                if not as_load:
                    add_storage_units_timeseries(
                        edisgo_obj=edisgo,
                        storage_unit_names=storage,
                        timeseries_storage_units=ts_active,
                        timeseries_storage_units_reactive_power=ts_reactive,
                    )
                else:
                    # ToDo change once fixed in timeseries
                    edisgo.timeseries.loads_active_power = pd.concat(
                        [edisgo.timeseries.loads_active_power, -ts_active],
                        axis=1, sort=False
                    )
                    edisgo.timeseries.loads_reactive_power = pd.concat(
                        [edisgo.timeseries.loads_reactive_power, ts_reactive],
                        axis=1, sort=False
                    )

            added_storage_units.append(storage)
        else:
            logger.info(
                "Storage size of storage unit at bus {} is too small and "
                "therefore discarded.".format(st)
            )
            storage_cap_discarded += storage_cap
    return added_storage_units, storage_cap_discarded


def get_curtailment_per_node(edisgo, curtailment_ts=None, tolerance=1e-3):
    """
    Gets curtailed power per node.

    As LV generators are aggregated at the corresponding LV station curtailment
    is not determined per generator but per node.

    This function also checks if curtailment requirements were met by OPF in
    case the curtailment requirement time series is provided.

    Parameters
    -----------
    edisgo : EDisGo object
    curtailment_ts : pd.Series
        Series with curtailment requirement per time step. Only needs to be
        provided if you want to check if requirement was met.
    tolerance : float
        Tolerance for checking if curtailment requirement and curtailed
        power are equal.

    Returns
    -------
    pd.DataFrame
        DataFrame with curtailed power in MW per node. Column names correspond
        to nodes and index to time steps calculated.

    """
    slack = edisgo.opf_results.pypsa.generators[
        edisgo.opf_results.pypsa.generators.control == "Slack"
    ].index[0]
    # feed-in with curtailment
    opf_gen_results = edisgo.opf_results.generators_t.pg.loc[
        :, edisgo.opf_results.generators_t.pg.columns != slack
    ]
    # feed-in without curtailment
    pypsa_gen_ts = edisgo.opf_results.pypsa.generators_t.p_set.loc[
        :, edisgo.opf_results.pypsa.generators_t.p_set.columns != slack
    ]

    diff = pypsa_gen_ts.loc[:, opf_gen_results.columns] - opf_gen_results
    # set very small differences to zero
    tol = 1e-3
    diff[abs(diff) < tol] = 0
    # check
    if diff[diff < 0].any().any():
        raise ValueError("Generator feed-in higher than allowed feed-in.")
    # drop columns with no curtailment
    diff = diff[diff > 0].dropna(axis=1, how="all").fillna(0)
    # group by node
    tmp = diff.T.copy()
    tmp.index = [
        edisgo.opf_results.pypsa.generators.at[g, "bus"]
        for g in diff.columns
    ]
    curtailment_per_node = (tmp.groupby(tmp.index).sum()).T

    if curtailment_ts is not None:
        if (
            abs(curtailment_ts - curtailment_per_node.sum(axis=1)) > tolerance
        ).any():
            logger.warning("Curtailment requirement not met through OPF.")
    return curtailment_per_node


def get_load_curtailment_per_node(edisgo, tolerance=1e-3):
    """
    Gets curtailed load per node.

    Parameters
    -----------
    edisgo : EDisGo object
    tolerance : float
        Tolerance for checking if curtailment requirement and curtailed
        power are equal.

    Returns
    -------
    pd.DataFrame
        DataFrame with curtailed power in MW per node. Column names correspond
        to nodes and index to time steps calculated.

    """
    load_agg_at_bus = pd.DataFrame(
        columns=edisgo.opf_results.pypsa.loads.bus.unique(),
        index=edisgo.opf_results.pypsa.snapshots)
    for b in edisgo.opf_results.pypsa.loads.bus.unique():
        loads = edisgo.opf_results.pypsa.loads[
            edisgo.opf_results.pypsa.loads.bus == b].index
        load_agg_at_bus.loc[:, b] = edisgo.opf_results.pypsa.loads_t.p_set.loc[
                                    :, loads].sum(axis=1)

    diff = load_agg_at_bus - edisgo.opf_results.loads_t.pd
    # set very small differences to zero
    diff[abs(diff) < tolerance] = 0
    # check
    if diff[diff < 0].any().any():
        raise ValueError("Dispatched load higher than given load.")
    # drop columns with no curtailment
    diff = diff[diff > 0].dropna(axis=1, how="all").fillna(0)

    return diff


def integrate_curtailment_as_load(edisgo, curtailment_per_node):
    """
    Adds load curtailed power per node as load

    This is done because curtailment results from OPF are not given per
    generator but per node (as LV generators are aggregated per LV grid).

    :param edisgo:
    :param curtailment_per_node:
    :return:
    """
    active_power_ts = pd.DataFrame(
        data=0,
        columns=curtailment_per_node.columns,
        index=edisgo.timeseries.timeindex)
    active_power_ts.loc[curtailment_per_node.index,
        :] = curtailment_per_node.apply(pd.to_numeric)
    # drop all zeros
    active_power_ts = active_power_ts.loc[:, ~(active_power_ts == 0.0).all()]
    reactive_power_ts = pd.DataFrame(
        data=0,
        columns=active_power_ts.columns,
        index=edisgo.timeseries.timeindex
    )

    curtailment_loads = edisgo.topology.loads_df[
        edisgo.topology.loads_df.sector == "curtailment"]

    for n in active_power_ts.columns:

        if not n in curtailment_loads.bus:
            # add load component
            load = edisgo.topology.add_load(
                load_id=1,
                bus=n,
                peak_load=curtailment_per_node.loc[:, n].max(),
                annual_consumption=0.0,
                sector="curtailment",
            )

            # add time series
            ts_active = active_power_ts.loc[:, [n]].rename(columns={n: load})
            ts_reactive = reactive_power_ts.loc[:, [n]].rename(
                columns={n: load})
            edisgo.timeseries.loads_active_power = pd.concat(
                [edisgo.timeseries.loads_active_power, ts_active],
                axis=1, sort=False
            )
            edisgo.timeseries.loads_reactive_power = pd.concat(
                [edisgo.timeseries.loads_reactive_power, ts_reactive],
                axis=1, sort=False
            )
        else:
            # add to existing load
            load = curtailment_loads[curtailment_loads.bus == n].index
            edisgo.timeseries._loads_active_power.loc[:, load] = \
                edisgo.timeseries._loads_active_power.loc[:,
                load] + active_power_ts.loc[:, n].rename(columns={n: load})
