"""
This module provides tools to convert eDisGo representation of the network
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA network
container.
"""

import collections
import logging

from math import sqrt

import numpy as np
import pandas as pd

from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe

logger = logging.getLogger(__name__)


def to_pypsa(edisgo_object, mode=None, timesteps=None, **kwargs):
    """
    Convert grid to :pypsa:`PyPSA.Network<network>` representation.

    You can choose between translation of the MV and all underlying LV grids
    (mode=None (default)), the MV network only (mode='mv' or mode='mvlv') or a
    single LV network (mode='lv').

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
        EDisGo object containing grid topology and time series information.
    mode : str
        Determines network levels that are translated to
        :pypsa:`PyPSA.Network<network>`.
        See `mode` parameter in :attr:`~.edisgo.EDisGo.to_pypsa` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        See `timesteps` parameter in :attr:`~.edisgo.EDisGo.to_pypsa` for more
        information.

    Other Parameters
    -----------------
    See other parameters in :attr:`~.edisgo.EDisGo.to_pypsa` for more
    information.

    Returns
    -------
    :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation.

    """

    def _set_slack(grid):
        """
        Sets slack at given grid's station secondary side.

        It is assumed that the secondary side bus is always given in
        transformer's bus1.

        Parameters
        -----------
        grid : :class:`~.network.grids.Grid`
            Low or medium voltage grid to position slack in.

        Returns
        -------
        """
        slack_bus = grid.transformers_df.bus1.iloc[0]
        return pd.DataFrame(
            data={"bus": slack_bus, "control": "Slack"},
            index=["Generator_slack"],
        )

    aggregate_loads = kwargs.get("aggregate_loads", None)
    aggregate_generators = kwargs.get("aggregate_generators", None)
    aggregate_storages = kwargs.get("aggregate_storages", None)
    aggregated_lv_components = {"Generator": {}, "Load": {}, "StorageUnit": {}}

    if timesteps is None:
        timesteps = edisgo_object.timeseries.timeindex
    # check if timesteps is array-like, otherwise convert to list (necessary
    # to obtain a dataframe when using .loc in time series functions)
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.set_snapshots(timesteps)

    # define buses_df, slack_df and components for each use case
    if mode is None:
        pypsa_network.mode = "mv"

        buses_df = edisgo_object.topology.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(edisgo_object.topology.mv_grid)

        components = {
            "Load": edisgo_object.topology.loads_df.loc[:, ["bus", "p_set"]],
            "Generator": edisgo_object.topology.generators_df.loc[
                :, ["bus", "control", "p_nom"]
            ],
            "StorageUnit": edisgo_object.topology.storage_units_df.loc[
                :, ["bus", "control", "p_nom", "max_hours"]
            ],
            "Line": edisgo_object.topology.lines_df.loc[
                :,
                ["bus0", "bus1", "x", "r", "b", "s_nom", "num_parallel", "length"],
            ],
            "Transformer": edisgo_object.topology.transformers_df.loc[
                :, ["bus0", "bus1", "x_pu", "r_pu", "type_info", "s_nom"]
            ].rename(columns={"r_pu": "r", "x_pu": "x"}),
        }

    elif "mv" in mode:
        pypsa_network.mode = "mv"

        grid_object = edisgo_object.topology.mv_grid
        buses_df = grid_object.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(grid_object)

        # MV components
        mv_components = _get_grid_component_dict(grid_object)
        mv_components["Generator"]["fluctuating"] = grid_object.generators_df.type.isin(
            ["solar", "wind"]
        )

        if mode == "mv":
            mv_components["Transformer"] = pd.DataFrame(dtype=float)
        elif mode == "mvlv":
            # get all MV/LV transformers
            mv_components["Transformer"] = edisgo_object.topology.transformers_df.loc[
                :, ["bus0", "bus1", "x_pu", "r_pu", "type_info", "s_nom"]
            ].rename(columns={"r_pu": "r", "x_pu": "x"})
        else:
            raise ValueError("Provide proper mode for mv network export.")

        # LV components
        lv_components_to_aggregate = {
            "Load": ["loads_df"],
            "Generator": ["generators_df"],
            "StorageUnit": ["storage_units_df"],
        }
        lv_components = {
            key: pd.DataFrame(dtype=float) for key in lv_components_to_aggregate
        }

        for lv_grid in grid_object.lv_grids:
            if mode == "mv":
                # get primary side of station to append loads and generators to
                station_bus = grid_object.buses_df.loc[
                    lv_grid.transformers_df.bus0.unique()
                ]
            elif mode == "mvlv":
                # get secondary side of station to append loads and generators to
                station_bus = lv_grid.buses_df.loc[
                    [lv_grid.transformers_df.bus1.unique()[0]]
                ]
                buses_df = pd.concat([buses_df, station_bus.loc[:, ["v_nom"]]])
            # handle one gate components
            for comp, dfs in lv_components_to_aggregate.items():
                comps = pd.DataFrame(dtype=float)
                for df in dfs:
                    comps_tmp = getattr(lv_grid, df).copy()
                    comps = pd.concat([comps, comps_tmp])

                comps.bus = station_bus.index.values[0]
                aggregated_lv_components[comp].update(
                    _append_lv_components(
                        comp,
                        comps,
                        lv_components,
                        repr(lv_grid),
                        aggregate_loads=aggregate_loads,
                        aggregate_generators=aggregate_generators,
                        aggregate_storages=aggregate_storages,
                    )
                )

        # merge components
        components = collections.defaultdict(pd.DataFrame)
        for comps in (mv_components, lv_components):
            for key, value in comps.items():
                components[key] = pd.concat(
                    [
                        components[key],
                        value,
                    ]
                )

    elif mode == "lv":
        pypsa_network.mode = "lv"

        lv_grid_id = kwargs.get("lv_grid_id", None)
        if lv_grid_id is None:
            raise ValueError(
                "For exporting LV grids, ID or name of LV grid has to be provided "
                "using parameter `lv_grid_id`."
            )
        grid_object = edisgo_object.topology.get_lv_grid(lv_grid_id)
        buses_df = grid_object.buses_df.loc[:, ["v_nom"]]
        slack_df = _set_slack(grid_object)

        components = _get_grid_component_dict(grid_object)
    else:
        raise ValueError(
            "Provide proper mode or leave it empty to export entire network topology."
        )

    # import network topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(buses_df, "Bus")
    pypsa_network.import_components_from_dataframe(slack_df, "Generator")
    for k, comps in components.items():
        pypsa_network.import_components_from_dataframe(comps, k)

    # import time series to PyPSA network

    # set all voltages except for slack
    import_series_from_dataframe(
        pypsa_network,
        _buses_voltage_set_point(
            edisgo_object,
            buses_df.index,
            slack_df.loc["Generator_slack", "bus"],
            timesteps,
        ),
        "Bus",
        "v_mag_pu_set",
    )

    # set slack time series
    slack_ts = pd.DataFrame(
        data=[0] * len(timesteps),
        columns=[slack_df.index[0]],
        index=timesteps,
    )
    import_series_from_dataframe(pypsa_network, slack_ts, "Generator", "p_set")
    import_series_from_dataframe(pypsa_network, slack_ts, "Generator", "q_set")

    # set generator time series in pypsa
    if len(components["Generator"].index) > 0:
        if len(aggregated_lv_components["Generator"]) > 0:
            (
                generators_timeseries_active,
                generators_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_object,
                timesteps,
                ["generators"],
                components["Generator"].index,
                aggregated_lv_components["Generator"],
            )
        else:
            generators_timeseries_active = (
                edisgo_object.timeseries.generators_active_power.loc[
                    timesteps,
                    edisgo_object.timeseries.generators_active_power.columns.isin(
                        components["Generator"].index
                    ),
                ]
            )
            generators_timeseries_reactive = (
                edisgo_object.timeseries.generators_reactive_power.loc[
                    timesteps,
                    edisgo_object.timeseries.generators_reactive_power.columns.isin(
                        components["Generator"].index
                    ),
                ]
            )

        import_series_from_dataframe(
            pypsa_network, generators_timeseries_active, "Generator", "p_set"
        )
        import_series_from_dataframe(
            pypsa_network, generators_timeseries_reactive, "Generator", "q_set"
        )

    if len(components["Load"].index) > 0:
        if len(aggregated_lv_components["Load"]) > 0:
            (
                loads_timeseries_active,
                loads_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_object,
                timesteps,
                ["loads"],
                components["Load"].index,
                aggregated_lv_components["Load"],
            )
        else:
            loads_timeseries_active = edisgo_object.timeseries.loads_active_power.loc[
                timesteps,
                edisgo_object.timeseries.loads_active_power.columns.isin(
                    components["Load"].index
                ),
            ]
            loads_timeseries_reactive = (
                edisgo_object.timeseries.loads_reactive_power.loc[
                    timesteps,
                    edisgo_object.timeseries.loads_reactive_power.columns.isin(
                        components["Load"].index
                    ),
                ]
            )
        import_series_from_dataframe(
            pypsa_network, loads_timeseries_active, "Load", "p_set"
        )
        import_series_from_dataframe(
            pypsa_network, loads_timeseries_reactive, "Load", "q_set"
        )

    if len(components["StorageUnit"].index) > 0:
        if len(aggregated_lv_components["StorageUnit"]) > 0:
            (
                storages_timeseries_active,
                storages_timeseries_reactive,
            ) = _get_timeseries_with_aggregated_elements(
                edisgo_object,
                timesteps,
                ["storage_units"],
                components["StorageUnit"].index,
                aggregated_lv_components["StorageUnit"],
            )
        else:
            storages_timeseries_active = (
                edisgo_object.timeseries.storage_units_active_power.loc[
                    timesteps,
                    edisgo_object.timeseries.storage_units_active_power.columns.isin(
                        components["StorageUnit"].index
                    ),
                ]
            )
            storages_timeseries_reactive = (
                edisgo_object.timeseries.storage_units_reactive_power.loc[
                    timesteps,
                    edisgo_object.timeseries.storage_units_reactive_power.columns.isin(
                        components["StorageUnit"].index
                    ),
                ]
            )
        import_series_from_dataframe(
            pypsa_network,
            storages_timeseries_active.apply(pd.to_numeric),
            "StorageUnit",
            "p_set",
        )
        import_series_from_dataframe(
            pypsa_network,
            storages_timeseries_reactive.apply(pd.to_numeric),
            "StorageUnit",
            "q_set",
        )

    if kwargs.get("use_seed", False) and pypsa_network.mode == "mv":
        set_seed(edisgo_object, pypsa_network)

    return pypsa_network


def set_seed(edisgo_obj, pypsa_network):
    """
    Set initial guess for the Newton-Raphson algorithm.

    In `PyPSA <https://pypsa.readthedocs.io/en/latest/>`_ an
    initial guess for the Newton-Raphson algorithm used in the power flow
    analysis can be provided to speed up calculations.
    For PQ buses, which besides the slack bus, is the only bus type in
    edisgo, voltage magnitude and angle need to be guessed. If the power
    flow was already conducted for the required time steps and buses, the
    voltage magnitude and angle results from previously conducted power
    flows stored in :attr:`~.network.results.Results.pfa_v_mag_pu_seed` and
    :attr:`~.network.results.Results.pfa_v_ang_seed` are used
    as the initial guess. Always the latest power flow calculation is used
    and only results from power flow analyses including the MV level are
    considered, as analysing single LV grids is currently not in the focus
    of edisgo and does not require as much speeding up, as analysing single
    LV grids is usually already quite quick.
    If for some buses or time steps no power flow results are available,
    default values are used. For the voltage magnitude the default value is 1
    and for the voltage angle 0.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    pypsa_network : :pypsa:`pypsa.Network<network>`
        Pypsa network in which seed is set.

    """

    # get all PQ buses for which seed needs to be set
    pq_buses = pypsa_network.buses[pypsa_network.buses.control == "PQ"].index

    # get voltage magnitude and angle results from previous power flow analyses
    pfa_v_mag_pu_seed = edisgo_obj.results.pfa_v_mag_pu_seed
    pfa_v_ang_seed = edisgo_obj.results.pfa_v_ang_seed

    # get busses seed cannot be set for from previous power flow analyses
    # and add default values for those
    buses_missing = [_ for _ in pq_buses if _ not in pfa_v_mag_pu_seed.columns]
    if len(buses_missing) > 0:
        pfa_v_mag_pu_seed = pd.concat(
            [
                pfa_v_mag_pu_seed,
                pd.DataFrame(
                    data=1.0, columns=buses_missing, index=pfa_v_ang_seed.index
                ),
            ],
            axis=1,
        )
        pfa_v_ang_seed = pd.concat(
            [
                pfa_v_ang_seed,
                pd.DataFrame(
                    data=0.0, columns=buses_missing, index=pfa_v_ang_seed.index
                ),
            ],
            axis=1,
        )
    # select only PQ buses
    pfa_v_mag_pu_seed = pfa_v_mag_pu_seed.loc[:, pq_buses]
    pfa_v_ang_seed = pfa_v_ang_seed.loc[:, pq_buses]

    # get time steps seed cannot be set for from previous power flow analyses
    # and add default values for those
    ts_missing = [
        _ for _ in pypsa_network.snapshots if _ not in pfa_v_mag_pu_seed.index
    ]
    if len(ts_missing) > 0:
        pfa_v_mag_pu_seed = pd.concat(
            [
                pfa_v_mag_pu_seed,
                pd.DataFrame(data=1.0, columns=pq_buses, index=ts_missing),
            ],
            axis=0,
        )
        pfa_v_ang_seed = pd.concat(
            [
                pfa_v_ang_seed,
                pd.DataFrame(data=0.0, columns=pq_buses, index=ts_missing),
            ],
            axis=0,
        )
    # select only snapshots
    pfa_v_mag_pu_seed = pfa_v_mag_pu_seed.loc[pypsa_network.snapshots, :]
    pfa_v_ang_seed = pfa_v_ang_seed.loc[pypsa_network.snapshots, :]

    pypsa_network.buses_t.v_mag_pu = pfa_v_mag_pu_seed
    pypsa_network.buses_t.v_ang = pfa_v_ang_seed


def _get_grid_component_dict(grid_object):
    """
    Method to extract component dictionary from given grid object.

    Components are divided into "Load", "Generator", "StorageUnit" and "Line". Used for
    translation to pypsa network.

    Parameters
    ----------
    grid_object : :class:`~.network.grids.Grid`

    Returns
    -------
    dict
        Component dictionary divided into "Load", "Generator", "StorageUnit"
        and "Line".

    """
    components = {
        "Load": grid_object.loads_df.loc[:, ["bus", "p_set"]],
        "Generator": grid_object.generators_df.loc[:, ["bus", "control", "p_nom"]],
        "StorageUnit": grid_object.storage_units_df.loc[
            :, ["bus", "control", "p_nom", "max_hours"]
        ],
        "Line": grid_object.lines_df.loc[
            :,
            ["bus0", "bus1", "x", "r", "s_nom", "num_parallel", "length"],
        ],
    }
    return components


def _append_lv_components(
    comp,
    comps,
    lv_components,
    lv_grid_name,
    aggregate_loads=None,
    aggregate_generators=None,
    aggregate_storages=None,
):
    """
    Method to append LV components to component dictionary.

    Used when only exporting mv grid topology. All underlying LV components of an
    LVGrid are then connected to one side of the LVStation. If required, the LV
    components can be aggregated in different modes. As an example, loads can be
    aggregated sector-wise or all loads can be aggregated into one
    representative load. The sum of p_nom/p_set of all cumulated components is
    calculated.

    Parameters
    ----------
    comp : str
        Indicator for component type to aggregate. Can be 'Load', 'Generator' or
        'StorageUnit'.
    comps : `pandas.DataFrame<DataFrame>`
        Component dataframe of elements to be aggregated.
    lv_components : dict
        Dictionary of LV grid components, keys are the 'Load', 'Generator' and
        'StorageUnit'.
    lv_grid_name : str
        Representative of LV grid of which components are aggregated.
    aggregate_loads : str
        Mode for load aggregation. Can be 'sectoral' aggregating the loads
        sector-wise, 'all' aggregating all loads into one or None,
        not aggregating loads but appending them to the station one by one.
        Default: None.
    aggregate_generators : str
        Mode for generator aggregation. Can be 'type' resulting in an
        aggregated generator for each generator type, 'curtailable' aggregating
        'solar' and 'wind' generators into one and all other generators into
        another one, or None, where no aggregation is undertaken
        and generators are added one by one. Default: None.
    aggregate_storages : str
        Mode for storage unit aggregation. Can be 'all' where all
        storage units are aggregated to one storage unit or None, in
        which case no aggregation is conducted and storage units are added one by
        one. Default: None.

    Returns
    -------
    dict
        Dictionary of aggregated elements for time series creation. Keys are names
        of aggregated elements and values are a list of the names of all
        components aggregated in that respective key component.
        An example could look as follows:
        {'LVGrid_1_loads':
            ['Load_agricultural_LVGrid_1_1', 'Load_cts_LVGrid_1_2']}

    """
    aggregated_elements = {}
    if len(comps) > 0:
        bus = comps.bus.unique()[0]
    else:
        return {}
    if comp == "Load":
        if aggregate_loads is None:
            comps_aggr = comps.loc[:, ["bus", "p_set"]]
        elif aggregate_loads == "sectoral":
            comps_aggr = (
                comps.loc[:, ["p_set", "sector"]]
                .groupby("sector")
                .sum()
                .loc[:, ["p_set"]]
            )
            for sector in comps_aggr.index.values:
                aggregated_elements[lv_grid_name + "_" + sector] = comps[
                    comps.sector == sector
                ].index.values
            comps_aggr.index = lv_grid_name + "_" + comps_aggr.index
            comps_aggr["bus"] = bus
        elif aggregate_loads == "all":
            comps_aggr = pd.DataFrame(
                {"bus": [bus], "p_set": [sum(comps.p_set)]},
                index=[lv_grid_name + "_loads"],
            )
            aggregated_elements[lv_grid_name + "_loads"] = comps.index.values
        else:
            raise ValueError("Aggregation type for loads invalid.")

        lv_components[comp] = pd.concat(
            [
                lv_components[comp],
                comps_aggr,
            ]
        )

    elif comp == "Generator":
        flucts = ["wind", "solar"]
        if aggregate_generators is None:
            comps_aggr = comps.loc[:, ["bus", "control", "p_nom"]]
            comps_aggr["fluctuating"] = comps.type.isin(flucts)
        elif aggregate_generators == "type":
            comps_aggr = (
                comps.groupby("type").sum().reindex(columns=["bus", "control", "p_nom"])
            )
            comps_aggr.bus = bus
            comps_aggr.control = "PQ"
            comps_aggr["fluctuating"] = comps_aggr.index.isin(flucts)
            for gen_type in comps_aggr.index.values:
                aggregated_elements[lv_grid_name + "_" + gen_type] = comps[
                    comps.type == gen_type
                ].index.values
            comps_aggr.index = lv_grid_name + "_" + comps_aggr.index
        elif aggregate_generators == "curtailable":
            comps_fluct = comps[comps.type.isin(flucts)]
            comps_disp = comps[~comps.index.isin(comps_fluct.index)]
            comps_aggr = pd.DataFrame(columns=["bus", "control", "p_nom"], dtype=float)
            if len(comps_fluct) > 0:
                comps_aggr = pd.concat(
                    [
                        comps_aggr,
                        pd.DataFrame(
                            {
                                "bus": [bus],
                                "control": ["PQ"],
                                "p_nom": [sum(comps_fluct.p_nom)],
                                "fluctuating": [True],
                            },
                            index=[lv_grid_name + "_fluctuating"],
                        ),
                    ]
                )
                aggregated_elements[
                    lv_grid_name + "_fluctuating"
                ] = comps_fluct.index.values

            if len(comps_disp) > 0:
                comps_aggr = pd.concat(
                    [
                        comps_aggr,
                        pd.DataFrame(
                            {
                                "bus": [bus],
                                "control": ["PQ"],
                                "p_nom": [sum(comps_disp.p_nom)],
                                "fluctuating": [False],
                            },
                            index=[lv_grid_name + "_dispatchable"],
                        ),
                    ]
                )
                aggregated_elements[
                    lv_grid_name + "_dispatchable"
                ] = comps_disp.index.values
        elif aggregate_generators == "all":
            comps_aggr = pd.DataFrame(
                {
                    "bus": [bus],
                    "control": ["PQ"],
                    "p_nom": [sum(comps.p_nom)],
                    "fluctuating": [
                        True
                        if (comps.type.isin(flucts)).all()
                        else False
                        if ~comps.type.isin(flucts).any()
                        else "Mixed"
                    ],
                },
                index=[lv_grid_name + "_generators"],
            )
            aggregated_elements[lv_grid_name + "_generators"] = comps.index.values
        else:
            raise ValueError("Aggregation type for generators invalid.")

        lv_components[comp] = pd.concat(
            [
                lv_components[comp],
                comps_aggr,
            ]
        )

    elif comp == "StorageUnit":
        if aggregate_storages is None:
            comps_aggr = comps.loc[:, ["bus", "control"]]
        elif aggregate_storages == "all":
            comps_aggr = pd.DataFrame(
                {"bus": [bus], "control": ["PQ"]},
                index=[lv_grid_name + "_storages"],
            )
            aggregated_elements[lv_grid_name + "_storages"] = comps.index.values
        else:
            raise ValueError("Aggregation type for storages invalid.")

        lv_components[comp] = pd.concat(
            [
                lv_components[comp],
                comps_aggr,
            ]
        )

    else:
        raise ValueError("Component type not defined.")

    return aggregated_elements


def _get_timeseries_with_aggregated_elements(
    edisgo_obj, timesteps, element_types, elements, aggr_dict
):
    """
    Creates time series for aggregated LV components by summing up the single
    time series.

    Parameters
    ----------
    edisgo_obj : :class:`~.self.edisgo.EDisGo`
        eDisGo object
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Time steps to export to pypsa representation.
    element_types : list(str)
        Type of element which was aggregated. Can be 'loads', 'generators' or
        'storage_units'
    elements: `pandas.DataFrame<DataFrame>`
        Component dataframe of all elements for which time series are added.
    aggr_dict: dict
        Dictionary containing aggregated elements as values and the
        representing new component as key. See :meth:`_append_lv_components`
        for structure of dictionary.

    Returns
    -------
    tuple(`pandas.DataFrame<DataFrame>`)
        Active and reactive power time series for chosen elements. Dataframes
        with timesteps as index and name of elements as columns.

    """
    # get relevant timeseries
    elements_timeseries_active_all = pd.DataFrame(dtype=float)
    elements_timeseries_reactive_all = pd.DataFrame(dtype=float)
    for element_type in element_types:
        elements_timeseries_active_all = pd.concat(
            [
                elements_timeseries_active_all,
                getattr(edisgo_obj.timeseries, element_type + "_active_power"),
            ],
            axis=1,
        )
        elements_timeseries_reactive_all = pd.concat(
            [
                elements_timeseries_reactive_all,
                getattr(edisgo_obj.timeseries, element_type + "_reactive_power"),
            ],
            axis=1,
        )
    # handle not aggregated elements
    non_aggregated_elements = elements[~elements.isin(aggr_dict.keys())]
    # get timeseries for non aggregated elements
    elements_timeseries_active = elements_timeseries_active_all.loc[
        timesteps, non_aggregated_elements
    ]
    elements_timeseries_reactive = elements_timeseries_reactive_all.loc[
        timesteps, non_aggregated_elements
    ]
    # append timeseries for aggregated elements
    for aggr_gen in aggr_dict.keys():
        elements_timeseries_active[aggr_gen] = elements_timeseries_active_all.loc[
            timesteps, aggr_dict[aggr_gen]
        ].sum(axis=1)

        elements_timeseries_reactive[aggr_gen] = elements_timeseries_reactive_all.loc[
            timesteps, aggr_dict[aggr_gen]
        ].sum(axis=1)
    return elements_timeseries_active, elements_timeseries_reactive


def _buses_voltage_set_point(edisgo_obj, buses, slack_bus, timesteps):
    """
    Time series in PyPSA compatible format for bus instances

    Set all buses except for the slack bus to voltage of 1 p.u. (it is assumed
    this setting is entirely ignored during solving the power flow problem).
    The slack bus voltage is set based on a given HV/MV transformer offset and
    a control deviation, both defined in the config files. The control
    deviation is added to the offset in the reverse power flow case and
    subtracted from the offset in the heavy load flow case.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        eDisGo object
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<Timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
    buses : list
        Buses names
    slack_bus : str

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Time series table in PyPSA format
    """

    # set all buses to nominal voltage
    v_nom = pd.DataFrame(1, columns=buses, index=timesteps)

    # set slack bus to operational voltage (includes offset and control
    # deviation)
    control_deviation = edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
        "hv_mv_trafo_control_deviation"
    ]
    if control_deviation != 0:
        control_deviation_ts = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: control_deviation if _ == "feedin_case" else -control_deviation
        ).loc[timesteps]
    else:
        control_deviation_ts = pd.Series(0, index=timesteps)

    slack_voltage_pu = (
        control_deviation_ts
        + 1
        + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "hv_mv_trafo_offset"
        ]
    )

    v_nom.loc[timesteps, slack_bus] = slack_voltage_pu

    return v_nom


def process_pfa_results(edisgo, pypsa, timesteps, dtype="float"):
    """
    Passing power flow results from PyPSA to :class:`~.network.results.Results`.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
    pypsa : :pypsa:`pypsa.Network<network>`
        The PyPSA network to retrieve results from.
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        Time steps for which latest power flow analysis was conducted and
        for which to retrieve pypsa results.

    Notes
    -----
    P and Q are returned from the line ending/transformer side with highest
    apparent power S, exemplary written as

    .. math::
        S_{max} = max(\\sqrt{P_0^2 + Q_0^2}, \\sqrt{P_1^2 + Q_1^2}) \\
        P = P_0 P_1(S_{max}) \\
        Q = Q_0 Q_1(S_{max})

    See Also
    --------
    :class:`~.network.results.Results` to understand how results of power flow
    analysis are structured in eDisGo.

    """
    # get the absolute losses in the system (in MW and Mvar)
    # subtracting total generation (including slack) from total load
    grid_losses = {
        "p": (
            abs(
                pypsa.generators_t["p"].sum(axis=1)
                + pypsa.storage_units_t["p"].sum(axis=1)
                - pypsa.loads_t["p"].sum(axis=1)
            )
        ),
        "q": (
            abs(
                pypsa.generators_t["q"].sum(axis=1)
                + pypsa.storage_units_t["q"].sum(axis=1)
                - pypsa.loads_t["q"].sum(axis=1)
            )
        ),
    }
    edisgo.results.grid_losses = pd.DataFrame(grid_losses, dtype=dtype)

    # get slack results in MW and Mvar
    pfa_slack = {
        "p": (pypsa.generators_t["p"]["Generator_slack"]),
        "q": (pypsa.generators_t["q"]["Generator_slack"]),
    }
    edisgo.results.pfa_slack = pd.DataFrame(pfa_slack, dtype=dtype)

    # get P and Q of lines and transformers in MW and Mvar
    q0 = pd.concat(
        [np.abs(pypsa.lines_t["q0"]), np.abs(pypsa.transformers_t["q0"])],
        axis=1,
        sort=False,
    )
    q1 = pd.concat(
        [np.abs(pypsa.lines_t["q1"]), np.abs(pypsa.transformers_t["q1"])],
        axis=1,
        sort=False,
    )
    p0 = pd.concat(
        [np.abs(pypsa.lines_t["p0"]), np.abs(pypsa.transformers_t["p0"])],
        axis=1,
        sort=False,
    )
    p1 = pd.concat(
        [np.abs(pypsa.lines_t["p1"]), np.abs(pypsa.transformers_t["p1"])],
        axis=1,
        sort=False,
    )
    # determine apparent power at line endings/transformer sides
    s0 = np.hypot(p0, q0)
    s1 = np.hypot(p1, q1)
    # choose P and Q from line ending with max(s0,s1)
    edisgo.results.pfa_p = p0.where(s0 > s1, p1).astype(dtype)
    edisgo.results.pfa_q = q0.where(s0 > s1, q1).astype(dtype)

    # calculate line and transformer currents in kA
    lines_bus0 = pypsa.lines["bus0"]
    bus0_v_mag_pu = pypsa.buses_t["v_mag_pu"].loc[:, lines_bus0.values].copy()
    bus0_v_mag_pu.columns = lines_bus0.index
    current_lines = np.hypot(pypsa.lines_t["p0"], pypsa.lines_t["q0"]).truediv(
        pypsa.lines["v_nom"] * bus0_v_mag_pu, axis="columns"
    ) / sqrt(3)
    transformers_bus0 = pypsa.transformers["bus0"]
    bus0_v_mag_pu = pypsa.buses_t["v_mag_pu"].loc[:, transformers_bus0.values].copy()
    bus0_v_mag_abs = pypsa.buses["v_nom"].loc[transformers_bus0.values] * bus0_v_mag_pu
    bus0_v_mag_abs.columns = transformers_bus0.index
    current_transformers = np.hypot(
        pypsa.transformers_t["p0"], pypsa.transformers_t["q0"]
    ).truediv(bus0_v_mag_abs, axis="columns") / sqrt(3)
    edisgo.results._i_res = pd.concat([current_lines, current_transformers], axis=1)

    # get voltage results in kV
    edisgo.results._v_res = pypsa.buses_t["v_mag_pu"].astype(dtype)

    # save seeds
    edisgo.results.pfa_v_mag_pu_seed = pd.concat(
        [
            edisgo.results.pfa_v_mag_pu_seed,
            pypsa.buses_t["v_mag_pu"].reindex(index=timesteps),
        ]
    )
    edisgo.results.pfa_v_mag_pu_seed = edisgo.results.pfa_v_mag_pu_seed[
        ~edisgo.results.pfa_v_mag_pu_seed.index.duplicated(keep="last")
    ].fillna(1)

    edisgo.results.pfa_v_ang_seed = pd.concat(
        [
            edisgo.results.pfa_v_ang_seed,
            pypsa.buses_t["v_ang"].reindex(index=timesteps),
        ]
    )
    edisgo.results.pfa_v_ang_seed = edisgo.results.pfa_v_ang_seed[
        ~edisgo.results.pfa_v_ang_seed.index.duplicated(keep="last")
    ].fillna(0)
