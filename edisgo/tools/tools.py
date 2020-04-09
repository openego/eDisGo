import pandas as pd
import networkx as nx
from networkx import OrderedGraph
from math import pi, sqrt

from edisgo.flex_opt import exceptions


def select_worstcase_snapshots(edisgo_obj):
    """
    Select two worst-case snapshots from time series

    Two time steps in a time series represent worst-case snapshots. These are

    1. Load case: refers to the point in the time series where the
        (load - generation) achieves its maximum and is greater than 0.
    2. Feed-in case: refers to the point in the time series where the
        (load - generation) achieves its minimum and is smaller than 0.

    These two points are identified based on the generation and load time
    series. In case load or feed-in case don't exist None is returned.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :obj:`dict`
        Dictionary with keys 'load_case' and 'feedin_case'. Values are
        corresponding worst-case snapshots of type
        :pandas:`pandas.Timestamp<timestamp>` or None.

    """
    residual_load = edisgo_obj.timeseries.residual_load

    timestamp = {}
    timestamp["load_case"] = (
        residual_load.idxmin() if min(residual_load) < 0 else None
    )
    timestamp["feedin_case"] = (
        residual_load.idxmax() if max(residual_load) > 0 else None
    )
    return timestamp


def get_residual_load_from_pypsa_network(edisgo_obj):
    """
    Calculates residual load in MW in MV network and underlying LV grids.

    Parameters
    ----------
    edisgo_obj :class:`~edisgo.EDisGo`

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Series with residual load in MW for each time step. Positiv values
        indicate a higher demand than generation and vice versa. Index of the
        series is a :pandas:`pandas.DatetimeIndex<datetimeindex>`

    """
    # Todo: write test
    loads_active_power = edisgo_obj.timeseries.loads_active_power.sum(axis=1)
    generators_active_power = edisgo_obj.timeseries.generators_active_power.loc[
        :,
        edisgo_obj.timeseries.generators_active_power.columns
        != "Generator_slack",
    ].sum(
        axis=1
    )
    storage_units_active_power = edisgo_obj.timeseries.storage_units_active_power.sum(
        axis=1
    )

    residual_load = loads_active_power - (
        generators_active_power + storage_units_active_power
    )
    return residual_load


def calculate_relative_line_load(
    edisgo_obj, line_load, lines=None, timesteps=None
):
    """
    Calculates relative line loading.

    Line loading is calculated by dividing the current at the given time step
    by the allowed current.


    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
        Pypsa network with lines to calculate line loading for.
    line_load : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with current results from power flow analysis in A. Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<datetimeindex>`,
        columns are the line representatives.
    lines : list(str) or None, optional
        Line names/representatives of lines to calculate line loading for. If
        None line loading of all lines in `line_load` dataframe are used.
        Default: None.
    timesteps : :pandas:`pandas.Timestamp<timestamp>` or list(:pandas:`pandas.Timestamp<timestamp>`) or None, optional
        Specifies time steps to calculate line loading for. If timesteps is
        None all time steps in `line_load` dataframe are used. Default: None.

    Returns
    --------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with relative line loading (unitless). Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<datetimeindex>`,
        columns are the line representatives.

    """
    # Todo: write test
    if timesteps is None:
        timesteps = line_load.index
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    if lines is not None:
        line_indices = lines
    else:
        line_indices = line_load.columns

    residual_load = get_residual_load_from_pypsa_network(edisgo_obj)
    case = residual_load.apply(
        lambda _: "feedin_case" if _ < 0 else "load_case"
    )

    load_factor = pd.DataFrame(
        data={
            "i_nom": [
                float(
                    edisgo_obj.config["grid_expansion_load_factors"][
                        "mv_{}_line".format(case.loc[_])
                    ]
                )
                for _ in timesteps
            ]
        },
        index=timesteps,
    )

    # current from power flow
    i_res = line_load.loc[timesteps, line_indices]
    # allowed current
    lines = edisgo_obj.topology.lines_df.loc[line_indices]
    lines = lines.join(
        edisgo_obj.topology.buses_df.loc[lines.bus0, "v_nom"],
        on="bus0",
        how="left",
    ).drop_duplicates()
    i_allowed = load_factor.dot(
        (lines.s_nom / (sqrt(3) * lines.v_nom)).to_frame("i_nom").T
    )

    return i_res.divide(i_allowed)


def calculate_line_reactance(line_inductance_per_km, line_length):
    """
    Calculates line reactance in Ohm from given line data and length.

    Parameters
    ----------
    line_inductance_per_km : float or array-like
        Line inductance in mH/km.
    line_length : float
        Length of line in km.

    Returns
    -------
    float
        Reactance in Ohm

    """
    return line_inductance_per_km / 1e3 * line_length * 2 * pi * 50


def calculate_line_resistance(line_resistance_per_km, line_length):
    """
    Calculates line resistance in Ohm from given line data and length.

    Parameters
    ----------
    line_resistance_per_km : float or array-like
        Line resistance in Ohm/km.
    line_length : float
        Length of line in km.

    Returns
    -------
    float
        Resistance in Ohm

    """
    return line_resistance_per_km * line_length


def calculate_apparent_power(nominal_voltage, current):
    """
    Calculates apparent power in MVA from given voltage and current.

    Parameters
    ----------
    nominal_voltage : float or array-like
        Nominal voltage in kV.
    current : float or array-like
        Current in kA.

    Returns
    -------
    float
        Apparent power in MVA.

    """
    return sqrt(3) * nominal_voltage * current


def translate_df_to_graph(buses_df, lines_df, transformers_df=None):
    graph = OrderedGraph()

    buses = buses_df.index
    # add nodes
    graph.add_nodes_from(buses)
    # add branches
    branches = []
    for line_name, line in lines_df.iterrows():
        branches.append(
            (
                line.bus0,
                line.bus1,
                {"branch_name": line_name, "length": line.length},
            )
        )
    if transformers_df is not None:
        for trafo_name, trafo in transformers_df.iterrows():
            branches.append(
                (
                    trafo.bus0,
                    trafo.bus1,
                    {"branch_name": trafo_name, "length": 0},
                )
            )
    graph.add_edges_from(branches)
    return graph


def check_bus_for_removal(topology, bus_name):
    """
    Checks whether bus is connected to elements other than one line. Returns
    True if bus of inserted name is only connected to one line. Returns False
    if bus is connected to other element or additional line.


    Parameters
    ----------
    topology: :class:`~.network.topology.Topology`
        Topology object containing bus of name bus_name
    bus_name: str
        Name of bus which has to be checked

    Returns
    -------
    Removable: bool
        Indicator if bus of name bus_name can be removed from topology
    """
    # Todo: move to topology?
    # check if bus is party of topology
    if bus_name not in topology.buses_df.index:
        raise ValueError(
            "Bus of name {} not in Topology. Cannot be checked "
            "to be removed.".format(bus_name)
        )
    connected_lines = topology.get_connected_lines_from_bus(bus_name)
    # if more than one line is connected to node, it cannot be removed
    if len(connected_lines) > 1:
        return False
    # if another element is connected to node, it cannot be removed
    elif (
        bus_name in topology.loads_df.bus.values
        or bus_name in topology.generators_df.bus.values
        or bus_name in topology.storage_units_df.bus.values
        or bus_name in topology.transformers_df.bus0.values
        or bus_name in topology.transformers_df.bus1.values
    ):
        return False
    else:
        return True


def check_line_for_removal(topology, line_name):
    """
    Checks whether line can be removed without leaving isolated nodes. Returns
    True if line can be removed safely.


    Parameters
    ----------
    topology: :class:`~.network.topology.Topology`
        Topology object containing bus of name bus_name
    line_name: str
        Name of line which has to be checked

    Returns
    -------
    Removable: bool
        Indicator if line of name line_name can be removed from topology
        without leaving isolated node

    """
    # Todo: move to topology?
    # check if line is part of topology
    if line_name not in topology.lines_df.index:
        raise ValueError(
            "Line of name {} not in Topology. Cannot be checked "
            "to be removed.".format(line_name)
        )

    bus0 = topology.lines_df.loc[line_name, "bus0"]
    bus1 = topology.lines_df.loc[line_name, "bus1"]
    # if either of the buses can be removed as well, line can be removed safely
    if check_bus_for_removal(topology, bus0) or check_bus_for_removal(
        topology, bus1
    ):
        return True
    # otherwise both buses have to be connected to at least two lines
    if (
        len(topology.get_connected_lines_from_bus(bus0)) > 1
        and len(topology.get_connected_lines_from_bus(bus1)) > 1
    ):
        return True
    else:
        return False
    # Todo: add check for creation of subnetworks, so far it is only checked,
    #  if isolated node would be created. It could still happen that two sub-
    #  networks are created by removing the line.


def drop_duplicated_indices(dataframe, keep="first"):
    """
    Drop rows of duplicate indices in dataframe.

    Parameters
    ----------
    dataframe::pandas:`pandas.DataFrame<dataframe>`
        handled dataframe
    keep: str
        indicator of row to be kept, 'first', 'last' or False,
        see pandas.DataFrame.drop_duplicates() method
    """
    return dataframe[~dataframe.index.duplicated(keep=keep)]


def select_cable(edisgo_obj, level, apparent_power):
    """Selects an appropriate cable type and quantity using given apparent
    power.

    ToDo: adapt to refactored code!

    Considers load factor.

    Parameters
    ----------
    edisgo_obj : :class:`~.network.topology.Topology`
        The eDisGo container object
    level : :obj:`str`
        Grid level ('mv' or 'lv')
    apparent_power : :obj:`float`
        Apparent power the cable must carry in kVA

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Cable type
    :obj:`ìnt`
        Cable count

    Notes
    ------
    Cable is selected to be able to carry the given `apparent_power`, no load
    factor is considered.

    """
    raise NotImplementedError

    cable_count = 1

    if level == "mv":

        cable_data = edisgo_obj.topology.equipment_data["mv_cables"]
        available_cables = cable_data[
            cable_data["U_n"] == edisgo_obj.topology.mv_grid.voltage_nom
        ]

        suitable_cables = available_cables[
            available_cables["I_max_th"]
            * edisgo_obj.topology.mv_grid.voltage_nom
            > apparent_power
        ]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty and cable_count < 20:
            cable_count += 1
            suitable_cables = available_cables[
                available_cables["I_max_th"]
                * edisgo_obj.topology.mv_grid.voltage_nom
                * cable_count
                > apparent_power
            ]
        if suitable_cables.empty and cable_count == 20:
            raise exceptions.MaximumIterationError(
                "Could not find a suitable cable for apparent power of "
                "{} kVA.".format(apparent_power)
            )

        cable_type = suitable_cables.ix[suitable_cables["I_max_th"].idxmin()]

    elif level == "lv":

        cable_data = edisgo_obj.topology.equipment_data["lv_cables"]
        suitable_cables = cable_data[
            cable_data["I_max_th"] * cable_data["U_n"] > apparent_power
        ]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty and cable_count < 20:
            cable_count += 1
            suitable_cables = cable_data[
                cable_data["I_max_th"] * cable_data["U_n"] * cable_count
                > apparent_power
            ]
        if suitable_cables.empty and cable_count == 20:
            raise exceptions.MaximumIterationError(
                "Could not find a suitable cable for apparent power of "
                "{} kVA.".format(apparent_power)
            )

        cable_type = suitable_cables.ix[suitable_cables["I_max_th"].idxmin()]

    else:
        raise ValueError("Please supply a level (either 'mv' or 'lv').")

    return cable_type, cable_count


def assign_mv_feeder_to_nodes(mv_grid):
    """
    Assigns an MV feeder to every generator, LV station, load, and branch tee

    ToDo: adapt to refactored code!

    Parameters
    -----------
    mv_grid : :class:`~.network.grids.MVGrid`

    """
    raise NotImplementedError

    mv_station_neighbors = list(mv_grid.graph.neighbors(mv_grid.station))
    # get all nodes in MV network and remove MV station to get separate subgraphs
    mv_graph_nodes = list(mv_grid.graph.nodes())
    mv_graph_nodes.remove(mv_grid.station)
    subgraph = mv_grid.graph.subgraph(mv_graph_nodes)

    for neighbor in mv_station_neighbors:
        # determine feeder
        mv_feeder = mv_grid.graph.line_from_nodes(mv_grid.station, neighbor)
        # get all nodes in that feeder by doing a DFS in the disconnected
        # subgraph starting from the node adjacent to the MVStation `neighbor`
        subgraph_neighbor = nx.dfs_tree(subgraph, source=neighbor)
        for node in subgraph_neighbor.nodes():
            # in case of an LV station assign feeder to all nodes in that LV
            # network
            if isinstance(node, LVStation):
                for lv_node in node.grid.graph.nodes():
                    lv_node.mv_feeder = mv_feeder
            else:
                node.mv_feeder = mv_feeder


def get_mv_feeder_from_line(line):
    """
    Determines MV feeder the given line is in.

    ToDo: adapt to refactored code!

    MV feeders are identified by the first line segment of the half-ring.

    Parameters
    ----------
    line : :class:`~.network.components.Line`
        Line to find the MV feeder for.

    Returns
    -------
    :class:`~.network.components.Line`
        MV feeder identifier (representative of the first line segment
        of the half-ring)

    """
    raise NotImplementedError
    try:
        # get nodes of line
        nodes = line.grid.graph.nodes_from_line(line)

        # get feeders
        feeders = {}
        for node in nodes:
            # if one of the nodes is an MV station the line is an MV feeder
            # itself
            if isinstance(node, MVStation):
                feeders[repr(node)] = None
            else:
                feeders[repr(node)] = node.mv_feeder

        # return feeder that is not None
        feeder_1 = feeders[repr(nodes[0])]
        feeder_2 = feeders[repr(nodes[1])]
        if not feeder_1 is None and not feeder_2 is None:
            if feeder_1 == feeder_2:
                return feeder_1
            else:
                logging.warning("Different feeders for line {}.".format(line))
                return None
        else:
            return feeder_1 if feeder_1 is not None else feeder_2
    except Exception as e:
        logging.warning("Failed to get MV feeder: {}.".format(e))
        return None
