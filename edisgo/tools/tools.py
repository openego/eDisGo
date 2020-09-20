import pandas as pd
import networkx as nx

from math import pi, sqrt

from edisgo.flex_opt import exceptions
from edisgo.flex_opt import check_tech_constraints


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
        :pandas:`pandas.Timestamp<Timestamp>` or None.

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


def calculate_relative_line_load(
    edisgo_obj, lines=None, timesteps=None
):
    """
    Calculates relative line loading for specified lines and time steps.

    Line loading is calculated by dividing the current at the given time step
    by the allowed current.


    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lines : list(str) or None, optional
        Line names/representatives of lines to calculate line loading for. If
        None, line loading is calculated for all lines in the network.
        Default: None.
    timesteps : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
        Specifies time steps to calculate line loading for. If timesteps is
        None, all time steps power flow analysis was conducted for are used.
        Default: None.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with relative line loading (unitless). Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`,
        columns are the line representatives.

    """
    if timesteps is None:
        timesteps = edisgo_obj.results.i_res.index
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    if lines is not None:
        line_indices = lines
    else:
        line_indices = edisgo_obj.topology.lines_df.index

    mv_lines_allowed_load = check_tech_constraints.lines_allowed_load(
        edisgo_obj, "mv")
    lv_lines_allowed_load = check_tech_constraints.lines_allowed_load(
        edisgo_obj, "lv")
    lines_allowed_load = pd.concat(
        [mv_lines_allowed_load, lv_lines_allowed_load],
        axis=1, sort=False).loc[timesteps, line_indices]

    return check_tech_constraints.lines_relative_load(
        edisgo_obj, lines_allowed_load)


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
    dataframe::pandas:`pandas.DataFrame<DataFrame>`
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
    :pandas:`pandas.Series<Series>`
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


def assign_voltage_level_to_component(edisgo_obj, df):
    """
    Adds column with specification of voltage level component is in.

    The voltage level ('mv' or 'lv') is determined based on the nominal
    voltage of the bus the component is connected to. If the nominal voltage
    is smaller than 1 kV, voltage level 'lv' is assigned, otherwise 'mv' is
    assigned.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names in the index. Only required column is
        column 'bus', giving the name of the bus the component is connected to.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Same dataframe as given in parameter `df` with new column
        'voltage_level' specifying the voltage level the component is in
        (either 'mv' or 'lv').

    """
    df["voltage_level"] = df.apply(
        lambda _: "lv"
        if edisgo_obj.topology.buses_df.at[_.bus, "v_nom"] < 1
        else "mv",
        axis=1,
    )
    return df
