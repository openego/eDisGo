import os

from copy import deepcopy

import networkx as nx

from edisgo.network.components import Switch


def remove_1m_lines_from_edisgo(edisgo):
    """
    Method to remove 1m lines to reduce size of edisgo object.
    """
    print("Removing 1m lines for grid {}".format(repr(edisgo)))

    # close switches such that lines with connected switches are not removed
    switches = [
        Switch(id=_, topology=edisgo.topology)
        for _ in edisgo.topology.switches_df.index
    ]
    switch_status = {}
    for switch in switches:
        switch_status[switch] = switch.state
        switch.close()
    # get all lines and remove end 1m lines
    lines = edisgo.topology.lines_df.loc[edisgo.topology.lines_df.length == 0.001]
    for name, line in lines.iterrows():
        remove_1m_end_line(edisgo, line)
    # set switches back to original state
    for switch in switches:
        if switch_status[switch] == "open":
            switch.open()
    return edisgo


def remove_1m_end_line(edisgo, line):
    """
    Method that removes end lines and moves components of end bus to neighboring bus.
    If the line is not an end line, the method will skip this line
    """
    # Check for end buses
    if len(edisgo.topology.get_connected_lines_from_bus(line.bus1)) == 1:
        end_bus = "bus1"
        neighbor_bus = "bus0"
    elif len(edisgo.topology.get_connected_lines_from_bus(line.bus0)) == 1:
        end_bus = "bus0"
        neighbor_bus = "bus1"
    else:
        end_bus = None
        neighbor_bus = None
        print("No end bus found. Implement method.")
        return
    # Move connected elements of end bus to the other bus
    connected_elements = edisgo.topology.get_connected_components_from_bus(
        line[end_bus]
    )
    # move elements to neighboring bus
    rename_dict = {line[end_bus]: line[neighbor_bus]}
    for Type, components in connected_elements.items():
        if not components.empty and Type != "lines":
            setattr(
                edisgo.topology,
                Type.lower() + "_df",
                getattr(edisgo.topology, Type.lower() + "_df").replace(rename_dict),
            )
    # remove line
    edisgo.topology.remove_line(line.name)
    print("{} removed.".format(line.name))


def extract_feeders_nx(edisgo_obj, save_dir=None, only_flex_ev=True):
    """
    Method to extract and optionally save MV-feeders.
    """
    edisgo_orig = deepcopy(edisgo_obj)
    buses_with_feeders = edisgo_orig.topology.buses_df
    station_bus = edisgo_obj.topology.mv_grid.station.index[0]
    # get lines connected to station
    feeder_lines = edisgo_obj.topology.lines_df.loc[
        edisgo_obj.topology.lines_df.bus0 == station_bus
    ].append(
        edisgo_obj.topology.lines_df.loc[
            edisgo_obj.topology.lines_df.bus1 == station_bus
        ]
    )
    for feeder_line in feeder_lines.index:
        edisgo_obj.remove_component("Line", feeder_line, force_remove=True)
    graph = edisgo_obj.topology.to_graph()
    subgraphs = list(graph.subgraph(c) for c in nx.connected_components(graph))
    feeders = []
    feeder_id = 0
    for subgraph in subgraphs:
        if only_flex_ev:
            cp_feeder = edisgo_obj.topology.charging_points_df.loc[
                edisgo_obj.topology.charging_points_df.bus.isin(list(subgraph.nodes))
                & edisgo_obj.topology.charging_points_df.use_case.isin(["home", "work"])
            ]
            if len(cp_feeder) > 0:
                buses_with_feeders.loc[list(subgraph.nodes), "feeder_id"] = feeder_id
                edisgo_feeder = create_feeder_edisgo_object(
                    buses_with_feeders, edisgo_orig, feeder_id
                )
                if save_dir:
                    os.makedirs(
                        save_dir + "/feeder/{}".format(int(feeder_id)), exist_ok=True
                    )
                    edisgo_feeder.save(save_dir + "/feeder/{}".format(int(feeder_id)))
                feeders.append(edisgo_feeder)
                feeder_id += 1
        else:
            raise NotImplementedError(
                "So far the method is only implemented for the extraction of "
                "feeders with flexible EV. Please adapt to your case."
            )
    return feeders


def create_feeder_edisgo_object(buses_with_feeders, edisgo_obj, feeder_id):
    """
    Method to create feeder edisgo object.
    """
    edisgo_feeder = deepcopy(edisgo_obj)
    # convert topology
    edisgo_feeder.topology.buses_df = edisgo_obj.topology.buses_df.loc[
        buses_with_feeders.feeder_id == feeder_id
    ].append(edisgo_feeder.topology.mv_grid.station)
    # Todo: code more efficiently using setattr and getattr
    edisgo_feeder.topology.lines_df = edisgo_obj.topology.lines_df.loc[
        edisgo_obj.topology.lines_df.bus0.isin(edisgo_feeder.topology.buses_df.index)
    ].loc[edisgo_obj.topology.lines_df.bus1.isin(edisgo_feeder.topology.buses_df.index)]
    edisgo_feeder.topology.transformers_df = edisgo_obj.topology.transformers_df.loc[
        edisgo_obj.topology.transformers_df.bus0.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ].loc[
        edisgo_obj.topology.transformers_df.bus1.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ]
    edisgo_feeder.topology.generators_df = edisgo_obj.topology.generators_df.loc[
        edisgo_obj.topology.generators_df.bus.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ]
    edisgo_feeder.topology.loads_df = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.bus.isin(edisgo_feeder.topology.buses_df.index)
    ]
    edisgo_feeder.topology.storage_units_df = edisgo_obj.topology.storage_units_df.loc[
        edisgo_obj.topology.storage_units_df.bus.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ]
    edisgo_feeder.topology.charging_points_df = (
        edisgo_obj.topology.charging_points_df.loc[
            edisgo_obj.topology.charging_points_df.bus.isin(
                edisgo_feeder.topology.buses_df.index
            )
        ]
    )
    edisgo_feeder.topology.switches_df = edisgo_obj.topology.switches_df.loc[
        edisgo_obj.topology.switches_df.branch.isin(
            edisgo_feeder.topology.lines_df.index
        )
        & edisgo_obj.topology.switches_df.bus_open.isin(
            edisgo_feeder.topology.buses_df.index
        )
        & edisgo_obj.topology.switches_df.bus_closed.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ]
    # convert timeseries
    # Todo: code more efficiently using setattr and getattr
    if not edisgo_obj.timeseries.charging_points_active_power.empty:
        edisgo_feeder.timeseries.charging_points_active_power = (
            edisgo_obj.timeseries.charging_points_active_power[
                edisgo_feeder.topology.charging_points_df.index
            ]
        )
    if not edisgo_obj.timeseries.charging_points_reactive_power.empty:
        edisgo_feeder.timeseries.charging_points_reactive_power = (
            edisgo_obj.timeseries.charging_points_reactive_power[
                edisgo_feeder.topology.charging_points_df.index
            ]
        )
    if not edisgo_obj.timeseries.generators_active_power.empty:
        edisgo_feeder.timeseries.generators_active_power = (
            edisgo_obj.timeseries.generators_active_power[
                edisgo_feeder.topology.generators_df.index
            ]
        )
    if not edisgo_obj.timeseries.generators_reactive_power.empty:
        edisgo_feeder.timeseries.generators_reactive_power = (
            edisgo_obj.timeseries.generators_reactive_power[
                edisgo_feeder.topology.generators_df.index
            ]
        )
    if not edisgo_obj.timeseries.loads_active_power.empty:
        edisgo_feeder.timeseries.loads_active_power = (
            edisgo_obj.timeseries.loads_active_power[
                edisgo_feeder.topology.loads_df.index
            ]
        )
    if not edisgo_obj.timeseries.loads_reactive_power.empty:
        edisgo_feeder.timeseries.loads_reactive_power = (
            edisgo_obj.timeseries.loads_reactive_power[
                edisgo_feeder.topology.loads_df.index
            ]
        )
    if not edisgo_obj.timeseries.storage_units_active_power.empty:
        edisgo_feeder.timeseries.storage_units_active_power = (
            edisgo_obj.timeseries.storage_units_active_power[
                edisgo_feeder.topology.storage_units_df.index
            ]
        )
    if not edisgo_obj.timeseries.storage_units_reactive_power.empty:
        edisgo_feeder.timeseries.storage_units_reactive_power = (
            edisgo_obj.timeseries.storage_units_reactive_power[
                edisgo_feeder.topology.storage_units_df.index
            ]
        )
    return edisgo_feeder
