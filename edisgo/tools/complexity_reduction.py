import logging
import os

from copy import deepcopy

import networkx as nx
import pandas as pd

from edisgo.network.components import Switch

logger = logging.getLogger(__name__)


def remove_1m_lines_from_edisgo(edisgo):
    """
    Method to remove 1m lines to reduce size of edisgo object.
    """
    logger.info("Removing 1m lines for grid {}".format(repr(edisgo)))

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
        logger.info("No end bus found. Implement method.")
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
    logger.info("{} removed.".format(line.name))


def extract_feeders_nx(
    edisgo_obj,
    export_path=None,
    only_flex_ev=False,
    flexible_loads: pd.DataFrame = None,
):
    """
    Method to extract and optionally save MV-feeders.
    """

    def _extract_feeder(
        edisgo_orig,
        subgraph,
        feeder_id,
        buses_with_feeders,
        flexible_loads,
        export_path=None,
    ):
        if len(list(set(subgraph.nodes))) > 1:
            buses_with_feeders.loc[list(subgraph.nodes), "feeder_id"] = feeder_id
            edisgo_feeder = create_feeder_edisgo_object(
                buses_with_feeders,
                edisgo_orig,
                feeder_id,
                flexible_loads=flexible_loads,
            )
            edisgo_feeder.topology.mv_grid
            if edisgo_feeder is None:
                # edisgo_feeder == None if loads_df is empty
                return None

            else:
                if export_path is not None:
                    # export_path = export_path / "feeder" / f"{feeder_id:02}"
                    export_path = export_path / f"{feeder_id:02}"
                    os.makedirs(export_path, exist_ok=True)
                    edisgo_feeder.save(
                        export_path,
                        save_topology=True,
                        save_timeseries=True,
                        save_heatpump=True,
                        save_electromobility=True,
                        electromobility_attributes=[
                            "integrated_charging_parks_df",
                            "simbev_config_df",
                            "flexibility_bands",
                        ],
                    )
                    logger.info(f"Saved feeder: {feeder_id} to {export_path}")
                feeder_id += 1
                return edisgo_feeder

        else:
            logger.info(
                f"Feeder {feeder_id} is ignored as it doesn't have any "
                f"significant subgraph nodes."
            )
            return None

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
        edisgo_obj.remove_component("line", feeder_line, force_remove=True)
    graph = edisgo_obj.topology.to_graph()
    subgraphs = list(graph.subgraph(c) for c in nx.connected_components(graph))
    feeders = []
    ignored_feeder = []
    if only_flex_ev:
        logger.info("Only flex ev active for feeder extraction")
    for feeder_id, subgraph in enumerate(subgraphs):
        if only_flex_ev:

            cp_feeder = edisgo_obj.topology.charging_points_df.loc[
                edisgo_obj.topology.charging_points_df.bus.isin(list(subgraph.nodes))
                & edisgo_obj.topology.charging_points_df.sector.isin(
                    ["home", "work", 1, "1"]
                )
            ]

            if len(cp_feeder) > 0:
                feeder = _extract_feeder(
                    edisgo_orig=edisgo_orig,
                    subgraph=subgraph,
                    feeder_id=feeder_id,
                    export_path=export_path,
                    buses_with_feeders=buses_with_feeders,
                    flexible_loads=flexible_loads,
                )
        else:
            feeder = _extract_feeder(
                edisgo_orig=edisgo_orig,
                subgraph=subgraph,
                feeder_id=feeder_id,
                export_path=export_path,
                buses_with_feeders=buses_with_feeders,
                flexible_loads=flexible_loads,
            )
        if feeder is None:
            ignored_feeder += [str(feeder_id)]
        else:
            feeders.append(feeder)

    logger.info(f"Ignored feeder: {', '.join(ignored_feeder)}.")
    return feeders, buses_with_feeders


def create_feeder_edisgo_object(
    buses_with_feeders, edisgo_obj, feeder_id, flexible_loads: pd.DataFrame = None
):
    """
    Method to create feeder edisgo object.
    """
    edisgo_feeder = deepcopy(edisgo_obj)

    # select topology
    # get buses of feeder and append mv-station
    edisgo_feeder.topology.buses_df = edisgo_obj.topology.buses_df.loc[
        buses_with_feeders.feeder_id == feeder_id
    ].append(edisgo_obj.topology.mv_grid.station)

    feeder_buses = edisgo_feeder.topology.buses_df.index

    attr_list = [
        "lines_df",
        "transformers_df",
    ]

    for attr_name in attr_list:
        attr_old = getattr(edisgo_obj.topology, attr_name)
        if not attr_old.empty:
            attr_new = attr_old.loc[
                attr_old.bus0.isin(feeder_buses) & attr_old.bus1.isin(feeder_buses)
            ]
            if attr_new.empty:
                logger.info(f"{attr_name} of feeder {feeder_id} is empty.")
            setattr(edisgo_feeder.topology, attr_name, attr_new)

    attr_list = [
        "generators_df",
        "loads_df",
        "storage_units_df",
        # "charging_points_df", # already covered in loads_df
    ]

    for attr_name in attr_list:
        attr_old = getattr(edisgo_obj.topology, attr_name)
        if not attr_old.empty:
            attr_new = attr_old.loc[attr_old.bus.isin(feeder_buses)]
            if attr_new.empty:
                logger.info(f"{attr_name} of feeder {feeder_id} is empty.")
            setattr(edisgo_feeder.topology, attr_name, attr_new)

    # get switches connected to a line of this feeder with open bus
    # TODO switches get lost?!
    edisgo_feeder.topology.switches_df = edisgo_obj.topology.switches_df.loc[
        edisgo_obj.topology.switches_df.branch.isin(
            edisgo_feeder.topology.lines_df.index
        )
        & edisgo_obj.topology.switches_df.bus_open.isin(
            edisgo_feeder.topology.buses_df.index
        )
    ]

    # select timeseries
    attr_list = {
        "generators_active_power": "generators_df",
        "generators_reactive_power": "generators_df",
        "loads_active_power": "loads_df",
        "loads_reactive_power": "loads_df",
        "storage_units_active_power": "storage_units_df",
        "storage_units_reactive_power": "storage_units_df",
    }

    for attr_name, id_attr in attr_list.items():

        # get attribute
        attr_old = getattr(edisgo_obj.timeseries, attr_name)
        if not attr_old.empty:

            # uncommented to keep all loads ts
            # # get ids for attribute but remove flexible loads
            # if isinstance(flexible_loads, pd.DataFrame) & (id_attr == "loads_df"):
            #     id_attr = getattr(edisgo_feeder.topology, id_attr)
            #     # remove flexible loads
            #     id_attr = id_attr.drop(
            #         id_attr.loc[id_attr.index.isin(flexible_loads.index)].index
            #     )
            # else:
            #     id_attr = getattr(edisgo_feeder.topology, id_attr)

            id_attr = getattr(edisgo_feeder.topology, id_attr)

            # reduce attribute
            attr_new = attr_old.loc[:, id_attr.index]
            if attr_new.empty:
                logger.info(f"{attr_name} of feeder {feeder_id} is empty.")
            # set attribute
            setattr(edisgo_feeder.timeseries, attr_name, attr_new)

    # select emob attributes
    emob_ids = flexible_loads.loc[
        (flexible_loads["type"] == "charging_point")
        & (flexible_loads["bus"].isin(feeder_buses))
    ].index

    for band, df in edisgo_obj.electromobility.flexibility_bands.items():
        if not df.empty:
            df = df.loc[:, df.columns.isin(emob_ids)]
            edisgo_feeder.electromobility.flexibility_bands.update({band: df})

    # select heat pump attributes
    attr_list = ["heat_demand_df", "cop_df", "thermal_storage_units_df"]

    hp_ids = edisgo_feeder.topology.loads_df.loc[
        edisgo_feeder.topology.loads_df["type"] == "heat_pump"
    ].index

    for attr_name in attr_list:
        attr_old = getattr(edisgo_obj.heat_pump, attr_name)
        if not attr_old.empty:
            if attr_name == "thermal_storage_units_df":
                attr_new = attr_old.loc[hp_ids]
            else:
                attr_new = attr_old.loc[:, hp_ids]

            if attr_new.empty:
                logger.info(f"{attr_name} of feeder {feeder_id} is empty.")
            setattr(edisgo_feeder.heat_pump, attr_name, attr_new)

    return edisgo_feeder
