import os

import pandas as pd

from pypsa import Network as PyPSANetwork

from edisgo.network.components import Switch
from edisgo.network.grids import MVGrid

if "READTHEDOCS" not in os.environ:
    from shapely.wkt import loads as wkt_loads

import logging

logger = logging.getLogger(__name__)


def import_ding0_grid(path, edisgo_obj):
    """
    Import an eDisGo network topology from
    `Ding0 data <https://github.com/openego/ding0>`_.

    This import method is specifically designed to load network topology data in
    the format as `Ding0 <https://github.com/openego/ding0>`_ provides it via
    csv files.

    Parameters
    ----------
    path : str
        Path to ding0 network csv files.
    edisgo_obj: :class:`~.EDisGo`
        The eDisGo data container object.

    """

    def sort_transformer_buses(transformers_df):
        """
        Sort buses of inserted transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        voltage_bus0 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus0
        ].v_nom.values
        voltage_bus1 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus1
        ].v_nom.values
        transformers_df.loc[
            voltage_bus1 > voltage_bus0, ["bus0", "bus1"]
        ] = transformers_df.loc[voltage_bus1 > voltage_bus0, ["bus1", "bus0"]].values
        return transformers_df

    def sort_hvmv_transformer_buses(transformers_df):
        """
        Sort buses of inserted HV/MV transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        for transformer in transformers_df.index:
            if (
                transformers_df.at[transformer, "bus1"]
                in edisgo_obj.topology.buses_df.index
            ):
                continue

            transformers_df.loc[transformer, ["bus0", "bus1"]] = transformers_df.loc[
                transformer, ["bus1", "bus0"]
            ].values

        return transformers_df

    grid = PyPSANetwork()
    grid.import_from_csv_folder(path)

    # write dataframes to edisgo_obj
    edisgo_obj.topology.buses_df = grid.buses[edisgo_obj.topology.buses_df.columns]
    edisgo_obj.topology.lines_df = grid.lines[edisgo_obj.topology.lines_df.columns]

    grid.loads = grid.loads.drop(columns="p_set").rename(columns={"peak_load": "p_set"})

    edisgo_obj.topology.loads_df = grid.loads[edisgo_obj.topology.loads_df.columns]
    # set loads without type information to be conventional loads
    # this is done, as ding0 currently does not provide information on the type of load
    # but ding0 grids currently also only contain conventional loads
    # ToDo: Change, once information is provided by ding0
    loads_without_type = edisgo_obj.topology.loads_df[
        (edisgo_obj.topology.loads_df.type.isnull())
        | (edisgo_obj.topology.loads_df.type == "")
    ].index
    edisgo_obj.topology.loads_df.loc[loads_without_type, "type"] = "conventional_load"
    # drop slack generator from generators
    slack = grid.generators.loc[grid.generators.control == "Slack"].index
    grid.generators.drop(slack, inplace=True)
    edisgo_obj.topology.generators_df = grid.generators[
        edisgo_obj.topology.generators_df.columns
    ]
    edisgo_obj.topology.storage_units_df = grid.storage_units[
        edisgo_obj.topology.storage_units_df.columns
    ]
    edisgo_obj.topology.transformers_df = sort_transformer_buses(
        grid.transformers.drop(labels=["x_pu", "r_pu"], axis=1).rename(
            columns={"r": "r_pu", "x": "x_pu"}
        )[edisgo_obj.topology.transformers_df.columns]
    )
    edisgo_obj.topology.transformers_hvmv_df = sort_hvmv_transformer_buses(
        pd.read_csv(os.path.join(path, "transformers_hvmv.csv"), index_col=[0]).rename(
            columns={"r": "r_pu", "x": "x_pu"}
        )
    )
    edisgo_obj.topology.switches_df = pd.read_csv(
        os.path.join(path, "switches.csv"), index_col=[0]
    )

    edisgo_obj.topology.grid_district = {
        "population": grid.mv_grid_district_population,
        "geom": wkt_loads(grid.mv_grid_district_geom),
        "srid": grid.srid,
    }

    # set up medium voltage grid
    mv_grid_id = list(set(grid.buses.mv_grid_id))[0]
    edisgo_obj.topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)

    # check data integrity
    edisgo_obj.topology.check_integrity()


def remove_1m_end_lines(edisgo):
    """
    Method to remove 1m end lines to reduce size of edisgo object.

    Short lines inside houses are removed in this function, including the end node.
    Components that were originally connected to the end node are reconnected to the
    upstream node.

    This function will become obsolete once it is changed in the ding0 export.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`

    Returns
    --------
    edisgo : :class:`~.EDisGo`
        EDisGo object where 1m end lines are removed from topology.

    """

    def remove_1m_end_line(edisgo, line):
        """
        Method that removes end lines and moves components of end bus to neighboring
        bus. If the line is not an end line, the method will skip this line.

        Returns
        -------
        int
            Number of removed lines. Either 0, if no line was removed, or 1, if line
            was removed.

        """
        # check for end buses
        if len(edisgo.topology.get_connected_lines_from_bus(line.bus1)) == 1:
            end_bus = "bus1"
            neighbor_bus = "bus0"
        elif len(edisgo.topology.get_connected_lines_from_bus(line.bus0)) == 1:
            end_bus = "bus0"
            neighbor_bus = "bus1"
        else:
            return 0

        # move connected elements of end bus to the other bus
        connected_elements = edisgo.topology.get_connected_components_from_bus(
            line[end_bus]
        )
        rename_dict = {line[end_bus]: line[neighbor_bus]}
        for comp_type, components in connected_elements.items():
            if not components.empty and comp_type != "lines":
                setattr(
                    edisgo.topology,
                    comp_type.lower() + "_df",
                    getattr(edisgo.topology, comp_type.lower() + "_df").replace(
                        rename_dict
                    ),
                )

        # remove line
        edisgo.topology.remove_line(line.name)
        return 1

    # close switches such that lines with connected switches are not removed
    switches = [
        Switch(id=_, topology=edisgo.topology)
        for _ in edisgo.topology.switches_df.index
    ]
    switch_status = {}
    for switch in switches:
        switch_status[switch] = switch.state
        switch.close()

    # get all lines with length of one meter and remove the ones that are end lines
    number_of_lines_removed = 0
    lines = edisgo.topology.lines_df.loc[edisgo.topology.lines_df.length == 0.001]
    for name, line in lines.iterrows():
        number_of_lines_removed += remove_1m_end_line(edisgo, line)
    logger.debug(f"Removed {number_of_lines_removed} 1 m end lines.")

    # set switches back to original state
    for switch in switches:
        if switch_status[switch] == "open":
            switch.open()
    return edisgo
