# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import pandas as pd

if "READTHEDOCS" not in os.environ:
    from shapely.ops import transform

from edisgo.tools.geo import proj2equidistant

logger = logging.getLogger(__name__)


def grid_expansion_costs(edisgo_obj, without_generator_import=False):
    """
    Calculates topology expansion costs for each reinforced transformer and line
    in kEUR.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    without_generator_import : bool
        If True excludes lines that were added in the generator import to
        connect new generators to the topology from calculation of topology expansion
        costs. Default: False.

    Returns
    -------
    `pandas.DataFrame<DataFrame>`
        DataFrame containing type and costs plus in the case of lines the
        line length and number of parallel lines of each reinforced
        transformer and line. Index of the DataFrame is the name of either line
        or transformer. Columns are the following:

        type : str
            Transformer size or cable name

        total_costs : float
            Costs of equipment in kEUR. For lines the line length and number of
            parallel lines is already included in the total costs.

        quantity : int
            For transformers quantity is always one, for lines it specifies the
            number of parallel lines.

        length : float
            Length of line or in case of parallel lines all lines in km.

        voltage_level : str {'lv' | 'mv' | 'mv/lv'}
            Specifies voltage level the equipment is in.

    Notes
    -------
    Total network expansion costs can be obtained through
    self.grid_expansion_costs.total_costs.sum().

    """

    def _get_transformer_costs(trafos):
        hvmv_trafos = trafos[
            trafos.index.isin(edisgo_obj.topology.transformers_hvmv_df.index)
        ].index
        mvlv_trafos = trafos[
            trafos.index.isin(edisgo_obj.topology.transformers_df.index)
        ].index
        costs_trafos = pd.DataFrame(
            {
                "costs_transformers": len(hvmv_trafos)
                * [float(edisgo_obj.config["costs_transformers"]["mv"])],
                "voltage_level": len(hvmv_trafos) * ["hv/mv"],
            },
            index=hvmv_trafos,
        )
        costs_trafos = pd.concat(
            [
                costs_trafos,
                pd.DataFrame(
                    {
                        "costs_transformers": len(mvlv_trafos)
                        * [float(edisgo_obj.config["costs_transformers"]["lv"])],
                        "voltage_level": len(mvlv_trafos) * ["mv/lv"],
                    },
                    index=mvlv_trafos,
                ),
            ]
        )
        return costs_trafos.loc[trafos.index, :]

    def _get_line_costs(lines_added):
        costs_lines = line_expansion_costs(edisgo_obj, lines_added.index)
        # Align quantity to costs_lines index and compute elementwise (vectorised
        # equivalent of the former per-row apply).
        quantity_aligned = lines_added["quantity"].reindex(costs_lines.index)
        costs_lines["costs"] = (
            costs_lines["costs_earthworks"]
            + costs_lines["costs_cable"] * quantity_aligned
        )

        return costs_lines[["costs", "voltage_level"]]

    costs = pd.DataFrame(dtype=float)

    if without_generator_import:
        equipment_changes = edisgo_obj.results.equipment_changes.loc[
            edisgo_obj.results.equipment_changes.iteration_step > 0
        ]
    else:
        equipment_changes = edisgo_obj.results.equipment_changes

    # costs for transformers
    if not equipment_changes.empty:
        transformers = equipment_changes[
            equipment_changes.equipment.str.contains("Transformer")
            | equipment_changes.equipment.str.contains("transformer")
        ]
        added_transformers = transformers[transformers["change"] == "added"]
        removed_transformers = transformers[transformers["change"] == "removed"]
        # check if any of the added transformers were later removed
        added_removed_transformers = added_transformers.loc[
            added_transformers["equipment"].isin(removed_transformers["equipment"])
        ]
        added_transformers = added_transformers[
            ~added_transformers["equipment"].isin(added_removed_transformers.equipment)
        ]
        # calculate costs for transformers
        all_trafos = pd.concat(
            [
                edisgo_obj.topology.transformers_hvmv_df,
                edisgo_obj.topology.transformers_df,
            ]
        )
        trafos = all_trafos.loc[added_transformers["equipment"]]
        # calculate costs for each transformer
        transformer_costs = _get_transformer_costs(trafos)
        costs = pd.concat(
            [
                costs,
                pd.DataFrame(
                    {
                        "type": trafos.type_info.values,
                        "total_costs": transformer_costs.costs_transformers,
                        "quantity": len(trafos) * [1],
                        "voltage_level": transformer_costs.voltage_level,
                    },
                    index=trafos.index,
                ),
            ]
        )

        # costs for lines
        # get changed lines
        lines = equipment_changes.loc[
            equipment_changes.index.isin(edisgo_obj.topology.lines_df.index)
        ]
        lines_added = lines.iloc[
            (
                lines.equipment
                == edisgo_obj.topology.lines_df.loc[lines.index, "type_info"]
            ).values
        ]["quantity"].to_frame()
        lines_added_unique = lines_added.index.unique()
        lines_added = (
            lines_added.groupby(level=0).sum().loc[lines_added_unique, ["quantity"]]
        )
        # use the minimum of quantity and num_parallel, as sometimes lines are added
        # and in a next reinforcement step removed again, e.g. when feeder is split
        # at 2/3 and a new single standard line is added
        lines_added = pd.merge(
            lines_added,
            edisgo_obj.topology.lines_df.loc[:, ["num_parallel"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        lines_added["quantity_added"] = lines_added.loc[
            :, ["quantity", "num_parallel"]
        ].min(axis=1)
        lines_added["length"] = edisgo_obj.topology.lines_df.loc[
            lines_added.index, "length"
        ]
        if not lines_added.empty:
            line_costs = _get_line_costs(lines_added)
            costs = pd.concat(
                [
                    costs,
                    pd.DataFrame(
                        {
                            "type": edisgo_obj.topology.lines_df.loc[
                                lines_added.index, "type_info"
                            ].values,
                            "total_costs": line_costs.costs.values,
                            "length": (
                                lines_added.quantity_added * lines_added.length
                            ).values,
                            "quantity": lines_added.quantity_added.values,
                            "voltage_level": line_costs.voltage_level.values,
                        },
                        index=lines_added.index,
                    ),
                ]
            )

    # if no costs incurred write zero costs to DataFrame
    if costs.empty:
        costs = pd.concat(
            [
                costs,
                pd.DataFrame(
                    {
                        "type": ["N/A"],
                        "total_costs": [0],
                        "length": [0],
                        "quantity": [0],
                        "voltage_level": "",
                        "mv_feeder": "",
                    },
                    index=["No reinforced equipment."],
                ),
            ]
        )

    return costs


def line_expansion_costs(edisgo_obj, lines_names=None):
    """
    Returns costs for earthwork and per added cable in kEUR as well as voltage level
    for chosen lines.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        eDisGo object
    lines_names: None or list(str)
        List of names of lines to return cost information for. If None, it is returned
        for all lines in :attr:`~.network.topology.Topology.lines_df`.

    Returns
    -------
    costs: :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with names of lines in index and columns 'costs_earthworks' with
        earthwork costs in kEUR, 'costs_cable' with costs per cable/line in kEUR, and
        'voltage_level' with information on voltage level the line is in.

    """
    if lines_names is None:
        lines_df = edisgo_obj.topology.lines_df.loc[:, ["length"]]
    else:
        lines_df = edisgo_obj.topology.lines_df.loc[lines_names, ["length"]]
    mv_lines = lines_df[
        lines_df.index.isin(edisgo_obj.topology.mv_grid.lines_df.index)
    ].index
    lv_lines = lines_df[~lines_df.index.isin(mv_lines)].index

    # get population density in people/km^2
    # transform area to calculate area in km^2
    projection = proj2equidistant(int(edisgo_obj.config["geo"]["srid"]))
    sqm2sqkm = 1e6
    population_density = edisgo_obj.topology.grid_district["population"] / (
        transform(projection, edisgo_obj.topology.grid_district["geom"]).area / sqm2sqkm
    )
    if population_density <= 500:
        population_density = "rural"
    else:
        population_density = "urban"

    costs_cable_mv = float(edisgo_obj.config["costs_cables"]["mv_cable"])
    costs_cable_lv = float(edisgo_obj.config["costs_cables"]["lv_cable"])
    costs_cable_earthwork_mv = float(
        edisgo_obj.config["costs_cables"][
            f"mv_cable_incl_earthwork_{population_density}"
        ]
    )
    costs_cable_earthwork_lv = float(
        edisgo_obj.config["costs_cables"][
            f"lv_cable_incl_earthwork_{population_density}"
        ]
    )

    costs_lines = pd.DataFrame(
        {
            "costs_earthworks": (costs_cable_earthwork_mv - costs_cable_mv)
            * lines_df.loc[mv_lines].length,
            "costs_cable": costs_cable_mv * lines_df.loc[mv_lines].length,
            "voltage_level": ["mv"] * len(mv_lines),
        },
        index=mv_lines,
    )

    costs_lines = pd.concat(
        [
            costs_lines,
            pd.DataFrame(
                {
                    "costs_earthworks": (costs_cable_earthwork_lv - costs_cable_lv)
                    * lines_df.loc[lv_lines].length,
                    "costs_cable": costs_cable_lv * lines_df.loc[lv_lines].length,
                    "voltage_level": ["lv"] * len(lv_lines),
                },
                index=lv_lines,
            ),
        ]
    )
    return costs_lines.loc[lines_df.index]


def transformer_expansion_costs(edisgo_obj, transformer_names=None):
    """
    Returns costs per transformer in kEUR as well as voltage level they are in.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        eDisGo object
    transformer_names: None or list(str)
        List of names of transformers to return cost information for. If None, it is
        returned for all transformers in
        :attr:`~.network.topology.Topology.transformers_df` and
        :attr:`~.network.topology.Topology.transformers_hvmv_df`.

    Returns
    -------
    costs: :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with names of transformers in index and columns 'costs' with
        costs per transformer in kEUR and 'voltage_level' with information on voltage
        level the transformer is in.

    """
    transformers_df = pd.concat(
        [
            edisgo_obj.topology.transformers_df.copy(),
            edisgo_obj.topology.transformers_hvmv_df.copy(),
        ]
    )
    if transformer_names is not None:
        transformers_df = transformers_df.loc[transformer_names, ["type_info"]]

    if len(transformers_df) == 0:
        return pd.DataFrame(columns=["costs", "voltage_level"])

    hvmv_transformers = transformers_df[
        transformers_df.index.isin(edisgo_obj.topology.transformers_hvmv_df.index)
    ].index
    mvlv_transformers = transformers_df[
        transformers_df.index.isin(edisgo_obj.topology.transformers_df.index)
    ].index

    costs_hvmv = float(edisgo_obj.config["costs_transformers"]["mv"])
    costs_mvlv = float(edisgo_obj.config["costs_transformers"]["lv"])

    costs_df = pd.DataFrame(
        {
            "costs": costs_hvmv,
            "voltage_level": "hv/mv",
        },
        index=hvmv_transformers,
    )
    costs_df = pd.concat(
        [
            costs_df,
            pd.DataFrame(
                {
                    "costs": costs_mvlv,
                    "voltage_level": "mv/lv",
                },
                index=mvlv_transformers,
            ),
        ]
    )
    return costs_df


def grid_expansion_costs_from_diff(edisgo_obj):
    """
    Returns grid expansion costs based on a comparison of the original grid topology
    and the current grid topology.

    The original grid topology is stored in :class:`~.EDisGo`

    Parameters
    ----------

    Returns
    --------
    `pandas.DataFrame<DataFrame>`

    """
    edisgo_grid_orig = edisgo_obj.topology.original_grid_topology

    buses_orig = edisgo_grid_orig.topology.buses_df.copy()
    lines_orig = edisgo_grid_orig.topology.lines_df.copy()

    buses = edisgo_obj.topology.buses_df.copy()
    lines = edisgo_obj.topology.lines_df.copy()

    # get lines that are in grid now but not in original grid - these were either
    # added or changed
    new_or_changed_or_split_lines = [_ for _ in lines.index if
                                     _ not in lines_orig.index]
    # get lines that were in original grid but not in grid now - these were split or
    # connected to other bus (when splitting LV grid lines where feeder is split are
    # renamed) or removed when component was deleted, e.g. generator decommissioned
    removed_lines = [_ for _ in lines_orig.index if _ not in lines.index]
    # get lines that are in both grids to check, whether they changed
    lines_in_both_grids = [_ for _ in lines_orig.index if _ in lines.index]

    # get new buses
    new_buses = [_ for _ in buses.index if _ not in buses_orig.index]
    # get removed buses - exist when generators were decommisioned or lines aggregated
    removed_buses = [_ for _ in buses_orig.index if _ not in buses.index]

    # track lines changes
    lines_changed = pd.DataFrame()

    # check lines in both grids whether they changed - check line type and length and
    # add to lines_changed if either changed
    lines_tmp = lines.loc[lines_in_both_grids, :]
    lines_changed_length = lines_tmp[
        lines_orig.loc[lines_in_both_grids, "length"] != lines.loc[
            lines_in_both_grids, "length"]]
    if not lines_changed_length.empty:
        lines_changed = pd.concat([lines_changed, lines_changed_length], axis=0)
    lines_changed_type = lines_tmp[
        lines_orig.loc[lines_in_both_grids, "type_info"] != lines.loc[
            lines_in_both_grids, "type_info"]]
    if not lines_changed_type.empty:
        lines_changed = pd.concat([lines_changed, lines_changed_type], axis=0)

    # check removed lines
    for removed_line in removed_lines.copy():
        # check whether any of its buses were also removed
        if (lines_orig.at[removed_line, "bus0"] in removed_buses) or (
                lines_orig.at[removed_line, "bus1"] in removed_buses):
            # drop from removed lines
            removed_lines.remove(removed_line)

    # remaining lines in removed_lines list should be lines that were split
    # match each line in removed_lines with lines in new_or_changed_or_split_lines -
    # important because these may not lead to additional costs
    for removed_line in removed_lines:
        # find path between original buses in new grid - all lines on that path
        line_bus0 = lines_orig.at[removed_line, "bus0"]
        line_bus1 = lines_orig.at[removed_line, "bus1"]
        if buses.at[line_bus0, "v_nom"] > 1.0:
            graph = edisgo_obj.topology.mv_grid.graph
        else:
            # check if buses are in same LV grid or not (could happen when grid is
            # split)
            if int(buses.at[line_bus0, "lv_grid_id"]) == int(
                    buses.at[line_bus1, "lv_grid_id"]):
                graph = edisgo_obj.topology.get_lv_grid(
                    int(buses.at[line_bus0, "lv_grid_id"])).graph
            else:
                graph = edisgo_obj.topology.to_graph()
        path = nx.shortest_path(graph, line_bus0, line_bus1)
        # get lines in path
        lines_in_path = lines[lines.bus0.isin(path) & lines.bus1.isin(path)]
        # drop these lines from new_or_changed_or_split_lines
        for l in lines_in_path.index:
            try:
                new_or_changed_or_split_lines.remove(l)
            except:
                logger.debug(f"Line {l} is in path but could not be removed.")
        # check whether line type changed or number of parallel lines and add
        # to lines_changed
        lines_changed_type = lines_in_path[
            lines_in_path.type_info != lines_orig.at[removed_line, "type_info"]]
        if not lines_changed_type.empty:
            # add to lines_changed dataframe
            lines_changed = pd.concat([lines_changed, lines_changed_type], axis=0)
            # drop from lines_in_path
            lines_in_path.drop(index=lines_changed_type.index, inplace=True)
        # for num parallel changes only consider additional line in costs
        lines_changed_num_parallel = lines_in_path[
            lines_in_path.num_parallel != lines_orig.at[removed_line, "num_parallel"]]
        if not lines_changed_num_parallel.empty:
            # reduce num_parallel by number of parallel lines in original grid
            lines_changed_num_parallel["num_parallel"] = lines_changed_num_parallel[
                                                       "num_parallel"] - lines_orig.at[
                                                       removed_line, "num_parallel"]
            lines_changed = pd.concat(
                [lines_changed, lines_changed_num_parallel], axis=0
            )

    # get new buses where new component is directly connected - these are most likely
    # not new branch tees where line was split
    buses_components_orig = pd.concat(
        [edisgo_grid_orig.topology.loads_df.loc[:, ["bus"]],
         edisgo_grid_orig.topology.generators_df.loc[:, ["bus"]],
         edisgo_grid_orig.topology.storage_units_df.loc[:, ["bus"]]]
    )
    buses_components = pd.concat(
        [edisgo_obj.topology.loads_df.loc[:, ["bus"]],
         edisgo_obj.topology.generators_df.loc[:, ["bus"]],
         edisgo_obj.topology.storage_units_df.loc[:, ["bus"]]]
    )
    buses_components_new = list(set([_ for _ in buses_components.bus if
                                     _ not in buses_components_orig.bus and _ in
                                     new_buses]))
    new_or_changed_or_split_lines_df = lines.loc[new_or_changed_or_split_lines, :]
    added_lines = new_or_changed_or_split_lines_df[
        (new_or_changed_or_split_lines_df.bus0.isin(buses_components_new)) | (
            new_or_changed_or_split_lines_df.bus1.isin(buses_components_new))]
    lines_changed = pd.concat([lines_changed, added_lines], axis=0)

    # remove from new_or_changed_or_split_lines
    for l in new_or_changed_or_split_lines_df.index:
        new_or_changed_or_split_lines.remove(l)

    if not len(new_or_changed_or_split_lines) == 0:
        logger.warning(
            f"new_or_changed_or_split_lines is not empty: "
            f"{new_or_changed_or_split_lines}"
        )

    # determine line costs
    lines_changed.drop_duplicates(keep="last", inplace=True, subset=["bus0", "bus1"])
    line_costs = line_expansion_costs(edisgo_obj, lines_names=lines_changed.index)
    costs_df = pd.DataFrame(
        {
            "type": lines_changed.type_info,
            "total_costs": (line_costs.costs_earthworks +
                            line_costs.costs_cable * lines_changed.num_parallel),
            "length": lines_changed.num_parallel * lines_changed.length,
            "quantity": lines_changed.num_parallel,
            "voltage_level": line_costs.voltage_level,
        },
        index=lines_changed.index
    )

    # add costs for transformers
    transformers_orig = pd.concat(
        [edisgo_grid_orig.topology.transformers_df.copy(),
         edisgo_grid_orig.topology.transformers_hvmv_df.copy()]
    )
    transformers = pd.concat(
        [edisgo_obj.topology.transformers_df.copy(),
         edisgo_obj.topology.transformers_hvmv_df.copy()]
    )
    new_transformers = [_ for _ in transformers.index if
                        _ not in transformers_orig.index]
    transformers_in_both_grids = [_ for _ in transformers_orig.index if
                                  _ in transformers.index]
    transformers_changed = transformers.loc[new_transformers, :]
    # check transformers in both grids whether they changed - check type_info
    # and add to transformers_changed if type changed
    transformers_tmp = transformers.loc[transformers_in_both_grids, :]
    transformers_changed_type = transformers_tmp[
        transformers_orig.loc[transformers_in_both_grids, "type_info"] !=
        transformers.loc[
            transformers_in_both_grids, "type_info"]]
    if not transformers_changed_type.empty:
        transformers_changed = pd.concat(
            [transformers_changed, transformers_changed_type], axis=0)
    transformer_costs = transformer_expansion_costs(
        edisgo_obj, transformers_changed.index
    )
    transformer_costs_df = pd.DataFrame(
        {
            "type": transformers_changed.type_info,
            "total_costs": transformer_costs.costs,
            "length": 0.0,
            "quantity": 1,
            "voltage_level": transformer_costs.voltage_level,
        },
        index=transformers_changed.index
    )
    costs_df = pd.concat([costs_df, transformer_costs_df])

    return lines_changed, transformers_changed, costs_df
