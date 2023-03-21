import os

import numpy as np
import pandas as pd

if "READTHEDOCS" not in os.environ:
    from shapely.ops import transform

from edisgo.tools.geo import proj2equidistant


def grid_expansion_costs(edisgo_obj, without_generator_import=False):
    """
    Calculates topology expansion costs for each reinforced transformer and line
    in kEUR.

    Attributes
    ----------
    edisgo_obj : :class:`~.self.edisgo.EDisGo`
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

        line_length : float
            Length of line or in case of parallel lines all lines in km.

        voltage_level : str {'lv' | 'mv' | 'mv/lv'}
            Specifies voltage level the equipment is in.

        mv_feeder : :class:`~.network.components.Line`
            First line segment of half-ring used to identify in which
            feeder the network expansion was conducted in.

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
                * [float(edisgo_obj.config["costs_transformers"]["mv"])]
            },
            index=hvmv_trafos,
        )
        costs_trafos = pd.concat(
            [
                costs_trafos,
                pd.DataFrame(
                    {
                        "costs_transformers": len(mvlv_trafos)
                        * [float(edisgo_obj.config["costs_transformers"]["lv"])]
                    },
                    index=mvlv_trafos,
                ),
            ]
        )
        return costs_trafos.loc[trafos.index, "costs_transformers"].values

    def _get_line_costs(lines_added):
        costs_lines = line_expansion_costs(edisgo_obj, lines_added.index)
        costs_lines["costs"] = costs_lines.apply(
            lambda x: x.costs_earthworks
            + x.costs_cable * lines_added.loc[x.name, "quantity"],
            axis=1,
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
            equipment_changes.index.isin(edisgo_obj.topology._grids_repr)
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
        costs = pd.concat(
            [
                costs,
                pd.DataFrame(
                    {
                        "type": trafos.type_info.values,
                        "total_costs": _get_transformer_costs(trafos),
                        "quantity": len(trafos) * [1],
                        "voltage_level": len(trafos) * ["mv/lv"],
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
            lines_added.groupby(axis=0, level=0)
            .sum()
            .loc[lines_added_unique, ["quantity"]]
        )
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
                            "total cable length": (
                                lines_added.quantity * lines_added.length
                            ).values,
                            "quantity": lines_added.quantity.values,
                            "voltage_level": line_costs.voltage_level.values,
                        },
                        index=lines_added.index,
                    ),
                ]
            )
        # costs for circuit breakers
        # get changed cbs
        circuit_breakers = equipment_changes.loc[
            equipment_changes.index.isin(edisgo_obj.topology.switches_df.index)
        ]

        cb_changed = circuit_breakers.iloc[
            (
                circuit_breakers.equipment
                == edisgo_obj.topology.switches_df.loc[
                    circuit_breakers.index, "type_info"
                ]
            ).values
        ]["quantity"].to_frame()

        if not cb_changed.empty:
            cb_costs = float(
                edisgo_obj.config["costs_circuit_breakers"][
                    "circuit_breaker_installation_work"
                ]
            )
            costs = pd.concat(
                [
                    costs,
                    pd.DataFrame(
                        {
                            "type": edisgo_obj.topology.switches_df.loc[
                                cb_changed.index, "type_info"
                            ].values,
                            "total_costs": cb_costs,
                            "quantity": cb_changed.quantity.values,
                            "voltage_level": "mv",
                        },
                        index=cb_changed.index,
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


def line_expansion_costs(edisgo_obj, lines_names):
    """
    Returns costs for earthworks and per added cable as well as voltage level
    for chosen lines in edisgo_obj.

    Parameters
    -----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
        eDisGo object of which lines of lines_df are part
    lines_names: list of str
        List of names of evaluated lines

    Returns
    -------
    costs: :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with names of lines as index and entries for
        'costs_earthworks', 'costs_cable', 'voltage_level' for each line

    """

    def cost_cable_types(mode):
        """

        Parameters
        ----------
        mode: mv or lv

        Returns
        -------
        The cost of each line type
        """
        # TODO: rewrite it with pd.merge or pd.concat
        equipment_df = edisgo_obj.topology.lines_df[
            edisgo_obj.topology.lines_df.index.isin(lines_names)
        ]
        costs_cable = []
        if mode == "mv":
            voltage_mv_grid = edisgo_obj.topology.mv_grid.buses_df.v_nom[0]
            mv_cable_df = (
                edisgo_obj.topology.equipment_data[f"{mode}_cables"]
                .loc[
                    edisgo_obj.topology.equipment_data[f"{mode}_cables"].U_n
                    == voltage_mv_grid
                ]
                .loc[:, ["cost"]]
            )
            mv_overhead_lines = (
                edisgo_obj.topology.equipment_data[f"{mode}_overhead_lines"]
                .loc[
                    edisgo_obj.topology.equipment_data[f"{mode}_overhead_lines"].U_n
                    == voltage_mv_grid
                ]
                .loc[:, ["cost"]]
            )
            cost_df = pd.concat([mv_cable_df, mv_overhead_lines])
        else:
            cost_df = edisgo_obj.topology.equipment_data[f"{mode}_cables"].loc[
                :, ["cost"]
            ]

        for equip_name1 in equipment_df.type_info:
            for equip_name2 in cost_df.index:
                if equip_name1 == equip_name2:
                    cost = cost_df.loc[cost_df.index.isin([equip_name1])].cost[0]
                    costs_cable.append(cost)
        return costs_cable

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

    costs_cable_mv = np.array(cost_cable_types("mv"))
    costs_cable_lv = np.array(cost_cable_types("lv"))
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
            "costs_earthworks": (
                [costs_cable_earthwork_mv] * len(costs_cable_mv) - costs_cable_mv
            )
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
                    "costs_earthworks": (
                        [costs_cable_earthwork_lv] * len(costs_cable_lv)
                        - costs_cable_lv
                    )
                    * lines_df.loc[lv_lines].length,
                    "costs_cable": costs_cable_lv * lines_df.loc[lv_lines].length,
                    "voltage_level": ["lv"] * len(lv_lines),
                },
                index=lv_lines,
            ),
        ]
    )
    return costs_lines.loc[lines_df.index]


def cost_breakdown(edisgo_obj, lines_df):
    """

    Parameters
    ----------
    edisgo_obj:    class:`~.edisgo.EDisGo`
        eDisGo object of which lines of lines_df are part
    lines: pandas.core.frame.DataFrame
        the changed lines

    Returns
    -------
    `pandas.DataFrame<DataFrame>`

        Example
                    costs_earthworks 	costs_cable 	voltage_level 	costs
    Line name	12.3840 	2.0160 	lv 	14.40

    """
    # cost-breakdown of changed lines
    # get changed lines

    lines_added = lines_df.iloc[
        (
            lines_df.equipment
            == edisgo_obj.topology.lines_df.loc[lines_df.index, "type_info"]
        ).values
    ]["quantity"].to_frame()
    lines_added_unique = lines_added.index.unique()
    lines_added = (
        lines_added.groupby(axis=0, level=0).sum().loc[lines_added_unique, ["quantity"]]
    )
    lines_added["length"] = edisgo_obj.topology.lines_df.loc[
        lines_added.index, "length"
    ]
    if not lines_added.empty:
        costs_lines = line_expansion_costs(edisgo_obj, lines_added.index)
        costs_lines["costs"] = costs_lines.apply(
            lambda x: x.costs_earthworks
            + x.costs_cable * lines_added.loc[x.name, "quantity"],
            axis=1,
        )
        costs_lines["costs_cable"] = costs_lines.apply(
            lambda x: x.costs_cable * lines_added.loc[x.name, "quantity"],
            axis=1,
        )
    return costs_lines
