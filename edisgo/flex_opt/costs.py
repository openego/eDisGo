import os

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
            equipment_changes.index.isin(
                [f"{_}_station" for _ in edisgo_obj.topology._grids_repr]
            )
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
                            "length": (
                                lines_added.quantity * lines_added.length
                            ).values,
                            "quantity": lines_added.quantity.values,
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
            "mv_cable_incl_earthwork_{}".format(population_density)
        ]
    )
    costs_cable_earthwork_lv = float(
        edisgo_obj.config["costs_cables"][
            "lv_cable_incl_earthwork_{}".format(population_density)
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
