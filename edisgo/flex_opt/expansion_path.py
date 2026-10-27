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

"""
Grid reinforcement along an expansion path of several target years.

Instead of reinforcing the original grid once for a single target year, the grid
is reinforced stage by stage: every stage starts from the grid reinforced in the
previous stage. This shows when which measure is needed, and the costs of every
stage can be discounted to a common base year.
"""

from __future__ import annotations

import copy
import logging

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from edisgo.flex_opt.costs import grid_expansion_costs

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


@dataclass
class ExpansionPathResults:
    """
    Results of :func:`reinforce_expansion_path`.

    Attributes
    ----------
    summary : :pandas:`pandas.DataFrame<DataFrame>`
        Costs per stage in kEUR. Index are the years of the stages. Columns are:

        costs : float
            Costs of the measures taken in this stage.
        costs_mv, costs_lv : float
            Share of `costs` in the MV (including HV/MV transformers) and LV
            (including MV/LV transformers) grid.
        present_value : float
            `costs` discounted to the base year.
        cumulative_costs : float
            Sum of `costs` of this and all previous stages.
        cumulative_present_value : float
            Sum of `present_value` of this and all previous stages.
        greenfield_costs : float
            Only if `greenfield` is True: costs of reinforcing the original grid
            directly for the scenario of this stage, see
            :func:`reinforce_expansion_path`.

    grid_expansion_costs : :pandas:`pandas.DataFrame<DataFrame>`
        Costs per reinforced line and transformer and stage, as in
        :attr:`~.network.results.Results.grid_expansion_costs`, with an
        additional column `year`.
    equipment_changes : :pandas:`pandas.DataFrame<DataFrame>`
        Equipment changes per stage, as in
        :attr:`~.network.results.Results.equipment_changes`, with an additional
        column `year`.

    """

    summary: pd.DataFrame
    grid_expansion_costs: pd.DataFrame
    equipment_changes: pd.DataFrame


def reinforce_expansion_path(
    edisgo_obj: EDisGo,
    stages: dict[int, Callable[[EDisGo], None]],
    base_year: int | None = None,
    discount_rate: float = 0.05,
    greenfield: bool = False,
    reinforce_kwargs: dict | None = None,
) -> ExpansionPathResults:
    """
    Reinforces the grid stage by stage along an expansion path.

    The stages are processed in the order of their years. For each stage, the
    function given for this stage is applied to the grid as reinforced in the
    previous stage, e.g. to add the heat pumps and charging points expected up to
    this year, and then the grid is reinforced with
    :meth:`~.edisgo.EDisGo.reinforce`.

    Applying the functions of all stages up to a year one after another to the
    original grid must give the grid of this year. The functions must be
    deterministic, as `greenfield` calls them again on a copy of the original grid.

    The costs of a stage only comprise the measures taken in this stage. To this
    end, the equipment changes of earlier stages are set aside while a stage is
    reinforced. A line that gets another parallel line in a later stage is
    therefore charged with earthworks again, as the cable has to be laid in a new
    trench, and :meth:`~.edisgo.EDisGo.reinforce` plans with these earthworks as
    well. The costs are discounted to `base_year` with `discount_rate`.

    `edisgo_obj` is changed in place: after the function, it contains the grid
    reinforced for the last stage, with the equipment changes of all stages in
    :attr:`~.network.results.Results.equipment_changes`.
    :attr:`~.network.results.Results.grid_expansion_costs` contains the costs of
    all these changes as calculated by :meth:`~.edisgo.EDisGo.reinforce`, i.e. with
    the earthworks of every line counted once. The costs per stage are in the
    returned :class:`ExpansionPathResults`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        Grid at the start of the expansion path.
    stages : dict(int, callable)
        Years of the stages and, for each year, a function that takes the
        :class:`~.EDisGo` object and applies the changes up to this year to it, see
        above. The function has to leave the object with time series for all
        components, e.g. by calling
        :meth:`~.edisgo.EDisGo.set_time_series_worst_case_analysis` after adding
        components.
    base_year : int or None
        Year the costs are discounted to. Per default, the year of the first stage
        is used. Default: None.
    discount_rate : float
        Discount rate per year used to calculate the present value of the costs,
        e.g. 0.05 for 5 %. Default: 0.05.
    greenfield : bool
        If True, the original grid is additionally reinforced directly for the
        scenario of every stage, i.e. with the functions of this and all previous
        stages applied at once. The costs are given in column `greenfield_costs`
        of the summary. Comparing them with column `cumulative_costs` shows the
        additional costs of building in stages. This needs one additional
        reinforcement per stage after the first. Default: False.
    reinforce_kwargs : dict or None
        Parameters passed to :meth:`~.edisgo.EDisGo.reinforce` in every stage.
        Default: None.

    Returns
    -------
    :class:`ExpansionPathResults`
        Costs and equipment changes per stage.

    """
    if not stages:
        raise ValueError("At least one stage has to be given.")
    for year, apply_stage in stages.items():
        if not callable(apply_stage):
            raise TypeError(f"The stage of year {year} is not callable.")
    years = sorted(stages)
    if base_year is None:
        base_year = years[0]
    if base_year > years[0]:
        raise ValueError(
            f"base_year {base_year} lies after the first stage in {years[0]}."
        )
    reinforce_kwargs = dict(reinforce_kwargs or {})
    if reinforce_kwargs.get("copy_grid", False):
        raise ValueError("copy_grid cannot be used in an expansion path.")

    original_edisgo = copy.deepcopy(edisgo_obj) if greenfield else None

    all_changes = [edisgo_obj.results.equipment_changes]
    stage_costs = []
    stage_changes = []
    for year in years:
        changes, costs = _reinforce_stage(
            edisgo_obj, [stages[year]], year, reinforce_kwargs
        )
        all_changes.append(changes)
        stage_changes.append(changes.assign(year=year))
        stage_costs.append(costs.assign(year=year))

    edisgo_obj.results.equipment_changes = pd.concat(all_changes)
    edisgo_obj.results.grid_expansion_costs = grid_expansion_costs(
        edisgo_obj,
        without_generator_import=reinforce_kwargs.get(
            "without_generator_import", False
        ),
    )

    costs_per_stage = pd.concat(stage_costs)
    summary = _summary(costs_per_stage, years, base_year, discount_rate)

    if greenfield:
        # reinforcing the original grid for the first stage is the first stage
        summary["greenfield_costs"] = [summary.costs.iloc[0]] + [
            _reinforce_stage(
                copy.deepcopy(original_edisgo),
                [stages[year] for year in years[: i + 1]],
                years[i],
                reinforce_kwargs,
            )[1].total_costs.sum()
            for i in range(1, len(years))
        ]

    return ExpansionPathResults(
        summary=summary,
        grid_expansion_costs=costs_per_stage,
        equipment_changes=pd.concat(stage_changes),
    )


def _reinforce_stage(edisgo_obj, stage_functions, year, reinforce_kwargs):
    """
    Applies the stage functions, reinforces the grid and returns the equipment
    changes and grid expansion costs of this stage only.

    Equipment changes made before are removed from `edisgo_obj`, so that
    :meth:`~.edisgo.EDisGo.reinforce` and the costs treat every line as not
    reinforced before. The caller has to keep them if needed.

    """
    edisgo_obj.results.equipment_changes = edisgo_obj.results.equipment_changes.iloc[:0]
    for apply_stage in stage_functions:
        apply_stage(edisgo_obj)
    _check_time_series(edisgo_obj, year)
    logger.info(f"Reinforce grid for stage {year}.")
    edisgo_obj.reinforce(**reinforce_kwargs)
    costs = grid_expansion_costs(
        edisgo_obj,
        without_generator_import=reinforce_kwargs.get(
            "without_generator_import", False
        ),
    )
    # grid_expansion_costs returns a row with zero costs if nothing was changed
    costs = costs.drop(index="No reinforced equipment.", errors="ignore")
    return edisgo_obj.results.equipment_changes, costs


def _check_time_series(edisgo_obj, year):
    """
    Raises a ValueError if a component has no active power time series.

    Without time series, the power flow ignores the component and the stage would
    silently need no reinforcement.

    """
    topology = edisgo_obj.topology
    timeseries = edisgo_obj.timeseries
    missing = [
        name
        for components_df, time_series_df in [
            (topology.generators_df, timeseries.generators_active_power),
            (topology.loads_df, timeseries.loads_active_power),
            (topology.storage_units_df, timeseries.storage_units_active_power),
        ]
        for name in components_df.index.difference(time_series_df.columns)
    ]
    if missing:
        raise ValueError(
            f"After the stage of year {year}, {len(missing)} component(s) have no "
            f"time series, e.g. {missing[:3]}. The stage function has to set time "
            f"series for all components it adds."
        )


def _summary(costs_per_stage, years, base_year, discount_rate):
    """
    Returns the costs per stage, see :class:`ExpansionPathResults`.

    """
    costs = costs_per_stage.astype({"total_costs": float})
    is_mv = costs.voltage_level.isin(["mv", "hv/mv"])
    summary = pd.DataFrame(
        {
            "costs": costs.groupby("year").total_costs.sum(),
            "costs_mv": costs[is_mv].groupby("year").total_costs.sum(),
            "costs_lv": costs[~is_mv].groupby("year").total_costs.sum(),
        },
        index=pd.Index(years, name="year"),
    ).fillna(0.0)
    discount_factor = (1 + discount_rate) ** (summary.index - base_year)
    summary["present_value"] = summary.costs / discount_factor
    summary["cumulative_costs"] = summary.costs.cumsum()
    summary["cumulative_present_value"] = summary.present_value.cumsum()
    return summary
