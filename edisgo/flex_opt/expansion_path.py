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
    :attr:`~.edisgo.EDisGo.reinforce`. Changes made in a stage are kept for all
    following stages, so the stage functions describe the growth between two
    stages, not the whole scenario.

    The costs of a stage only comprise the measures taken in this stage. A line
    that gets another parallel line in a later stage is therefore charged with
    earthworks again, as the cable has to be laid in a new trench. The costs are
    discounted to `base_year` with `discount_rate`.

    `edisgo_obj` is changed in place: after the function, it contains the grid
    reinforced for the last stage, with the equipment changes of all stages in
    :attr:`~.network.results.Results.equipment_changes`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        Grid at the start of the expansion path.
    stages : dict(int, callable)
        Years of the stages and, for each year, a function that takes the
        :class:`~.EDisGo` object and applies the changes up to this year to it.
        The function has to leave the object with time series for all components,
        e.g. by calling
        :attr:`~.edisgo.EDisGo.set_time_series_worst_case_analysis` after adding
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
        reinforcement per stage. Default: False.
    reinforce_kwargs : dict or None
        Parameters passed to :attr:`~.edisgo.EDisGo.reinforce` in every stage.
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
    without_generator_import = reinforce_kwargs.get("without_generator_import", False)

    original_edisgo = copy.deepcopy(edisgo_obj) if greenfield else None

    stage_costs = []
    stage_changes = []
    for year in years:
        n_changes_before = len(edisgo_obj.results.equipment_changes)
        stages[year](edisgo_obj)
        logger.info(f"Reinforce grid for stage {year}.")
        edisgo_obj.reinforce(**reinforce_kwargs)

        changes = edisgo_obj.results.equipment_changes.iloc[n_changes_before:]
        costs = _costs_of_changes(edisgo_obj, changes, without_generator_import)
        stage_changes.append(changes.assign(year=year))
        stage_costs.append(costs.assign(year=year))

    costs_per_stage = pd.concat(stage_costs)
    summary = _summary(costs_per_stage, years, base_year, discount_rate)

    if greenfield:
        summary["greenfield_costs"] = [
            _greenfield_costs(
                original_edisgo,
                [stages[year] for year in years[: i + 1]],
                reinforce_kwargs,
            )
            for i in range(len(years))
        ]

    return ExpansionPathResults(
        summary=summary,
        grid_expansion_costs=costs_per_stage,
        equipment_changes=pd.concat(stage_changes),
    )


def _costs_of_changes(edisgo_obj, changes, without_generator_import):
    """
    Returns the grid expansion costs of the given equipment changes only.

    The costs are calculated with
    :func:`~.flex_opt.costs.grid_expansion_costs` on the current topology, as if
    `changes` were all equipment changes made so far.

    """
    edisgo_view = copy.copy(edisgo_obj)
    edisgo_view.results = copy.copy(edisgo_obj.results)
    edisgo_view.results.equipment_changes = changes
    costs = grid_expansion_costs(
        edisgo_view, without_generator_import=without_generator_import
    )
    if costs.empty:
        costs = pd.DataFrame(
            columns=["type", "total_costs", "quantity", "length", "voltage_level"]
        )
    return costs


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


def _greenfield_costs(original_edisgo, stage_functions, reinforce_kwargs):
    """
    Returns the costs of reinforcing the original grid for all given stages at once.

    """
    edisgo_obj = copy.deepcopy(original_edisgo)
    n_changes_before = len(edisgo_obj.results.equipment_changes)
    for apply_stage in stage_functions:
        apply_stage(edisgo_obj)
    edisgo_obj.reinforce(**reinforce_kwargs)
    changes = edisgo_obj.results.equipment_changes.iloc[n_changes_before:]
    costs = _costs_of_changes(
        edisgo_obj,
        changes,
        reinforce_kwargs.get("without_generator_import", False),
    )
    return float(costs.total_costs.sum())
