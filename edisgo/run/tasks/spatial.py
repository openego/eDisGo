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
Spatial complexity reduction tasks bracketing ``optimize``.

* :func:`task_spatial_reduce` (``spatial_reduce``) — stashes a deepcopy of
  the full grid on ``ctx`` and spatially reduces the working object so
  ``optimize`` runs on a smaller grid.
* :func:`task_spatial_restore` (``spatial_restore``) — writes the optimized
  flexible-component dispatch back onto the stashed full grid, carries the
  OPF results over to it, and makes it the active object again, so
  ``reinforce`` runs on the full topology.

Both are no-ops when ``spatial_reduction.enabled`` is false (the default),
so a pipeline that carries this bracket behaves exactly like one that
doesn't when spatial reduction is turned off.
"""

from __future__ import annotations

import copy

from edisgo.run.registry import register_task


@register_task("spatial_reduce", requires={"grid"}, provides={"reduced_grid"})
def task_spatial_reduce(edisgo, ctx, **overrides):
    """
    Deepcopy and stash the full grid, then spatially reduce the working
    object.

    Configuration is read from the top-level ``spatial_reduction:`` config
    block (so eGo can inject it the same way it injects
    ``timeseries_selection``); inline step params override individual keys
    of that block.

    A no-op when ``enabled`` is not true — ``edisgo`` is returned unchanged
    and ``ctx.full_grid_stash`` is left ``None``, so a downstream
    ``spatial_restore`` also no-ops (see its docstring) and ``optimize``/
    ``reinforce`` run on the same, unreduced grid as if this task were
    absent from the pipeline.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to spatially reduce in place.
    ctx : RunContext
        Run context. Reads ``ctx.raw_config['spatial_reduction']``. Sets
        ``ctx.full_grid_stash`` to the pre-reduction deepcopy.
    **overrides
        Inline step params overriding keys of the ``spatial_reduction``
        block. Recognized keys: ``enabled`` (bool, default ``False``),
        ``mode``, ``cluster_area``, ``reduction_factor``,
        ``reduction_factor_not_focused``, ``aggregation_mode``, and the
        aggregation sub-modes ``load_aggregation_mode`` /
        ``generator_aggregation_mode`` — forwarded to
        :meth:`~.EDisGo.spatial_complexity_reduction`.

    Returns
    -------
    edisgo.EDisGo
        The (possibly) spatially-reduced EDisGo instance.

    """
    cfg = {**ctx.raw_config.get("spatial_reduction", {}), **overrides}
    if not cfg.get("enabled", False):
        return edisgo

    ctx.full_grid_stash = copy.deepcopy(edisgo)

    kwargs = {
        k: v
        for k, v in cfg.items()
        if k
        in (
            "mode",
            "cluster_area",
            "reduction_factor",
            "reduction_factor_not_focused",
            "apply_pseudo_coordinates",
            "aggregation_mode",
            "load_aggregation_mode",
            "generator_aggregation_mode",
            "line_naming_convention",
            "mv_pseudo_coordinates",
        )
    }
    edisgo.spatial_complexity_reduction(copy_edisgo=False, **kwargs)
    return edisgo


@register_task("spatial_restore", requires={"reduced_grid", "optimized_dispatch"})
def task_spatial_restore(edisgo, ctx, **overrides):
    """
    Write optimized flexible-component dispatch back onto the stashed full
    grid, and make it the active object again.

    Reads the flexible-component name lists ``optimize`` wrote to
    ``ctx.flags`` and passes them, together with ``edisgo`` (the reduced,
    just-optimized grid) and ``ctx.full_grid_stash`` (the pre-reduction
    grid), to :meth:`~.EDisGo.map_reduced_results_to_full_grid`. See that
    method (and the core function it wraps,
    :func:`~.tools.spatial_complexity_reduction.apply_reduced_results_to_full_grid`)
    for the matching/disaggregation rules.

    :attr:`~.EDisGo.opf_results` is additionally carried over from the
    reduced grid, since ``map_reduced_results_to_full_grid`` transfers only
    dispatch time series. Without it the returned object would keep the
    empty :class:`~.opf.results.opf_result_class.OPFResults` it was
    deepcopied with before the solve, and ``save`` would write an empty
    ``opf_results/`` directory.

    The results are attached **verbatim**, so note what their indices refer
    to. ``hv_requirement_slacks_t`` and ``overlying_grid`` are indexed by
    flexibility category (``curt``, ``storage``, ``cp``, ``hp``, ``dsm``)
    and are therefore topology-independent — these are what the
    overlying-grid conservation checks read. ``battery_storage_t`` is
    indexed by storage-unit name, which also carries over, as
    :func:`~.tools.spatial_complexity_reduction.spatial_complexity_reduction`
    never aggregates storage units (it only relabels their buses). The
    remaining frames — ``lines_t``, ``slack_generator_t``,
    ``heat_storage_t`` and ``grid_slacks_t`` — are indexed by *reduced-grid*
    component names, which under ``aggregation_mode=True`` do not match the
    full grid's ``topology``. They are kept rather than dropped (having them
    under reduced names is strictly more information than not having them),
    but they cannot be joined against the full grid's components without
    first mapping the names back.

    A no-op when ``ctx.full_grid_stash`` is ``None`` — i.e. when
    ``spatial_reduce`` did not run or ran disabled — so ``edisgo`` (the
    grid ``optimize`` already ran on, which already holds its own
    ``opf_results``) is returned unchanged.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        The reduced EDisGo instance ``optimize`` ran on.
    ctx : RunContext
        Run context. Reads ``ctx.full_grid_stash`` and the
        ``flexible_cps`` / ``flexible_hps`` / ``flexible_loads`` /
        ``flexible_storage_units`` flags ``optimize`` set. Clears
        ``ctx.full_grid_stash`` back to ``None`` after restoring.
    **overrides
        Unused; accepted for signature consistency with other tasks.

    Returns
    -------
    edisgo.EDisGo
        The full-grid EDisGo instance with flexible dispatch and
        ``opf_results`` restored, or ``edisgo`` unchanged if there is no
        stash to restore from.

    """
    full_grid = ctx.full_grid_stash
    if full_grid is None:
        return edisgo

    full_grid.map_reduced_results_to_full_grid(
        reduced_grid=edisgo,
        flexible_cps=ctx.flags.get("flexible_cps"),
        flexible_hps=ctx.flags.get("flexible_hps"),
        flexible_loads=ctx.flags.get("flexible_loads"),
        flexible_storage_units=ctx.flags.get("flexible_storage_units"),
    )

    # Carry the OPF results over to the object the pipeline continues with.
    # map_reduced_results_to_full_grid only transfers dispatch time series, so
    # without this the full grid would keep the pristine, empty OPFResults it
    # was deepcopied with before the solve, and `save` would write an empty
    # opf_results/ directory. Plain rebinding, not a copy: the reduced grid is
    # dropped right after this returns and nothing mutates opf_results again.
    full_grid.opf_results = edisgo.opf_results

    ctx.full_grid_stash = None
    return full_grid
