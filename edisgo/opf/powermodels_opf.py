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

import copy
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from edisgo.flex_opt import exceptions
from edisgo.io.powermodels_io import from_powermodels
from edisgo.network.topology import Topology

logger = logging.getLogger(__name__)

# Time-indexed opf_results attributes that from_powermodels overwrites on each
# call. When the OPF is run separately per interval, these must be concatenated
# across intervals so opf_results covers the full (reduced) time index. Nested
# containers (LineVariables etc.) are listed via their sub-frame attribute names.
_OPF_FLAT_TIME_FRAMES = (
    "slack_generator_t",
    "hv_requirement_slacks_t",
)
_OPF_NESTED_TIME_FRAMES = (
    "lines_t",
    "heat_storage_t",
    "grid_slacks_t",
    "battery_storage_t",
)


def _with_freq(index):
    """Return the DatetimeIndex with its frequency inferred/attached if regular.

    Reducing/uniting time indices drops the ``freq`` attribute; several
    downstream consumers (notably the powermodels OPF) do
    ``timeindex[-1] + timeindex.freq`` and break on ``freq is None``. This
    re-attaches the freq when the index is regularly spaced (a no-op otherwise).
    """
    if index.freq is not None or len(index) < 2:
        return index
    inferred = pd.infer_freq(index)
    if inferred is not None:
        try:
            return pd.DatetimeIndex(index, freq=inferred)
        except (ValueError, TypeError):
            return index
    return index


def _contiguous_intervals(timeindex):
    """
    Split a time index into contiguous intervals.

    Automatic timestep selection can reduce the time index to disconnected
    intervals (e.g. one load-case and one feed-in-case week). This helper detects
    the gap(s) so the OPF can be run separately per interval — storage/heat state
    does not carry across a gap, so a single OPF over the concatenated steps would
    be wrong.

    A boundary is placed wherever the spacing between two consecutive time steps
    exceeds the regular step (the smallest spacing in the index). A contiguous
    index therefore yields a single interval. Each returned interval has its
    ``freq`` restored (set operations that produced the reduced index drop it).

    Parameters
    ----------
    timeindex : pandas.DatetimeIndex

    Returns
    -------
    list of pandas.DatetimeIndex
        One entry per contiguous interval, in chronological order. Empty index
        in -> empty list out; a single time step -> one interval.
    """
    timeindex = timeindex.sort_values()
    if len(timeindex) <= 1:
        return [timeindex] if len(timeindex) else []
    diffs = timeindex[1:] - timeindex[:-1]
    step = diffs.min()
    breaks = [i + 1 for i, d in enumerate(diffs) if d > step]
    starts = [0] + breaks
    ends = breaks + [len(timeindex)]
    return [_with_freq(timeindex[s:e]) for s, e in zip(starts, ends)]


def _snapshot_opf_time_frames(opf_results):
    """Copy the time-indexed opf_results frames produced by one interval's OPF."""
    snap = {}
    for attr in _OPF_FLAT_TIME_FRAMES:
        snap[attr] = getattr(opf_results, attr).copy()
    for attr in _OPF_NESTED_TIME_FRAMES:
        container = getattr(opf_results, attr)
        snap[attr] = {
            sub: getattr(container, sub).copy() for sub in container._attributes()
        }
    return snap


def _merge_opf_time_frames(opf_results, snapshots):
    """
    Concatenate per-interval opf_results snapshots by time index and write them
    back onto ``opf_results``, so its detailed frames cover the full reduced
    index rather than only the last interval's.
    """

    def _concat(frames):
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    for attr in _OPF_FLAT_TIME_FRAMES:
        setattr(opf_results, attr, _concat([s[attr] for s in snapshots]))
    for attr in _OPF_NESTED_TIME_FRAMES:
        container = getattr(opf_results, attr)
        for sub in container._attributes():
            setattr(container, sub, _concat([s[attr][sub] for s in snapshots]))

    # Recompute the overlying_grid summary (opf_version 3/4) from the merged HV
    # requirement slacks, since it is a reduction over the whole time index.
    hv = opf_results.hv_requirement_slacks_t
    if not hv.empty:
        opf_results.overlying_grid = pd.DataFrame(
            columns=["Highest error", "Mean error", "Sum error"],
            index=hv.columns,
            data=pd.concat([hv.max(), hv.mean(), hv.sum()], axis=1).values,
        )


def pm_optimize(
    edisgo_obj,
    s_base: int = 1,
    flexible_cps: np.ndarray | None = None,
    flexible_hps: np.ndarray | None = None,
    flexible_loads: np.ndarray | None = None,
    flexible_storage_units: np.ndarray | None = None,
    opf_version: int = 1,
    method: str = "soc",
    warm_start: bool = False,
    silence_moi: bool = False,
) -> None:
    """
    Run OPF for the edisgo object and write results back to it.

    If the time index is a single contiguous interval, this runs one OPF
    (:func:`_pm_optimize_single`). If the time index is NON-contiguous
    (disconnected intervals, e.g. from automatic timestep selection), each
    contiguous interval is optimized separately and independently — storage/heat
    state does not carry across the gap — and the results are combined:

    * per-interval operation schedules accumulate in ``edisgo.timeseries``;
    * the detailed ``edisgo.opf_results`` frames are merged by time index;
    * a per-interval solve report is stored in
      ``edisgo.opf_results.interval_results``;
    * if any interval was infeasible, an
      :class:`~.flex_opt.exceptions.InfeasibleModelError` is raised after the
      feasible intervals' results have been stored.

    The overlying-grid SOC attributes and reactive-power time series (which
    ``to_powermodels`` / ``from_powermodels`` mutate or replace on the current
    interval) are snapshotted and restored pristine before each interval so a
    later interval sees intact input. Parameters are as for
    :func:`_pm_optimize_single`.
    """
    opf_kwargs = dict(
        s_base=s_base,
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        flexible_storage_units=flexible_storage_units,
        opf_version=opf_version,
        method=method,
        warm_start=warm_start,
        silence_moi=silence_moi,
    )

    intervals = _contiguous_intervals(edisgo_obj.timeseries.timeindex)
    if len(intervals) <= 1:
        # single contiguous optimization. Re-set the (freq-restored) interval so
        # the OPF sees a time index with a frequency — set operations upstream
        # (e.g. timestep selection) drop it, and the OPF needs timeindex.freq.
        if intervals:
            edisgo_obj.set_timeindex(intervals[0])
        _pm_optimize_single(edisgo_obj, **opf_kwargs)
        return

    logger.info(
        f"pm_optimize: time index has {len(intervals)} disconnected intervals; "
        f"running a separate OPF per interval."
    )
    full_timeindex = edisgo_obj.timeseries.timeindex

    # Snapshot the shared input state that per-interval OPF runs mutate:
    #  * overlying-grid SOC attributes are rewritten in place by to_powermodels;
    #  * the reactive-power time series are fully REPLACED (not .loc-updated) by
    #    the set_time_series_reactive_power_control() call inside from_powermodels.
    # Reactive power was set on the full reduced index before this call; restore
    # this input pristine before each interval. Active-power frames are NOT
    # restored — they accumulate each interval's OPF results via .loc.
    og = edisgo_obj.overlying_grid
    og_snapshot = {attr: copy.deepcopy(getattr(og, attr)) for attr in og._attributes}
    reactive_attrs = [
        "_generators_reactive_power",
        "_loads_reactive_power",
        "_storage_units_reactive_power",
    ]
    reactive_snapshot = {
        attr: copy.deepcopy(getattr(edisgo_obj.timeseries, attr, None))
        for attr in reactive_attrs
    }

    def _restore_pristine_inputs():
        for attr, value in og_snapshot.items():
            setattr(og, attr, copy.deepcopy(value))
        for attr, value in reactive_snapshot.items():
            if value is not None:
                setattr(edisgo_obj.timeseries, attr, copy.deepcopy(value))

    # Pre-allocate the storage active-power schedule over the FULL reduced index
    # so from_powermodels .loc-accumulates each interval's storage result instead
    # of replacing the frame with an interval-only one (which would drop earlier
    # intervals' storage schedules).
    su_names = edisgo_obj.topology.storage_units_df.index
    if len(su_names) > 0 and edisgo_obj.timeseries.storage_units_active_power.empty:
        edisgo_obj.timeseries.storage_units_active_power = pd.DataFrame(
            0.0, index=full_timeindex, columns=su_names
        )

    snapshots = []
    report = []
    try:
        for interval in intervals:
            _restore_pristine_inputs()
            edisgo_obj.set_timeindex(interval)
            entry = {
                "start": interval[0],
                "end": interval[-1],
                "status": None,
                "solver": None,
                "solution_time": None,
            }
            try:
                _pm_optimize_single(edisgo_obj, **opf_kwargs)
                entry["status"] = edisgo_obj.opf_results.status
                entry["solver"] = edisgo_obj.opf_results.solver
                entry["solution_time"] = edisgo_obj.opf_results.solution_time
                snapshots.append(_snapshot_opf_time_frames(edisgo_obj.opf_results))
            except exceptions.InfeasibleModelError:
                entry["status"] = "infeasible"
                logger.warning(
                    f"pm_optimize: OPF infeasible for interval "
                    f"{interval[0]}..{interval[-1]}."
                )
            report.append(entry)
    finally:
        # restore the full (reduced) index so all intervals' schedules are exposed
        # and undo the per-interval mutations of the overlying-grid/reactive input.
        _restore_pristine_inputs()
        edisgo_obj.set_timeindex(full_timeindex)

    _merge_opf_time_frames(edisgo_obj.opf_results, snapshots)
    edisgo_obj.opf_results.interval_results = report
    solution_times = [
        e["solution_time"] for e in report if e["solution_time"] is not None
    ]
    edisgo_obj.opf_results.solution_time = (
        sum(solution_times) if solution_times else None
    )
    statuses = [e["status"] for e in report]
    infeasible = [e for e in report if e["status"] == "infeasible"]
    edisgo_obj.opf_results.status = (
        "infeasible" if infeasible else (statuses[0] if statuses else None)
    )
    if infeasible:
        raise exceptions.InfeasibleModelError(
            f"OPF infeasible for {len(infeasible)} of {len(intervals)} time "
            f"intervals; see edisgo.opf_results.interval_results. Results for "
            f"feasible intervals have been stored."
        )


def _pm_optimize_single(
    edisgo_obj,
    s_base: int = 1,
    flexible_cps: np.ndarray | None = None,
    flexible_hps: np.ndarray | None = None,
    flexible_loads: np.ndarray | None = None,
    flexible_storage_units: np.ndarray | None = None,
    opf_version: int = 1,
    method: str = "soc",
    warm_start: bool = False,
    silence_moi: bool = False,
) -> None:
    """
    Run a single-interval OPF for the edisgo object in a julia subprocess and
    write results back to the edisgo object. Assumes the time index is a single
    contiguous interval; :func:`pm_optimize` is the public entry point that
    handles non-contiguous indices by calling this per interval.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 1 MVA.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all charging points that allow for flexible charging.
        Default: None.
    flexible_hps : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
        Default: None.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all flexible loads that allow for application of demand side
        management strategy.
        Default: None.
    flexible_storage_units : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all flexible storage units. Non-flexible storage units operate
        to optimize self consumption.
        Default: None
    opf_version : int
        Version of optimization models to choose from. The grid model is a radial branch
        flow model (BFM). Optimization versions differ in lifted or additional
        constraints and the objective function.
        Implemented versions are:

        * 1
            * Lifted constraints: grid restrictions
            * Objective: minimize line losses and maximal line loading
        * 2
            * Objective: minimize line losses and grid related slacks
        * 3
            * Additional constraints: high voltage requirements
            * Lifted constraints: grid restrictions
            * Objective: minimize line losses, maximal line loading and HV slacks
        * 4
            * Additional constraints: high voltage requirements
            * Objective: minimize line losses, HV slacks and grid related slacks

        Must be one of [1, 2, 3, 4].
        Default: 1.
    method : str
        Optimization method to use. Must be either "soc" (Second Order Cone) or "nc"
        (Non Convex).
        If method is "soc", OPF is run in PowerModels with Gurobi solver with SOC
        relaxation of equality constraint P²+Q² = V²*I². If method is "nc", OPF is run
        with Ipopt solver as a non-convex problem due to quadratic equality constraint
        P²+Q² = V²*I².
        Default: "soc".
    warm_start : bool
        If set to True and if method is set to "soc", non-convex IPOPT OPF will be run
        additionally and will be warm started with Gurobi SOC solution. Warm-start will
        only be run if results for Gurobi's SOC relaxation is exact.
        Default: False.
    silence_moi : bool
        If set to True, MathOptInterface's optimizer attribute "MOI.Silent" is set
        to True in julia subprocess. This attribute is for silencing the output of
        an optimizer. When set to True, it requires the solver to produce no output,
        hence there will be no logging coming from julia subprocess in python
        process.
        Default: False.
    save_heat_storage : bool
        Indicates whether to save results of heat storage variables from the
        optimization to eDisGo object.
        Default: True.
    save_slack_gen : bool
        Indicates whether to save results of slack generator variables from the
        optimization to eDisGo object.
        Default: True.
    save_slacks : bool
        Indicates whether to save results of slack variables of OPF. Depending on
        chosen opf_version, different slacks are used. For more information see
        :func:`edisgo.io.powermodels_io.from_powermodels`.
        Default: True.

    """
    Topology.find_meshes(edisgo_obj)
    opf_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(opf_dir, "opf_solutions")
    pm, hv_flex_dict = edisgo_obj.to_powermodels(
        s_base=s_base,
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        flexible_storage_units=flexible_storage_units,
        opf_version=opf_version,
    )

    def _convert(o):
        """Helper function for json dump, as int64 cannot be dumped."""
        for f in [np.int8, np.int16, np.int32, np.int64]:
            if isinstance(o, f):
                return int(o)
        raise TypeError

    json_str = json.dumps(pm, default=_convert)

    logger.info("starting julia process")
    julia_process = subprocess.Popen(
        [
            "julia",
            os.path.join(opf_dir, "eDisGo_OPF.jl/Main.jl"),
            pm["name"],
            solution_dir,
            method,
            str(silence_moi),
            str(warm_start),
        ],
        stdin=subprocess.PIPE,
        text=True,
        stdout=subprocess.PIPE,
    )
    julia_process.stdin.write(json_str)
    julia_process.stdin.close()
    while True:
        out = julia_process.stdout.readline()
        if out == "" and julia_process.poll() is not None:
            if julia_process.poll() == 0:
                logger.info("Julia process was successful.")
            else:
                raise exceptions.InfeasibleModelError("Julia process failed!")
            break
        if out.rstrip().startswith('{"name"'):
            pm_opf = json.loads(out)
            # write results to edisgo object
            from_powermodels(
                edisgo_obj,
                pm_results=pm_opf,
                hv_flex_dict=hv_flex_dict,
                s_base=s_base,
            )
        elif out.rstrip().startswith("Set parameter") or out.rstrip().startswith(
            "Academic"
        ):
            continue
        elif out != "":
            sys.stdout.write(out)
            sys.stdout.flush()
