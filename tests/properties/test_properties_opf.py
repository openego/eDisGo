"""
Property-based tests for the Julia-independent parts of :mod:`edisgo.opf`.

Covered are the splitting of a reduced time index into contiguous OPF intervals,
the merging of per-interval OPF results, the CSV persistence of
:class:`~.opf.results.opf_result_class.OPFResults` and the (currently unused)
time series reduction helpers. Each test states the property it checks.
"""

from __future__ import annotations

import copy
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from edisgo.opf import powermodels_opf, timeseries_reduction
from edisgo.opf.results.opf_result_class import (
    OPFResults,
    _get_matching_dict_of_attributes_and_file_names,
)

from .conftest import examples

FIXTURE_SETTINGS = settings(
    deadline=None,
    max_examples=examples(5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ======================================================================================
# contiguous OPF intervals
# ======================================================================================

#: Blocks of consecutive time steps: (gap before the block in steps, block length).
#: A gap of at least 2 steps leaves at least one time step out.
BLOCKS = st.lists(
    st.tuples(st.integers(2, 30), st.integers(1, 30)), min_size=1, max_size=6
)


def _blocks_to_index(blocks, minutes, start="2011-03-01"):
    step = pd.Timedelta(minutes=minutes)
    stamps, t = [], pd.Timestamp(start)
    for gap, length in blocks:
        t = t + gap * step
        stamps.extend(t + i * step for i in range(length))
        t = stamps[-1]
    return pd.DatetimeIndex(stamps)


@given(blocks=BLOCKS, minutes=st.sampled_from([15, 60]), seed=st.integers(0, 2**16))
@settings(deadline=None, max_examples=examples(60))
def test_contiguous_intervals_partition_the_index(blocks, minutes, seed):
    """
    The intervals partition the (sorted) time index - their concatenation is
    the sorted input, nothing is lost or duplicated, also for shuffled input -
    every interval has the regular step inside and a larger gap to the next.
    """
    index = _blocks_to_index(blocks, minutes)
    # the regular step must occur in the index, otherwise it is not detectable
    assume(any(length > 1 for _, length in blocks))
    shuffled = index[np.random.default_rng(seed).permutation(len(index))]
    shuffled = pd.DatetimeIndex(shuffled, freq=None)
    # two-step intervals are covered by the dedicated (xfail) test below
    assume(all(length != 2 for _, length in blocks))
    intervals = powermodels_opf._contiguous_intervals(shuffled)
    step = pd.Timedelta(minutes=minutes)
    joined = intervals[0].append(intervals[1:]) if len(intervals) > 1 else intervals[0]
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(joined, freq=None), pd.DatetimeIndex(index, freq=None)
    )
    assert len(intervals) == len(blocks)
    for interval, (_, length) in zip(intervals, blocks):
        assert len(interval) == length
        assert (np.diff(interval.asi8) == step.value).all()
    for left, right in zip(intervals[:-1], intervals[1:]):
        assert right[0] - left[-1] > step


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Bug: _with_freq calls pd.infer_freq, which needs at least 3 time steps, "
    "outside its try block, so an interval of exactly two time steps in an index "
    "without freq raises 'Need at least 3 dates to infer frequency'.",
)
def test_contiguous_intervals_with_two_step_interval():
    """
    An interval of two time steps is a valid contiguous interval; it is returned
    with its frequency like every other interval of two or more steps.
    """
    index = pd.DatetimeIndex(
        ["2011-01-01 00:00", "2011-01-01 01:00", "2011-01-01 05:00"]
        + [f"2011-01-01 {h:02d}:00" for h in (6, 7)]
    ).union(pd.DatetimeIndex(["2011-01-01 08:00"]))
    index = pd.DatetimeIndex(index, freq=None)
    intervals = powermodels_opf._contiguous_intervals(index)
    assert [len(i) for i in intervals] == [2, 4]
    assert all(i.freq == pd.Timedelta("1h") for i in intervals)


@given(blocks=BLOCKS, minutes=st.sampled_from([15, 60]))
@settings(deadline=None, max_examples=examples(40))
def test_intervals_of_three_or_more_steps_get_freq(blocks, minutes):
    """
    Every returned interval with at least three time steps carries the regular
    step as ``freq`` (needed by the OPF for ``timeindex[-1] + freq``), with
    unchanged values.
    """
    assume(all(length != 2 for _, length in blocks))
    assume(any(length > 1 for _, length in blocks))
    index = pd.DatetimeIndex(_blocks_to_index(blocks, minutes), freq=None)
    for interval in powermodels_opf._contiguous_intervals(index):
        if len(interval) >= 3:
            assert interval.freq == pd.Timedelta(minutes=minutes)


# ======================================================================================
# merging per-interval OPF results
# ======================================================================================


def _random_frame(index, columns, rng):
    return pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )


@given(blocks=BLOCKS, seed=st.integers(0, 2**16))
@settings(deadline=None, max_examples=examples(30))
def test_merged_opf_frames_are_union_of_intervals(blocks, seed):
    """
    Merging the snapshots of all intervals gives, for every time-indexed result
    frame, exactly the rows of all intervals, sorted and without duplicates,
    independent of the order the intervals were solved in.
    """
    rng = np.random.default_rng(seed)
    index = _blocks_to_index(blocks, 60)
    starts = np.cumsum([0] + [length for _, length in blocks])
    intervals = [index[a:b] for a, b in zip(starts[:-1], starts[1:])]
    snapshots, expected = [], {}
    cols = {"p": ["Line_1", "Line_2"], "e": ["Storage_1"]}
    for interval in intervals:
        res = OPFResults()
        res.slack_generator_t = _random_frame(interval, ["pg", "qg"], rng)
        res.lines_t.p = _random_frame(interval, cols["p"], rng)
        res.battery_storage_t.e = _random_frame(interval, cols["e"], rng)
        snapshots.append(powermodels_opf._snapshot_opf_time_frames(res))
        for key, df in (
            ("slack", res.slack_generator_t),
            ("lines_p", res.lines_t.p),
            ("battery_e", res.battery_storage_t.e),
        ):
            expected.setdefault(key, []).append(df)
    order = rng.permutation(len(snapshots))
    merged = OPFResults()
    powermodels_opf._merge_opf_time_frames(merged, [snapshots[i] for i in order])
    for key, actual in (
        ("slack", merged.slack_generator_t),
        ("lines_p", merged.lines_t.p),
        ("battery_e", merged.battery_storage_t.e),
    ):
        pd.testing.assert_frame_equal(actual, pd.concat(expected[key]), obj=key)
        assert actual.index.is_unique and actual.index.is_monotonic_increasing
    assert merged.lines_t.q.empty and merged.heat_storage_t.p.empty


@pytest.mark.xfail(
    strict=True,
    reason="Bug: for several intervals _merge_opf_time_frames recomputes "
    "opf_results.overlying_grid from the absolute HV slacks with columns 'Highest "
    "error'/'Mean error'/'Sum error', while a single OPF run (from_powermodels) "
    "stores relative errors under 'Highest relative error' etc.",
)
def test_merged_overlying_grid_summary_has_single_run_columns():
    """
    The summary of the HV requirement deviations has the same columns (and
    meaning) whether the OPF ran in one or in several intervals.
    """
    index = pd.date_range("2011-01-01", periods=6, freq="h")
    snapshots = []
    for interval in (index[:3], index[3:]):
        res = OPFResults()
        res.hv_requirement_slacks_t = pd.DataFrame(
            {"dsm": 0.1, "storage": 0.0}, index=interval
        )
        snapshots.append(powermodels_opf._snapshot_opf_time_frames(res))
    merged = OPFResults()
    powermodels_opf._merge_opf_time_frames(merged, snapshots)
    assert list(merged.overlying_grid.columns) == [
        "Highest relative error",
        "Mean relative error",
        "Sum relative error",
    ]


# ======================================================================================
# OPFResults persistence
# ======================================================================================

FRAME = st.integers(0, 4).flatmap(
    lambda ncols: st.integers(1, 12).map(lambda nrows: (nrows, ncols))
)


def _filled_opf_results(shapes, seed):
    """OPFResults with a random frame (hourly DatetimeIndex) for every attribute."""
    rng = np.random.default_rng(seed)
    res = OPFResults()
    res.status, res.solver, res.solution_time = "OPTIMAL", "Gurobi", 1.5
    shapes = iter(shapes)
    for attr, files in _get_matching_dict_of_attributes_and_file_names().items():
        targets = (
            [(res, attr)]
            if isinstance(files, str)
            else [(getattr(res, attr), sub) for sub in files]
        )
        for obj, name in targets:
            nrows, ncols = next(shapes)
            if attr == "overlying_grid":
                index = pd.Index(["curt", "storage", "cp", "hp", "dsm"][:nrows])
                columns = ["Highest relative error", "Mean relative error"][:ncols]
            else:
                index = pd.date_range("2011-01-01", periods=nrows, freq="h")
                columns = [f"Comp_{i}" for i in range(ncols)]
            df = pd.DataFrame(
                rng.normal(scale=10, size=(len(index), len(columns))),
                index=index,
                columns=columns,
            )
            setattr(obj, name, df)
    return res


def _all_frames(res):
    for attr, files in _get_matching_dict_of_attributes_and_file_names().items():
        if isinstance(files, str):
            yield attr, getattr(res, attr)
        else:
            for sub in files:
                yield f"{attr}.{sub}", getattr(getattr(res, attr), sub)


def test_every_result_attribute_has_a_unique_file():
    """
    Every frame attribute of the OPF result containers is persisted, each in a
    file of its own.
    """
    files = _get_matching_dict_of_attributes_and_file_names()
    res = OPFResults()
    names = []
    for attr, value in files.items():
        if isinstance(value, str):
            assert isinstance(getattr(res, attr), pd.DataFrame)
            names.append(value)
        else:
            assert set(value) == set(getattr(res, attr)._attributes()), attr
            names.extend(value.values())
    assert len(names) == len(set(names))


@given(
    shapes=st.lists(FRAME, min_size=17, max_size=17),
    seed=st.integers(0, 2**16),
    archive=st.booleans(),
)
@settings(
    deadline=None,
    max_examples=examples(15),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_opf_results_csv_round_trip(tmp_path_factory, shapes, seed, archive):
    """
    ``from_csv(to_csv(x))`` restores every result frame (values, columns and a
    DatetimeIndex), from a directory and from a zip archive; empty frames stay
    empty. Solver status, solver and solution time are not persisted. CSV
    stores floats with repr precision; the C parser may differ in the last
    bit, hence rtol 1e-15.
    """
    res = _filled_opf_results(shapes, seed)
    directory = tmp_path_factory.mktemp("opf")
    res.to_csv(str(directory / "opf_results"))
    restored = OPFResults()
    if archive:
        path = directory / "edisgo.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for f in os.listdir(directory / "opf_results"):
                zf.write(directory / "opf_results" / f, f"opf_results/{f}")
        restored.from_csv(str(path), from_zip_archive=True)
    else:
        restored.from_csv(str(directory / "opf_results"))
    for (name, expected), (_, actual) in zip(_all_frames(res), _all_frames(restored)):
        if expected.empty:
            assert actual.empty, name
            continue
        pd.testing.assert_frame_equal(
            actual, expected, check_freq=False, rtol=1e-15, obj=name
        )
        if name != "overlying_grid":
            assert isinstance(actual.index, pd.DatetimeIndex), name
    assert restored.status is None and restored.solver is None
    assert restored.solution_time is None


# ======================================================================================
# time series reduction (module is currently not used by eDisGo itself)
# ======================================================================================


@given(
    n=st.integers(5, 40),
    ncols=st.integers(1, 3),
    k=st.integers(1, 5),
    n_keep=st.integers(0, 3),
    seed=st.integers(0, 2**16),
)
@settings(deadline=None, max_examples=examples(25))
def test_linked_steps_map_to_representatives(n, ncols, k, n_keep, seed):
    """
    ``get_linked_steps`` links every time step that is neither a
    representative nor to be kept to a representative: keys and values are
    1-based positions in 1..N, no representative and no kept step is a key,
    at most ``num_steps`` different representatives are used and all steps
    that are neither linked nor kept are representatives (at most
    ``num_steps``; a kept step may also be a representative).
    """
    assume(k < n)
    rng = np.random.default_rng(seed)
    index = pd.date_range("2011-01-01", periods=n, freq="h")
    params = pd.DataFrame(rng.normal(size=(n, ncols)), index=index)
    keep = list(index[rng.choice(n, size=min(n_keep, n), replace=False)])
    linked = timeseries_reduction.get_linked_steps(params, num_steps=k, keep_steps=keep)
    keys, values = set(linked), set(linked.values())
    assert keys | values <= set(range(1, n + 1))
    assert not keys & values
    kept = {index.get_loc(t) + 1 for t in keep}
    assert not keys & kept
    assert len(values) <= k
    # steps that are neither linked nor kept are representatives
    assert len(set(range(1, n + 1)) - keys - kept) <= k


@pytest.fixture(scope="module")
def _analyzed_mv_original(_wc_grid_original):
    edisgo = copy.deepcopy(_wc_grid_original)
    edisgo.analyze(mode="mv")
    return edisgo


def test_scored_critical_loading_is_sorted_overload_sum(_analyzed_mv_original):
    """
    The loading score is, per time step with at least one MV line overload,
    the sum of the relative loads > 1 of all overloaded lines; it is > 1,
    sorted in descending order and indexed by power flow time steps.
    """
    from edisgo.flex_opt import check_tech_constraints

    edisgo = _analyzed_mv_original
    score = timeseries_reduction._scored_critical_loading(edisgo)
    rel = check_tech_constraints.lines_relative_load(
        edisgo, lines=edisgo.topology.mv_grid.lines_df.index
    )
    expected = rel.where(rel > 1).sum(axis=1)
    expected = expected[(rel > 1).any(axis=1)]
    assert len(score) > 0
    pd.testing.assert_series_equal(
        score.sort_index(), expected.sort_index(), check_names=False, check_freq=False
    )
    assert (score > 1).all()
    assert score.is_monotonic_decreasing
    assert score.index.isin(edisgo.results.v_res.index).all()


@pytest.mark.xfail(
    strict=True,
    reason="Bug: get_steps_storage (and get_steps_curtailment, which calls it) "
    "uses Series.append, removed in pandas 2, and fails with AttributeError. "
    "Besides, the windows are built with a fixed hourly pd.date_range around each "
    "critical step and not intersected with the time index, so steps outside the "
    "time index would be returned, unsorted (from a set).",
)
def test_steps_storage_within_time_index(_analyzed_mv_original):
    """
    The time steps selected for a storage optimisation are a sorted, unique
    subset of the time index that contains every critical time step.
    """
    edisgo = copy.deepcopy(_analyzed_mv_original)
    steps = timeseries_reduction.get_steps_storage(edisgo, window=2)
    assert len(steps) > 0
    assert steps.isin(edisgo.timeseries.timeindex).all()
    assert steps.is_unique and steps.is_monotonic_increasing
