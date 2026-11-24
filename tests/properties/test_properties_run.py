"""
Property-based tests for the pipeline runner in :mod:`edisgo.run` (config loader,
registry metadata, tasks and end-to-end chains without database).
"""

from __future__ import annotations

import copy
import json
import os

from pathlib import Path

import pandas as pd
import pytest
import yaml

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
from edisgo.run import run_edisgo
from edisgo.run.config import deep_merge, load_config
from edisgo.run.context import RunContext
from edisgo.run.registry import get_task_meta
from edisgo.run.tasks.analysis import (
    task_analyze,
    task_base_reinforce,
    task_reinforce,
)
from edisgo.run.tasks.flex import task_apply_heat_pump_strategy
from edisgo.run.tasks.grid import task_load_from_base, task_setup_grid
from edisgo.run.tasks.io import task_import_overlying_grid_data, task_save
from edisgo.run.tasks.timeseries import task_reactive_power, task_worst_case_ts

from .conftest import (
    TIMESERIES_ATTRIBUTES,
    assert_topology_equal,
    examples,
    max_abs_diff,
    reinforce_checks_per_worst_case,
)

# --------------------------------------------------------------------------------------
# config loader
# --------------------------------------------------------------------------------------

_keys = st.text(alphabet="abcdefgh_", min_size=1, max_size=4)
_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-1000, 1000),
    st.text(alphabet="xyz0123", max_size=5),
    st.lists(st.integers(0, 9), max_size=3),
)
_nested = st.recursive(
    st.dictionaries(_keys, _leaves, max_size=4),
    lambda children: st.dictionaries(_keys, st.one_of(_leaves, children), max_size=4),
    max_leaves=12,
)


def _leaf_paths(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaf_paths(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _get(d, path):
    for k in path:
        d = d[k]
    return d


class TestConfigLoader:
    @given(base=_nested, override=_nested)
    def test_deep_merge_properties(self, base, override):
        """
        ``deep_merge`` lets every leaf of ``override`` win, keeps leaves only in
        ``base`` unless a parent key is overridden by a non-dict, replaces lists
        wholesale and does not mutate its inputs.
        """
        base_copy, override_copy = copy.deepcopy(base), copy.deepcopy(override)
        merged = deep_merge(base, override)

        assert base == base_copy and override == override_copy
        for path, value in _leaf_paths(override):
            assert _get(merged, path) == value
        for path, value in _leaf_paths(base):
            try:
                override_value = _get(override, path[:-1]) if path[:-1] else override
            except (KeyError, TypeError):
                # an ancestor was replaced by a non-dict override value
                continue
            if isinstance(override_value, dict) and path[-1] not in override_value:
                assert _get(merged, path) == value

    @given(d=_nested)
    def test_deep_merge_identities(self, d):
        """Empty dicts are neutral elements and merging a dict with itself is
        idempotent."""
        assert deep_merge(d, {}) == d
        assert deep_merge({}, d) == d
        assert deep_merge(d, d) == d

    @settings(deadline=None, max_examples=examples(25))
    @given(extra=_nested, tasks=st.lists(st.sampled_from(["a", "b", "c"]), min_size=1))
    def test_yaml_json_and_dict_are_equivalent(self, tmp_path_factory, extra, tasks):
        """
        The same config given as dict, YAML file or JSON file loads to the same
        result, and loading a dict does not mutate it.
        """
        cfg = {**extra, "pipeline": list(tasks)}
        cfg.pop("extends", None)
        cfg.pop("stages", None)
        cfg.pop("external_config", None)
        before = copy.deepcopy(cfg)
        directory = tmp_path_factory.mktemp("cfg")
        (directory / "c.yaml").write_text(yaml.safe_dump(cfg))
        (directory / "c.json").write_text(json.dumps(cfg))

        from_dict = load_config(cfg)
        assert cfg == before
        assert load_config(directory / "c.yaml") == from_dict
        assert load_config(directory / "c.json") == from_dict

    @settings(deadline=None, max_examples=examples(25))
    @given(parent=_nested, child=_nested)
    def test_extends_equals_deep_merge(self, tmp_path_factory, parent, child):
        """``extends`` merges the parent below the child exactly like
        ``deep_merge`` (no ``set_timeindex`` parameters involved)."""
        for d in (parent, child):
            for key in ("extends", "stages", "external_config", "pipeline"):
                d.pop(key, None)
        parent["pipeline"] = ["a"]
        directory = tmp_path_factory.mktemp("ext")
        (directory / "parent.yaml").write_text(yaml.safe_dump(parent))
        child_with_ext = {**child, "extends": "parent.yaml"}
        (directory / "child.yaml").write_text(yaml.safe_dump(child_with_ext))

        assert load_config(directory / "child.yaml") == deep_merge(parent, child)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: cyclic 'extends' is not detected, loading recurses until "
        "RecursionError",
    )
    def test_cyclic_extends_gives_clear_error(self, tmp_path):
        """A cycle in ``extends`` is reported as a configuration error."""
        (tmp_path / "a.yaml").write_text("extends: b.yaml\npipeline: [x]\n")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\npipeline: [y]\n")
        with pytest.raises(ValueError):
            load_config(tmp_path / "a.yaml")


# --------------------------------------------------------------------------------------
# registry metadata
# --------------------------------------------------------------------------------------


def _grid_with_heat_pumps():
    """Test grid with two heat pumps, COP and heat demand, 3 time steps."""
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    ti = pd.date_range("2011-01-01", periods=3, freq="h")
    edisgo.set_timeindex(ti)
    for bus_i in (27, 40):
        edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="individual_heating",
            ts_active_power=pd.Series(0.0, index=ti),
            ts_reactive_power="default",
            bus=edisgo.topology.buses_df.index[bus_i],
            p_set=0.01,
        )
    hps = edisgo.topology.loads_df.index[edisgo.topology.loads_df.type == "heat_pump"]
    edisgo.heat_pump.cop_df = pd.DataFrame(3.0, index=ti, columns=hps)
    edisgo.heat_pump.heat_demand_df = pd.DataFrame(0.006, index=ti, columns=hps)
    return edisgo, RunContext(flags={"has_heat_pumps": True}), {}


def _grid_with_overlying_grid_data():
    """Test grid with eTraGo dispatch for 'gas' generators, 3 time steps."""
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    ti = pd.date_range("2011-01-01", periods=3, freq="h")
    edisgo.set_timeindex(ti)
    ctx = RunContext(
        raw_config={"overlying_grid": {"enabled": True, "source": "etrago"}},
        overlying_grid_data={
            "dispatchable_generators_active_power": pd.DataFrame({"gas": 0.5}, index=ti)
        },
    )
    return edisgo, ctx, {}


def _active_power(edisgo):
    return {
        attr: getattr(edisgo.timeseries, attr).copy()
        for attr in TIMESERIES_ATTRIBUTES
        if attr.endswith("active_power")
    }


@pytest.mark.parametrize(
    "name, task, setup",
    [
        pytest.param(
            "apply_heat_pump_strategy",
            task_apply_heat_pump_strategy,
            _grid_with_heat_pumps,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Bug: task alters loads_active_power but is not registered "
                "with ts_altering=True, so the validator allows it after "
                "reactive_power",
            ),
        ),
        pytest.param(
            "import_overlying_grid_data",
            task_import_overlying_grid_data,
            _grid_with_overlying_grid_data,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Bug: task sets generator active power but is not "
                "registered with ts_altering=True",
            ),
        ),
    ],
)
def test_tasks_altering_active_power_are_declared(name, task, setup, monkeypatch):
    """
    Registry consistency: a task that changes any active power time series must
    be registered with ``ts_altering=True`` - the validator relies on this flag
    to keep such tasks before ``reactive_power``.
    """
    monkeypatch.setattr("edisgo.edisgo.egon_engine", lambda *a, **k: None)
    edisgo, ctx, kwargs = setup()
    before = _active_power(edisgo)
    task(edisgo, ctx, **kwargs)
    after = _active_power(edisgo)
    changed = any(
        not before[attr].equals(after[attr].reindex_like(before[attr]))
        or before[attr].shape != after[attr].shape
        for attr in before
    )
    assert changed, "precondition: the task must change active power here"
    assert get_task_meta(name).ts_altering


# --------------------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Bug: setup_grid(import_generators=True) passes generator_scenario=None "
    "instead of falling back to ctx.scenario (preset flex_opf relies on it)",
)
def test_setup_grid_imports_generators_of_run_scenario(monkeypatch):
    """
    ``setup_grid(import_generators=True)`` without an explicit generator
    scenario imports the generators of the run's scenario, like the stand-alone
    ``import_generators`` task.
    """
    calls = []
    monkeypatch.setattr(
        EDisGo,
        "import_generators",
        lambda self, generator_scenario=None, **kw: calls.append(generator_scenario),
    )
    ctx = RunContext(
        scenario="eGon2035",
        raw_config={
            "grid": {
                "ding0_path": pytest.ding0_test_network_path,
                "legacy_ding0_grids": True,
            }
        },
    )
    monkeypatch.setattr(ctx, "ensure_engine", lambda: None)
    task_setup_grid(None, ctx, import_generators=True)
    assert calls == ["eGon2035"]


def test_analyze_task_equals_facade(analyzed_grid, wc_grid):
    """
    The ``analyze`` task gives the same power flow as ``EDisGo.analyze`` and
    reports the number of non-converged time steps.
    """
    ctx = RunContext()
    task_analyze(wc_grid, ctx)
    assert ctx.flags["not_converged_steps"] == 0
    assert max_abs_diff(wc_grid.results.v_res, analyzed_grid.results.v_res) < 1e-12


@pytest.mark.xfail(
    strict=True,
    reason="Bug: task 'reinforce' with copy_grid=True discards the reinforced copy "
    "and returns the unchanged input (docstring: 'return it as a new instance')",
)
def test_reinforce_task_copy_grid_returns_reinforced_copy(wc_grid):
    """
    With ``copy_grid=True`` the task leaves its input untouched and returns a
    new, reinforced EDisGo object.
    """
    topo_before = copy.deepcopy(wc_grid)
    result = task_reinforce(wc_grid, RunContext(), copy_grid=True)
    assert_topology_equal(wc_grid, topo_before)
    assert result is not wc_grid
    assert not result.results.equipment_changes.empty


@pytest.mark.xfail(
    strict=True,
    reason="Bug: task 'save' builds the flag path with 'directory + str', which "
    "raises TypeError for a pathlib.Path",
)
def test_save_task_accepts_path(wc_grid, tmp_path):
    """The ``save`` task accepts a :class:`pathlib.Path` like ``EDisGo.save``."""
    ctx = RunContext()
    task_save(wc_grid, ctx, directory=tmp_path / "out")
    assert Path(ctx.flags["last_saved"]).is_dir()


# --------------------------------------------------------------------------------------
# end-to-end chains
# --------------------------------------------------------------------------------------


def test_worst_case_preset_equals_facade_chain(reinforced_grid, tmp_path):
    """
    Pipeline and facade are equivalent: the bundled ``worst_case`` preset gives
    the same reinforced topology, equipment changes and costs as calling
    ``set_time_series_worst_case_analysis`` + ``reinforce`` on EDisGo directly;
    the saved result reloads to the returned object and the input config is
    not mutated.
    """
    cfg = {
        "extends": "worst_case",
        "grid": {
            "ding0_path": pytest.ding0_test_network_path,
            "legacy_ding0_grids": True,
        },
        "results": {"directory": str(tmp_path / "run")},
    }
    cfg_before = copy.deepcopy(cfg)
    result = run_edisgo(cfg)
    assert cfg == cfg_before

    assert_topology_equal(result, reinforced_grid)
    pd.testing.assert_frame_equal(
        result.results.equipment_changes, reinforced_grid.results.equipment_changes
    )
    pd.testing.assert_frame_equal(
        result.results.grid_expansion_costs,
        reinforced_grid.results.grid_expansion_costs,
    )
    loaded = import_edisgo_from_files(tmp_path / "run", import_timeseries=True)
    assert_topology_equal(loaded, result, check_dtype=False)


@pytest.fixture(scope="module")
def base_reinforced(tmp_path_factory):
    """
    Run ``base_reinforce`` on a fresh test grid without time index, as in the
    ``flex_opf`` preset. Returns the EDisGo object, the context and the object
    state right before the task.
    """
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    before = copy.deepcopy(edisgo)
    ctx = RunContext(results_dir=tmp_path_factory.mktemp("base"))
    task_base_reinforce(edisgo, ctx)
    return edisgo, ctx, before


class TestBaseReinforce:
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: base_reinforce resets equipment_changes but leaves "
        "grid_expansion_costs of the base reinforcement in place",
    )
    def test_costs_consistent_with_reset_equipment_changes(self, base_reinforced):
        """After the reset, no costs are reported without equipment changes."""
        edisgo, _, _ = base_reinforced
        assert edisgo.results.equipment_changes.empty
        costs = edisgo.results.grid_expansion_costs
        assert costs.empty or costs.total_costs.sum() == 0

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: with an empty prior time index base_reinforce keeps the 1970 "
        "worst-case time index and time series (flex_opf: later imports use it)",
    )
    def test_prior_time_index_is_restored(self, base_reinforced):
        """
        ``base_reinforce`` restores the time index the object had before, so
        later tasks do not see the synthetic worst-case time steps.
        """
        edisgo, _, before = base_reinforced
        pd.testing.assert_index_equal(
            edisgo.timeseries.timeindex, before.timeseries.timeindex
        )

    def test_artifact_matches_reinforced_grid(self, base_reinforced):
        """The saved base artifact reloads to the base-reinforced topology."""
        edisgo, ctx, _ = base_reinforced
        artifact = ctx.flags["base_reinforce_artifact"]
        assert os.path.isfile(artifact)
        loaded = import_edisgo_from_files(artifact, from_zip_archive=True)
        assert_topology_equal(loaded, edisgo, check_dtype=False)

    def test_base_grid_needs_no_further_worst_case_reinforcement(self, base_reinforced):
        """
        Idempotence across persistence: reloading the base artifact and running
        worst case + reactive power + reinforce again (the second phase of a
        two-phase run) adds no equipment and leaves no violation.
        """
        _, ctx, _ = base_reinforced
        edisgo = task_load_from_base(
            None, ctx, path=ctx.flags["base_reinforce_artifact"]
        )
        assert edisgo.results.equipment_changes.empty
        edisgo = task_worst_case_ts(edisgo, ctx)
        edisgo = task_reactive_power(edisgo, ctx)
        edisgo = task_reinforce(edisgo, ctx)
        assert edisgo.results.equipment_changes.empty
        assert reinforce_checks_per_worst_case(edisgo) == {
            "mv": {},
            "mvlv": {},
            "lv": {},
        }


@settings(
    deadline=None,
    max_examples=examples(10),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_load_from_base_resets_only_equipment_changes(
    _reinforced_grid_original, tmp_path_factory, data
):
    """
    ``load_from_base`` restores the saved topology; equipment changes are
    cleared exactly when ``reset_equipment_changes`` is set.
    """
    reset = data.draw(st.booleans())
    archive = data.draw(st.booleans())
    directory = tmp_path_factory.mktemp("lfb") / "grid"
    _reinforced_grid_original.save(directory, archive=archive)
    path = str(directory) + ".zip" if archive else str(directory)
    loaded = task_load_from_base(
        None,
        RunContext(),
        path=path,
        import_results=True,
        reset_equipment_changes=reset,
    )
    assert_topology_equal(loaded, _reinforced_grid_original, check_dtype=False)
    if reset:
        assert loaded.results.equipment_changes.empty
    else:
        pd.testing.assert_frame_equal(
            loaded.results.equipment_changes,
            _reinforced_grid_original.results.equipment_changes,
            check_dtype=False,
        )
