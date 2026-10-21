.. _grid-reinforcement:
.. _grid_expansion_methodology:

Grid reinforcement
==================

In plain terms
--------------

*Reinforcement* (grid expansion) is eDisGo's answer to the question "what does it
cost to make this grid handle the load and generation?". It looks for lines and
transformers that are overloaded and buses whose voltage is out of bounds, then
applies the measures a German distribution grid operator would typically use
(parallel cables, bigger/extra transformers, split feeders) until all problems are
solved — and reports the cost.

How it works
------------

Reinforcement is orchestrated by
:py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid` (exposed as
:meth:`~edisgo.edisgo.EDisGo.reinforce`). It runs the following measures in order:

#. Reinforce stations and lines due to **overloading**.
#. Reinforce **MV lines** due to voltage issues.
#. Reinforce **distribution substations (MV/LV stations)** due to voltage issues.
#. Reinforce **LV lines** due to voltage issues.
#. Reinforce stations and lines due to overloading **again** — the lower impedance
   created by the voltage measures can produce new overloads.

.. figure:: ../images/grid_expansion_measures.png
   :scale: 50%

   Grid reinforcement measures and the order in which issues are identified and
   solved.

Overloading is usually fixed in a single step. Voltage issues can only be solved
**iteratively**: after each measure a power flow is run and the voltages are
re-checked, up to ``max_while_iterations`` times (default 20).

Useful options of :meth:`~edisgo.edisgo.EDisGo.reinforce`:

* ``copy_grid=True`` — compute the needs without changing the grid topology.
* ``mode`` — restrict to ``"mv"``, ``"mvlv"`` or ``"lv"``.
* ``split_voltage_band`` — split the allowed voltage band between the voltage levels
  MV, MV/LV stations and LV (default ``True``; see below).
* ``reduced_analysis`` — only analyse the most critical time steps to save time.
* ``catch_convergence_problems=True`` — fall back to
  :py:func:`~edisgo.flex_opt.reinforce_grid.catch_convergence_reinforce_grid`, which
  handles non-convergence in stages: it first reinforces using only the converging
  time steps, then the initially non-converging ones, and only as a last resort scales
  the time series iteratively (starting from a minimal factor of 0.05 and increasing
  it, for up to 10 iterations).

Enhanced reinforcement
----------------------

On very large or heavily overloaded grids even
``catch_convergence_problems=True`` can fail to find a feasible grid.
:py:func:`~edisgo.flex_opt.reinforce_grid.enhanced_reinforce_grid` is a more robust
wrapper that reinforces **voltage level by voltage level** and falls back to ever
more drastic measures until a grid is found on which the power flow converges:

#. **Separate heavily overloaded LV grids** (if ``separate_lv_grids=True``,
   default). An LV grid whose overloading exceeds ``separation_threshold`` times the
   nominal apparent power of its MV/LV transformer(s) is split by adding a new MV/LV
   station (:py:func:`~edisgo.flex_opt.reinforce_grid.run_separate_lv_grids`).
#. **Reinforce each LV grid on its own.** If a single LV grid does not converge in
   the power flow, it is split first
   (:py:func:`~edisgo.flex_opt.reinforce_measures.separate_lv_grid`) and then
   reinforced.
#. **Reinforce everything at once.** If that fails, it runs, **in sequence** (each
   step is attempted regardless of whether the previous one succeeded), the **MV**
   level only, then **MV + MV/LV stations** (``mode="mvlv"``), then each **LV grid
   separately** (again splitting any that do not converge).
#. **Cost-distorting last resort** (only if
   ``activate_cost_results_disturbing_mode=True``). For LV grids that still cannot be
   solved, first **all lines are replaced by the standard line type**; if that is
   still not enough, **all components of the LV grid are aggregated onto the MV/LV
   station bus**. These measures restore convergence but distort the reported costs:
   the line replacement is recorded in ``equipment_changes`` and *does* enter the cost
   calculation, whereas the aggregation leaves no costed trace, so the total is an
   *underestimate* (a warning is logged and recorded in ``edisgo.results.measures``).
#. A final full reinforcement is run over the now-feasible grid. Note that the
   fallback sequence in step 3, the cost-distorting measures in step 4 and this final
   reinforcement only run if the *"everything at once"* attempt at the start of step 3
   fails; if it converges right away, the function returns at that point.

Use it when a normal :meth:`~edisgo.edisgo.EDisGo.reinforce` raises convergence or
iteration errors; for well-behaved grids the standard reinforcement is sufficient.

.. note::

   The MV line reinforcement that splits a feeder with **voltage issues** at an LV
   station does not yet actually insert a switch disconnector at the split point
   (``TODO`` in :py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_lines_voltage_issues`),
   so the resulting half-rings are not switchable in the model. See
   :ref:`switches-explained` for the role of switch disconnectors in reinforcement.

Identifying problems
--------------------

Constraint checking lives in
:py:mod:`~edisgo.flex_opt.check_tech_constraints`.

**Overloading** is determined from allowed *load factors*
(:ref:`config_grid_expansion`, section ``grid_expansion_load_factors``). These factors
can differ between load and feed-in case (:ref:`load-feedin-case`), but for normal
operation they all default to 1.0; case-dependent factors (e.g. 0.5 for the MV load
case) exist only in the currently unusable n-1 section.

* Lines: :py:func:`~edisgo.flex_opt.check_tech_constraints.mv_line_max_relative_overload`
  and :py:func:`~edisgo.flex_opt.check_tech_constraints.lv_line_max_relative_overload`
  return lines whose relative loading exceeds 1.0. The allowed and relative loads are
  computed by :py:func:`~edisgo.flex_opt.check_tech_constraints.lines_allowed_load`
  and :py:func:`~edisgo.flex_opt.check_tech_constraints.lines_relative_load`. The
  allowed current uses the manufacturer's ``I_max_th`` (tables
  :ref:`lv_cables_table`, :ref:`mv_cables_table`, :ref:`mv_lines_table`).
* Stations:
  :py:func:`~edisgo.flex_opt.check_tech_constraints.hv_mv_station_max_overload` and
  :py:func:`~edisgo.flex_opt.check_tech_constraints.mv_lv_station_max_overload` use
  the transformer rating ``S_nom`` (tables :ref:`lv_transformers_table`,
  :ref:`mv_transformers_table`);
  :py:func:`~edisgo.flex_opt.check_tech_constraints.stations_relative_load` gives the
  relative loading.

**Voltage problems** are determined by
:py:func:`~edisgo.flex_opt.check_tech_constraints.voltage_issues` against the allowed
deviations in :ref:`config_grid_expansion` (section
``grid_expansion_allowed_voltage_deviations``). With ``split_voltage_band=True``
(default) the band is split between MV, MV/LV stations and LV, which get separate
limits — a combined limit can leave almost no room in the LV grids when the MV
deviation is already close to the limit.
:py:func:`~edisgo.flex_opt.check_tech_constraints.voltage_deviation_from_allowed_voltage_limits`
returns the absolute deviations.

Physics
-------

* **Loading / current.** A line's apparent power must stay below
  :math:`S_\text{allowed} = S_\text{nom}\cdot \text{load factor}`, with the current
  :math:`I = S / (\sqrt{3}\,V)`. A transformer's loading uses
  :math:`S = \sqrt{P^2 + Q^2} \le S_\text{nom}\cdot \text{load factor}`.
* **Voltage.** The voltage deviation along a feeder is approximately
  :math:`\Delta V \approx I\,Z` (with line impedance :math:`Z = R + \mathrm{j}X`).
  The configured allowed deviations follow DIN EN 50160 (combined ±10 %) and
  VDE-AR-N 4105 (LV voltage drop); the component-connection limits in
  ``config_grid`` additionally reference VDE-AR-N 4100/4110. Reinforcement reduces
  :math:`\Delta V` either by lowering :math:`Z` (parallel lines, shorter feeders) or
  by adding transformer capacity.

Reinforcement measures
----------------------

Measures are implemented in :py:mod:`~edisgo.flex_opt.reinforce_measures`.

Lines due to overloading
~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_lines_overloading` decides per
line up front, in three cases: lines that are already of standard type get as many
additional parallel standard lines as needed; single-circuit **cables** with a
relative overload below 2 get one parallel line of the **same** type; all other lines
are **replaced** by as many parallel standard lines as needed.

Stations due to overloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_hv_mv_station_overloading`
and :py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_mv_lv_station_overloading`
add a parallel transformer of the existing type (the smallest one that solves the
problem if several exist); otherwise the existing transformers are **replaced** by as
many standard transformers as needed to cover the previous plus the missing capacity.

MV/LV stations due to voltage issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_mv_lv_station_voltage_issues`
installs a parallel standard transformer, re-runs the power flow and repeats until
the voltage is within limits or the iteration limit is reached.

Lines due to voltage issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_lines_voltage_issues`
addresses, per feeder, the critical node **farthest from the station** (by path
length). The treatment differs by voltage level: in **MV** grids the feeder is split
at an LV station (after 2/3 of the path length), or — if there is none to split at —
the node is connected directly to the MV station busbar; in **LV** grids the split
point is the first node at or beyond 2/3 of the path length, moved back to a node
outside a building. If the relevant node is already directly connected to the station,
a parallel standard line is added when the line is already of standard type, otherwise
the line is **replaced** by a standard line. Only one voltage problem per feeder is
treated per iteration, because each measure affects the whole feeder.

Separating overloaded LV grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~edisgo.flex_opt.reinforce_measures.separate_lv_grid` splits a heavily
overloaded LV grid by adding a new MV/LV station, redistributing the load.

.. _grid-expansion-costs:

Grid-expansion costs
--------------------

Costs are computed by :py:func:`~edisgo.flex_opt.costs.grid_expansion_costs` (which
uses :py:func:`~edisgo.flex_opt.costs.line_expansion_costs`;
:py:func:`~edisgo.flex_opt.costs.transformer_expansion_costs` is a standalone helper
not called by it). The total is the sum of the cost of every added transformer and
line. Costs are distinguished only by voltage level, not by equipment type, and for
lines additionally by whether they run in a **rural** (≤ 500 inhabitants/km², lower
earthwork cost) or **urban** area ([DENA]_). Unit costs come from
:ref:`config_grid_expansion`.

References
----------

.. [DENA] A.C. Agricola et al.: *dena-Verteilnetzstudie: Ausbau- und
   Innovationsbedarf der Stromverteilnetze in Deutschland bis 2030*. 2012.
