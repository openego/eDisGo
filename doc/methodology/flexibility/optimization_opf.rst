.. _flexibility-opf:

Multi-period optimal power flow
===============================

In plain terms
--------------

The optimal power flow (OPF) is the engine that schedules all flexibilities at once.
Given the grid, the inflexible loads/generation and the flexibility bands
(:ref:`flexibility-overview`), it finds operation schedules for electric vehicles,
heat pumps, DSM loads and storage that keep voltages and loadings as healthy as
possible — so that little or no grid reinforcement is needed — while respecting every
component's bands. It is a *multi-period* OPF: all time steps are optimised together,
because shifting energy in time only makes sense across time.

How to run it
-------------

.. code-block:: python

    edisgo.pm_optimize(
        flexible_cps=flexible_cps,            # EV charging points
        flexible_hps=flexible_hps,            # heat pumps with thermal storage
        flexible_loads=flexible_loads,        # DSM loads
        flexible_storage_units=flexible_storage,
        opf_version=2,
        method="soc",
    )

:meth:`~edisgo.edisgo.EDisGo.pm_optimize` exports the grid and the flexibility bands,
runs the optimisation in a **Julia** subprocess using
`PowerModels.jl <https://lanl-ansi.github.io/PowerModels.jl/stable/>`_, and writes
the resulting operation schedules back into ``edisgo.timeseries`` (and the heat /
slack results into ``edisgo.results``). The Julia/Gurobi prerequisites are described
in :ref:`opf-requirements`.

The four flexibility arrays select which components may be optimised; any component
not listed keeps its given time series. Non-flexible storage units operate to
optimise self-consumption rather than being scheduled by the OPF.

The grid model: radial branch flow
-----------------------------------

eDisGo's OPF uses a **radial branch flow model (BFM)** — the natural formulation for
the tree-shaped distribution grids ding0 produces.

.. figure:: ../../images/branch_flow_model.png
   :width: 75%
   :align: center

   Radial branch-flow model: for each branch :math:`i \to j` the model couples the
   sending-end power :math:`(P_{ij}, Q_{ij})`, the branch current :math:`I_{ij}` and
   the bus voltages :math:`V_i, V_j`, with a power balance at every bus.

For every branch and bus the model enforces:

* **Power balance** at each bus — injected active and reactive power equals what
  flows out on the branches plus losses.
* **Branch flow / thermal limit** — apparent power on a branch stays within its
  rating, :math:`S = \sqrt{P^2 + Q^2} \le S_\text{lim}`.
* **Voltage limits** — :math:`V_\text{min} \le V \le V_\text{max}` at every bus.
* **Flexibility constraints** — each flexibility's power and energy bands
  (:ref:`flexibility-overview`), e.g. for a store the state of energy must stay
  within :math:`[E_\text{min}, E_\text{max}]` and the charging requirement must be
  met by departure.

The coupling between branch power, current and voltage is the quadratic equality

.. math::

   P^2 + Q^2 = V^2 \cdot I^2 ,

which is what makes an exact AC-OPF non-convex.

Solution method: SOC vs. non-convex
-----------------------------------

The ``method`` argument chooses how that quadratic constraint is handled:

.. figure:: ../../images/soc_relaxation.png
   :width: 60%
   :align: center

   The SOC relaxation replaces the non-convex equality :math:`P^2+Q^2=V^2 I^2` (the
   cone surface) with the convex inequality :math:`P^2+Q^2 \le V^2 I^2` (the filled
   cone). For radial grids the optimum lies on the surface, so the relaxation is
   usually *exact*.

* ``"soc"`` (default) — a **second-order cone relaxation** replaces the equality
  :math:`P^2+Q^2=V^2 I^2` with the convex inequality :math:`P^2+Q^2 \le V^2 I^2`. The
  resulting convex problem is solved with **Gurobi** and is fast and reliable. For
  radial grids the relaxation is usually *exact* (the inequality is tight at the
  optimum), so the solution is also feasible for the original AC problem.
* ``"nc"`` — the **non-convex** problem with the exact equality, solved with the
  **Ipopt** interior-point solver. More accurate in principle but slower and not
  guaranteed to find the global optimum.
* ``warm_start=True`` (with ``method="soc"``) — additionally runs the non-convex
  Ipopt OPF, warm-started from the (exact) Gurobi SOC solution, to polish the result.

Optimisation versions
----------------------

The ``opf_version`` argument selects the objective and which constraints are active.
All versions minimise line losses; they differ in how grid limits and the overlying
(high-voltage) grid are treated:

.. list-table::
   :header-rows: 1
   :widths: 8 40 52

   * - Version
     - Constraints
     - Objective
   * - 1 (default)
     - Grid restrictions **lifted**
     - minimise line losses **and** maximum line loading
   * - 2
     - Standard grid restrictions
     - minimise line losses **and** grid-related slacks
   * - 3
     - HV requirements added; grid restrictions lifted
     - minimise line losses, maximum line loading **and** HV slacks
   * - 4
     - HV requirements added; standard grid restrictions
     - minimise line losses, HV slacks **and** grid-related slacks

"Lifting" the grid restrictions (versions 1 and 3) turns hard voltage/loading limits
into a penalised objective term (maximum loading), which is useful for assessing how
much flexibility *could* help before deciding on reinforcement. Versions 2 and 4 keep
the limits as constraints and penalise the remaining unavoidable violations via
*slack* variables. Versions 3 and 4 additionally honour requirements handed down from
the overlying grid (:ref:`overlying-grid-flex`).

Results and slacks
------------------

Operation schedules are written into ``edisgo.timeseries`` (charging-point, heat-pump,
DSM and storage active power) and can then feed a final
:meth:`~edisgo.edisgo.EDisGo.reinforce`. The ``save_heat_storage``, ``save_slack_gen``
and ``save_slacks`` flags control whether heat-storage states, the slack generator and
the optimisation slack variables are stored as well; the exact slack set depends on
``opf_version`` (see :func:`edisgo.io.powermodels_io.from_powermodels`).

Performance
-----------

Because all time steps are optimised jointly, the problem grows with both grid size
and the number of time steps. Use :ref:`complexity-reduction` (spatial reduction and
the temporal selection in ``edisgo.opf.timeseries_reduction``) and a per-unit base
``s_base`` to keep the problem tractable.

.. seealso::

   :meth:`~edisgo.edisgo.EDisGo.pm_optimize` and
   :func:`~edisgo.opf.powermodels_opf.pm_optimize` for the full parameter reference.
