.. _workflow:
.. _usage-details:

The eDisGo workflow
===================

eDisGo is a toolbox, so there is no single "run" function — you assemble a study
from building blocks. Those blocks must be called in a particular order, because
later steps consume data produced by earlier ones. This page gives the canonical
order, the rules that must be respected, and the errors you get when they are not.

Canonical call order
---------------------

A full study typically follows these stages (steps 2, 5 and 6 are optional
depending on the scenario):

#. **Load the grid** — :class:`~edisgo.edisgo.EDisGo` from a ding0 grid, or
   :func:`~edisgo.edisgo.import_edisgo_from_files` to reload a saved object.
#. **Import the generator park** —
   :meth:`~edisgo.edisgo.EDisGo.import_generators` (to adopt a future-scenario
   generator park; skip if you want to keep the status-quo park that every ding0 grid
   already ships with).
#. **Set the time index** — pass ``timeindex`` to the constructor or call
   :meth:`~edisgo.edisgo.EDisGo.set_timeindex`. Required before any time-series
   import.
#. **Import flexible assets** —
   :meth:`~edisgo.edisgo.EDisGo.import_home_batteries`,
   :meth:`~edisgo.edisgo.EDisGo.import_heat_pumps`,
   :meth:`~edisgo.edisgo.EDisGo.import_dsm`,
   :meth:`~edisgo.edisgo.EDisGo.import_electromobility`.
#. **Set active power time series** —
   :meth:`~edisgo.edisgo.EDisGo.set_time_series_active_power_predefined` (or
   ``worst_case`` / manual), then apply operation strategies
   (:meth:`~edisgo.edisgo.EDisGo.apply_charging_strategy`,
   :meth:`~edisgo.edisgo.EDisGo.apply_heat_pump_operating_strategy`).
#. **Set reactive power time series** —
   :meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control`. **Always
   last** of the time-series steps.
#. **(Optional) optimise flexibilities** —
   :meth:`~edisgo.edisgo.EDisGo.pm_optimize` (see :ref:`flexibility-opf`).
#. **Analyse** — :meth:`~edisgo.edisgo.EDisGo.analyze` (power flow) and/or
#. **Reinforce** — :meth:`~edisgo.edisgo.EDisGo.reinforce`.
#. **Inspect & save** — read :class:`~edisgo.network.results.Results`, plot, and
   :meth:`~edisgo.edisgo.EDisGo.save`.

The :doc:`../tutorials/full_workflow_walkthrough` notebook implements exactly this
sequence, stage by stage.

At any point before analysing you can call
:meth:`~edisgo.edisgo.EDisGo.check_integrity` to validate that the time series,
components and flexibility data are mutually consistent — a quick way to catch the
ordering mistakes described below.

.. _ordering-rules:

Critical ordering rules
-----------------------

.. warning::

   * **Time index first.** Set the time index *before* importing any
     ``oedb``/predefined time series — otherwise the imported series cannot be
     aligned.
   * **All component imports before time series.** Import generators and flexible
     assets before setting their time series.
   * **Active power before reactive power.** Reactive-power control derives ``Q``
     from the active-power series.
   * **Reactive power last.** Call
     :meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control` only after
     *all* active-power series and components are in place.
   * **Import electromobility before the charging strategy.**
     :meth:`~edisgo.edisgo.EDisGo.apply_charging_strategy` operates on the charging
     parks from :meth:`~edisgo.edisgo.EDisGo.import_electromobility` (it writes the
     charging-point series itself); run the import first, or it has nothing to do.

Common errors and how to fix them
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Cause / fix
   * - Reactive power is empty / zero after
       :meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control`
     - Reactive-power control was called before the active-power series existed (``Q``
       is derived from ``P``). Set active power first; call reactive power last.
   * - ``KeyError`` in :meth:`~edisgo.edisgo.EDisGo.pm_optimize` (missing ``q_set``)
     - Reactive power was not set. Run
       :meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control` after all
       active-power steps.
   * - :meth:`~edisgo.edisgo.EDisGo.import_heat_pumps` imports the wrong technologies
     - By default **all** of ``individual_heat_pumps``, ``central_heat_pumps`` and
       ``central_resistive_heaters`` are imported; pass ``import_types`` to restrict to
       a subset.
   * - :meth:`~edisgo.edisgo.EDisGo.apply_charging_strategy` does nothing
     - No charging parks present; run
       :meth:`~edisgo.edisgo.EDisGo.import_electromobility` first.
   * - Storage unit has no time series
     - Storage series are not auto-generated; set them manually (e.g. assign a
       DataFrame to ``edisgo.timeseries.storage_units_active_power``) or use a
       worst-case analysis.
   * - Power flow does not converge
     - Extreme voltages/loads. Use the ``troubleshooting_mode`` argument of
       :meth:`~edisgo.edisgo.EDisGo.analyze` (``"lpf"`` or ``"iteration"``), or let
       :meth:`~edisgo.edisgo.EDisGo.reinforce` handle it via
       ``catch_convergence_problems=True``.
   * - ``MemoryError`` on large grids / many time steps
     - Reduce the problem with
       :meth:`~edisgo.edisgo.EDisGo.spatial_complexity_reduction`,
       :meth:`~edisgo.edisgo.EDisGo.reduce_memory`, or a reduced/temporal analysis.

For the meaning of each step, continue with :ref:`data-model`, :ref:`time-series`
and :ref:`analysis-results`; for the engineering behind analysis and reinforcement
see :ref:`methodology`.
