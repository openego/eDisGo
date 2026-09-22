.. _overlying-grid-flex:

Overlying-grid requirements
===========================

In plain terms
--------------

A distribution grid does not exist in isolation: it hangs off the high-voltage
(transmission) grid above it. When the whole system is planned together — for example
eDisGo coupled with `eTraGo <https://github.com/openego/eTraGo>`_ via
`eGo <https://github.com/openego/eGo>`_ — the upper grid level may *require* the
distribution grid to dispatch its flexibilities in a certain way (e.g. a given
curtailment, a storage or DSM profile). eDisGo stores these requirements and can pass
them to the optimisation as additional constraints.

Data
----

The :class:`~edisgo.network.overlying_grid.OverlyingGrid` container
(``edisgo.overlying_grid``) holds the requirements handed down from the overlying
grid. It has thirteen attributes; :ref:`overlying-grid-mapping` below lists every one
of them together with where it ends up in eDisGo.

The helper
:py:func:`~edisgo.network.overlying_grid.distribute_overlying_grid_requirements`
returns a new EDisGo object in which the aggregate requirements are distributed onto
the individual components, with a distribution key that differs per flexibility (EV by
flexibility-band power, storage by installed capacity ``p_nom``, power-to-heat by
``p_set``, DSM by its potential bands, and curtailment proportional to current
feed-in). Note that this helper is **not** part of the production pipeline — it is
used only inside
:mod:`~edisgo.tools.temporal_complexity_reduction` to rank critical time steps on a
throwaway copy. In a production run the requirements reach the results through the
optimisation, as described below.

.. _overlying-grid-mapping:

Expected data mapping (eTraGo → eDisGo)
---------------------------------------

This section states, per overlying-grid attribute, **which eDisGo variable carries it
into the optimisation, where the corresponding result appears, and what relation
between the two is expected to hold**. It is meant as the reference for automated
validation tests.

Everything in this section applies to ``opf_version`` 3 and 4 only. Under versions 1
and 2 no high-voltage requirement constraint is built at all, and the only
overlying-grid attributes with any effect are the two generator time series in
:ref:`overlying-grid-mapping-generators`. Versions 3 and 4 therefore require
``import_overlying_grid_data`` to have run. Without overlying-grid data they degrade
to version 2 and the run carries none of the constraints described here, so a
validation test should assert on ``pm["opf_version"]`` (or on the presence of
:attr:`~edisgo.opf.results.opf_result_class.OPFResults.hv_requirement_slacks_t`)
rather than assume the requested version was the one solved.

.. _overlying-grid-mapping-constraints:

Group 1 — attributes that become high-voltage requirement constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~edisgo.io.powermodels_io.to_powermodels` collects these into
``hv_flex_dict`` and :func:`~edisgo.io.powermodels_io._build_hv_requirements` turns
each entry into one PowerModels ``HV_requirements`` component. On the Julia side each
becomes a single equality constraint per time step

.. math::

   \sum_{c \in \mathcal{F}} p_{c,t} \;+\; p^{\mathrm{hvs}}_t \;=\; P_t

where :math:`\mathcal{F}` is the set of *flexible* components addressed by that
requirement and :math:`p^{\mathrm{hvs}}` is a slack that the objective penalises
quadratically (with a reduced weight for DSM). The requirement is therefore a
**soft equality, not a bound**: over- and under-fulfilment are both penalised, and a
non-zero slack means the grid could not deliver the requested dispatch.

.. list-table::
   :header-rows: 1
   :widths: 26 10 30 34

   * - Overlying-grid attribute
     - Key
     - Requirement :math:`P_t` fed to the OPF
     - Where the eDisGo result appears
   * - ``renewables_curtailment``
       (Series, MW)
     - ``curt``
     - the series itself
     - :attr:`~edisgo.network.timeseries.TimeSeries.generators_active_power` of the
       non-dispatchable generators, *reduced* by the optimised curtailment
       :math:`p_{gc}`
   * - ``storage_units_active_power``
       (Series, MW)
     - ``storage``
     - the series minus the ``p_set`` sum of all storage units **not** in
       ``flexible_storage_units``
     - :attr:`~edisgo.network.timeseries.TimeSeries.storage_units_active_power`
       of the flexible storage units (taken from the virtual branch, so charging and
       discharging losses are included)
   * - ``electromobility_active_power``
       (Series, MW)
     - ``cp``
     - the series minus the ``p_set`` sum of the inflexible charging points,
       clipped at 0
     - :attr:`~edisgo.network.timeseries.TimeSeries.loads_active_power` of the
       loads with ``type == "charging_point"``
   * - ``heat_pump_decentral_active_power``
       **plus**
       ``heat_pump_central_active_power``
       (Series, MW)
     - ``hp``
     - the **sum of the two series** minus the ``p_set`` sum of the inflexible heat
       pumps, clipped at 0
     - :attr:`~edisgo.network.timeseries.TimeSeries.loads_active_power` of the
       loads with ``type == "heat_pump"``
   * - ``dsm_active_power``
       (Series, MW)
     - ``dsm``
     - the series minus the ``p_set`` sum of the inflexible DSM loads
     - :attr:`~edisgo.network.timeseries.TimeSeries.loads_active_power` of the DSM
       loads, as *base load plus* the optimised DSM shift :math:`p_{dsm}`

Constraints a validation test may rely on:

* **One aggregate for both heat-pump attributes.** There is no separate central and
  decentral heat-pump requirement. ``heat_pump_central_active_power`` and
  ``heat_pump_decentral_active_power`` are added together into a single ``hp``
  requirement, so the invariant is

  .. math::

     \sum_{h \in \mathrm{flexible\ HPs}} p_{h,t}
     \;=\; \max\!\left(0,\;
     P^{\mathrm{hp,central}}_t + P^{\mathrm{hp,decentral}}_t
     - \!\!\sum_{h \in \mathrm{inflexible\ HPs}}\!\! p^{\mathrm{set}}_{h,t}
     \right)

  up to the slack. With the default runner configuration every heat pump is
  flexible, so this collapses to ``sum of all heat-pump load time series ==
  heat_pump_central_active_power + heat_pump_decentral_active_power``.

  Either way, comparing the central series against the ``district_heating`` loads
  alone will not match, and neither will comparing it against the
  ``district_heating`` plus ``district_heating_resistive_heater`` loads, since the
  decentral units are part of the same aggregate.
* **Inflexible units are subtracted from the requirement, not from the result.** The
  requirement handed to the OPF is what the *flexible* components must deliver. Which
  components those are comes from the ``flexible:`` parameter of the ``optimize``
  step, or, if that is absent, from the top-level ``flexibilities:`` list, whose
  carrier names are mapped onto the OPF-level ones (``electromobility`` becomes
  ``charging_points``, ``home_batteries`` becomes ``storage``). All shipped presets
  end up selecting every component of each of the four types, so the inflexible sets
  are empty and the subtraction is a no-op; a test that restricts the flexible sets
  must account for it.
* **``cp`` and ``hp`` are clipped at 0, ``storage`` and ``dsm`` are not.** If the
  inflexible units alone already exceed the requirement, the requirement for the
  flexible ones becomes zero rather than negative.
* **The requirement is an equality with a penalised slack.** A perfect match is only
  expected when the slack is zero. The realised slacks are stored in
  :attr:`~edisgo.opf.results.opf_result_class.OPFResults.hv_requirement_slacks_t`.
  Because the constraint is :math:`\sum p_{c,t} + p^{\mathrm{hvs}}_t = P_t`, the
  unmet share of a requirement is :math:`|p^{\mathrm{hvs}}_t| / P_t` — **that slack
  frame is the anchor for an automated check**.

  Do *not* use the summary frame ``opf_results.overlying_grid`` for this. Its columns
  are labelled "Highest / Mean / Sum relative error", but the quantity is computed as
  :math:`|p^{\mathrm{hvs}}_t - P_t|`, which by the constraint above equals the
  *achieved* dispatch, not the error. The 5 % warning eDisGo logs off that frame is
  inverted in the same way: it stays quiet exactly when the requirement is missed
  completely. See `openego/eDisGo#755
  <https://github.com/openego/eDisGo/issues/755>`_.
* **Units.** All overlying-grid power series are in MW; the OPF works in per unit and
  divides by ``s_base``. The values written back to the time series are in MW again.

Group 2 — attributes that become storage boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These three do not produce a requirement constraint. They fix the state of charge at
the beginning and at the end of the optimised horizon (``soc_start`` / ``soc_end``),
which is why they carry one time step more than the other series.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Overlying-grid attribute
     - Applies to
     - Where the eDisGo result appears
   * - ``storage_units_soc``
       (Series, p.u.)
     - every flexible storage unit (the same series is used for all of them)
     - :attr:`~edisgo.opf.results.opf_result_class.OPFResults.battery_storage_t`
   * - ``thermal_storage_units_decentral_soc``
       (Series, p.u.)
     - the flexible heat pumps with ``sector == "individual_heating"``
     - :attr:`~edisgo.opf.results.opf_result_class.OPFResults.heat_storage_t`
   * - ``thermal_storage_units_central_soc``
       (DataFrame, p.u., one column per district-heating area)
     - the remaining flexible heat pumps, matched by their
       ``district_heating_id``
     - :attr:`~edisgo.opf.results.opf_result_class.OPFResults.heat_storage_t`

Constraints a validation test may rely on:

* **p.u. of the storage capacity.** The series are multiplied by ``p_nom *
  max_hours`` (battery) resp. the thermal storage ``capacity`` to obtain MWh.
* **Column naming of ``thermal_storage_units_central_soc``.** Its columns have to be
  the district-heating ID as the *string of an integer* (``"130"``, not ``"130.0"``
  and not ``130``), because
  :func:`~edisgo.io.powermodels_io._build_heat_storage` looks them up as
  ``loads_df.district_heating_id.astype(int).astype(str)``. The
  ``import_overlying_grid_data`` task normalises the labels of this frame and of
  ``feedin_district_heating`` on import, so a float label coming from eTraGo is
  repaired for the runner path. An object built by hand or restored by
  :meth:`~edisgo.network.overlying_grid.OverlyingGrid.from_csv` outside the runner
  still has to obey the convention, or the lookup raises a ``KeyError``.
* **Only start and end are binding.** The values in between are scaffolding; the OPF
  chooses the trajectory. Comparing the full input SoC series against the result is
  not a meaningful check.

.. _overlying-grid-mapping-generators:

Group 3 — attributes applied outside the optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These three bypass the OPF entirely; a pipeline task applies them. The two generator
series are set by ``import_overlying_grid_data`` via
:meth:`~edisgo.edisgo.EDisGo.set_time_series_active_power_predefined`, the
district-heating feed-in by ``aggregate_district_heating``. They therefore take effect
under **every** ``opf_version``, and also without any OPF at all — but only if the
pipeline contains the respective task.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Overlying-grid attribute
     - Where the eDisGo result appears
   * - ``dispatchable_generators_active_power``
       (DataFrame, MW, one column per technology)
     - :attr:`~edisgo.network.timeseries.TimeSeries.generators_active_power` of the
       dispatchable generators — set directly, not optimised
   * - ``renewables_potential``
       (Series, p.u.)
     - :attr:`~edisgo.network.timeseries.TimeSeries.generators_active_power` of the
       fluctuating generators, before any curtailment from the ``curt`` requirement
   * - ``feedin_district_heating``
       (DataFrame, MW, one column per district-heating area)
     - not a result of its own: it is subtracted from the heat demand of its
       district-heating area in
       :attr:`~edisgo.network.heat.HeatPump.heat_demand_df`, bounded at zero, before
       the power-to-heat units of that area are merged into one component. It
       therefore lowers the electricity the heat pumps draw

Constraint a validation test may rely on: for dispatchable generators the eDisGo time
series should reproduce the eTraGo input exactly (up to the technology-to-generator
distribution), because nothing downstream modifies it. For fluctuating generators the
result equals the potential **minus** the curtailment from Group 1.

Group 4 — attributes that currently reach nothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the thirteen attributes has no consumer in the production pipeline. A
validation test must not expect it to show up anywhere.

* ``dispatchable_generators_reactive_power`` (DataFrame, Mvar) — has no consumer
  anywhere in eDisGo. It is read from CSV / accepted from eTraGo, stored, saved back
  out, and otherwise ignored. Reactive power of dispatchable generators is instead
  derived from the configured power factor by
  :meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control`.

Known limitation: two attributes never arrive over the eTraGo path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``eGo`` hands the overlying-grid data to
:func:`~edisgo.run.run_edisgo` as a dict whose keys are matched one-to-one against
the attribute names above. Two of its keys do not match:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Key produced by eGo
     - Attribute eDisGo expects
   * - ``thermal_storage_central_soc``
     - ``thermal_storage_units_central_soc``
   * - ``thermal_storage_decentral_soc``
     - ``thermal_storage_units_decentral_soc``

Both are therefore silently dropped, and the same applies to the CSV path, whose
files are named after the eGo keys. Nothing warns about it — an unrecognised name is
discarded without a message, see `openego/eDisGo#758
<https://github.com/openego/eDisGo/issues/758>`_. In an eGo run the two thermal-storage
state-of-charge attributes of Group 2 are consequently always empty, and the OPF
falls back to a zero state of charge. The remaining eleven attributes match by name
and do arrive. A validation test should treat the two as absent until the naming is
reconciled.

Known limitation: temporal reduction with several intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the time index is a reduced, non-contiguous selection (as produced by
``select_critical_timesteps``),
:func:`~edisgo.opf.powermodels_opf.pm_optimize` solves one OPF per contiguous
interval. The mapping above holds per interval. Any comparison between an
overlying-grid input and an eDisGo result must therefore be evaluated per interval and
not across the whole reduced index.

Known limitation: open defects that break the balances above
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mapping above describes the intended behaviour. Three known defects keep some of
the balances from closing, and a validation test written today will fail on them for
reasons that have nothing to do with the grid or the scenario:

* **The requirement handed to the solver can be silently reduced.** For ``cp`` and
  ``hp`` the contribution of the inflexible units is subtracted and the remainder is
  clipped at zero. Where the inflexible demand alone already exceeds the requirement,
  the discarded surplus appears nowhere — not in the slack, not in a log message, not
  on the eDisGo object — so energy conservation between eTraGo and eDisGo is lost
  without a trace. ``storage`` and ``dsm`` are subtracted without a clip. See
  `openego/eDisGo#756 <https://github.com/openego/eDisGo/issues/756>`_.
* **The DSM shift is not kept in the results.** ``pdsm`` is written into
  :attr:`~edisgo.network.timeseries.TimeSeries.loads_active_power`, but
  :class:`~edisgo.opf.results.opf_result_class.OPFResults` has no DSM container, so
  the shift cannot be recovered after the run. A DSM check has to snapshot the load
  time series *before* :meth:`~edisgo.edisgo.EDisGo.pm_optimize` and difference
  against it. See `openego/eDisGo#757
  <https://github.com/openego/eDisGo/issues/757>`_.
* **Battery power may be compared on the wrong side of the converter.** The storage
  requirement is reported to be matched against battery-side power while eTraGo
  exports the grid-side value; since the ohmic losses are not stored either, the
  storage balance cannot currently be reconstructed from the saved results. See
  `openego/eDisGo#753 <https://github.com/openego/eDisGo/issues/753>`_.

Use in the optimisation
-----------------------

The high-voltage requirements are honoured by the OPF versions that add HV
constraints — ``opf_version`` 3 and 4 of :meth:`~edisgo.edisgo.EDisGo.pm_optimize`
(see :ref:`flexibility-opf`). This keeps the local, distribution-level flexibility
schedule consistent with what the transmission-level planning expects, so the two
levels can be optimised coherently.
