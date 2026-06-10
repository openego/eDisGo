.. _electromobility-methodology:

Electromobility
===============

In plain terms
--------------

Electric vehicles are a large but *shiftable* load: a car must be charged before it
leaves, but *when* it charges in between can often be chosen freely. eDisGo models
this and offers both simple rule-based charging strategies and full optimisation.

Data and allocation
-------------------

EV data is held in :class:`~edisgo.network.electromobility.Electromobility`:

* ``charging_processes_df`` — one row per charging event (from
  `SimBEV <https://github.com/rl-institut/simbev>`_): use case (home/work/public/hpc),
  nominal charging power, energy demand, and the park start/end time steps.
* ``potential_charging_parks_gdf`` — candidate charging-point locations (from
  `TracBEV <https://github.com/rl-institut/tracbev>`_), with a weighting factor.
* ``integrated_charging_parks_df`` — the parks that were connected to the grid (also
  appear in ``topology.charging_points_df``).

Allocation of charging demand to charging points is done in
:py:func:`~edisgo.io.electromobility_import.distribute_charging_demand`. **Private**
charging (home/work) gets one charging point per vehicle, selected randomly and
weighted by the TracBEV factor
(:py:func:`~edisgo.io.electromobility_import.distribute_private_charging_demand`).
**Public** charging is allocated per process: an existing point is reused if it is
free and powerful enough, otherwise a new one is chosen the same weighted way
(:py:func:`~edisgo.io.electromobility_import.distribute_public_charging_demand`).

.. _charging-strategies:

Charging strategies (heuristic)
-------------------------------

Rule-based strategies are applied with
:meth:`~edisgo.edisgo.EDisGo.apply_charging_strategy`
(:py:func:`~edisgo.flex_opt.charging_strategies.charging_strategy`). Every strategy
must fully cover each charging requirement, and only *private* processes are
flexibilised (public charging prioritises service):

* ``"dumb"`` — charge at maximum power immediately on arrival. No flexibility; the
  worst case for the grid.
* ``"reduced"`` — *preventive*: charge at the **minimum** power that still fully
  charges the car during its parking time (controlled by
  ``minimum_charging_capacity_factor``), spreading the load out.
* ``"residual"`` — *active*: charge when the residual load in the MV grid is lowest
  (high generation, low consumption); processes with little flexibility get priority.

Flexibility bands (for optimisation)
------------------------------------

For the optimal power flow, each flexible charging point is described by bands
computed with
:meth:`~edisgo.network.electromobility.Electromobility.get_flexibility_bands`:

* ``upper_power`` — the maximum charging power available at each time step (zero when
  the car is not plugged in).
* ``lower_energy`` / ``upper_energy`` — the minimum and maximum *cumulative* energy
  that may have been charged by each time step.

Physics
~~~~~~~

The bands encode two facts. The **power band** follows from the connected charger and
the parking schedule: :math:`0 \le P(t) \le P_\text{max}` only while the car is
parked. The **energy band** guarantees the service: the cumulative charged energy
:math:`E(t)=\sum_{\tau\le t} P(\tau)\,\eta_\text{cp}\,\Delta t` must reach the
required demand by departure and never exceed the battery capacity, i.e.
:math:`E_\text{lower}(t) \le E(t) \le E_\text{upper}(t)`, where
:math:`\eta_\text{cp}` is the charging-point efficiency. Any operation inside these
bands fully serves the user; the OPF picks the grid-friendliest one.

See the :doc:`../../tutorials/electromobility_example` notebook for a worked example,
and :ref:`data-sources` for how to obtain SimBEV/TracBEV data.
