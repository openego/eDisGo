.. _storage-flex:

Battery storage
===============

In plain terms
--------------

A battery can absorb power when there is too much (e.g. midday PV feed-in) and return
it when it is needed, smoothing the load the grid sees. eDisGo supports both a simple
self-consumption reference operation and full optimisation, and storage can be added
manually anywhere in the grid (see :ref:`components-guide`).

Data
----

Storage units live in ``topology.storage_units_df`` with the key attributes
``p_nom`` (nominal power), ``max_hours`` (energy capacity / ``p_nom``),
``efficiency_store`` / ``efficiency_dispatch`` (charge/discharge efficiencies) and,
for home batteries, the associated ``building_id``. Home-battery potentials can be
imported from the OEP with :meth:`~edisgo.edisgo.EDisGo.import_home_batteries`.

Reference operation
-------------------

:py:func:`~edisgo.flex_opt.battery_storage_operation.apply_reference_operation`
implements a **self-consumption** strategy for home batteries: charge when local
generation exceeds demand, discharge when demand exceeds generation, exchanging power
with the grid only when the battery is full or empty.

The state of energy evolves as

.. math::

   \text{SOE}(t+1) = \text{SOE}(t)
   + \Big(P_\text{in}(t)\,\eta_\text{store}
   - \frac{P_\text{out}(t)}{\eta_\text{dispatch}}\Big)\,\Delta t ,
   \qquad 0 \le \text{SOE}(t) \le p_\text{nom}\cdot \text{max\_hours},

where the efficiencies account for conversion losses on charging
(:math:`\eta_\text{store}`) and discharging (:math:`\eta_\text{dispatch}`).

Optimised operation
-------------------

Storage units passed to :meth:`~edisgo.edisgo.EDisGo.pm_optimize` via
``flexible_storage_units`` are dispatched by the OPF to minimise grid expansion,
subject to their state-of-energy dynamics and capacity limits. (The OPF models the
battery losses with a resistance-based model rather than the
``efficiency_store``/``efficiency_dispatch`` factors of the reference operation —
see :ref:`flexibility-opf`.) Storage units **not** marked flexible keep the
self-consumption reference operation.

.. note::

   Storage time series are not generated automatically. For a time-series study you
   must either provide a series (e.g. via
   :meth:`~edisgo.edisgo.EDisGo.add_component`), let the OPF schedule the unit, or use
   a worst-case analysis.

A heuristic that automatically *sizes and places* storage to relieve the grid exists
but is currently unmaintained — see :ref:`legacy`.
