.. _dsm-flex:

Demand side management
======================

In plain terms
--------------

Demand side management (DSM) is the flexibility of (mostly industrial and commercial)
processes that can shift their consumption in time — running a cold store, a
compressor or a water treatment a bit earlier or later — without changing the total
energy used. eDisGo represents this potential as power and energy bands that the
optimisation can exploit.

Data
----

DSM potential is held in :class:`~edisgo.network.dsm.DSM`, with four time series per
DSM load. The naming is easiest to read as *deviations from the baseline load*:

* ``p_min`` — the maximum **load decrease** (how far consumption may be reduced) at
  each time step.
* ``p_max`` — the maximum **load increase** at each time step.
* ``e_min`` — the maximum **preponing** (energy shifted *earlier*), cumulative.
* ``e_max`` — the maximum **postponing** (energy shifted *later*), cumulative.

These are imported per industrial/CTS load from the OEP with
:meth:`~edisgo.edisgo.EDisGo.import_dsm` (see :ref:`data-sources`).

Physics / optimisation
----------------------

In :meth:`~edisgo.edisgo.EDisGo.pm_optimize` (loads listed in ``flexible_loads``) the
DSM bands enter as constraints around the baseline demand :math:`P_0(t)`:

.. math::

   P_\text{min}(t) \;\le\; \Delta P(t) \;\le\; P_\text{max}(t),
   \qquad
   E_\text{min}(t) \;\le\; \sum_{\tau\le t}\Delta P(\tau)\,\Delta t \;\le\; E_\text{max}(t),

i.e. the instantaneous shift is bounded by the power band and the *accumulated* shift
by the energy band. The energy band ensures the process still does the same total
work — energy is only moved in time, never created or destroyed. The OPF uses this
freedom to flatten grid-critical peaks.
