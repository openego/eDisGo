.. _time-series:

Time series
===========

Before a power flow can be run, every load, generator and storage unit needs an
active- and reactive-power time series. eDisGo offers several ways to set them.

.. note::

   Set the **time index** before setting any time series — either via the
   ``timeindex`` argument of the :class:`~edisgo.edisgo.EDisGo` constructor or with
   :meth:`~edisgo.edisgo.EDisGo.set_timeindex`. And remember the
   :ref:`ordering rules <ordering-rules>`: reactive power is always set last.

Active power
------------

There are five ways to set active-power series:

Manual
~~~~~~

Provide your own series with :meth:`~edisgo.edisgo.EDisGo.set_time_series_manual`.
Pass a pandas ``DataFrame`` per component category, indexed by the time index and
with one column per component:

.. code-block:: python

    import pandas as pd

    edisgo.set_time_series_manual(
        loads_p=pd.DataFrame(
            {"Load_1": [0.5, 0.6]}, index=edisgo.timeseries.timeindex
        ),
    )

The arguments are ``generators_p``, ``loads_p``, ``storage_units_p`` for active power
and ``generators_q``, ``loads_q``, ``storage_units_q`` for reactive power. Values are
in MW (active) and Mvar (reactive); the ``DataFrame`` index must contain the time
index set on the object.

.. _worst-case-ts:

Worst-case
~~~~~~~~~~

Set feed-in and load for the two classic grid-planning situations — heavy load
(``load_case``) and reverse power flow (``feed-in_case``) — using simultaneity
factors from the configuration files. This sets **both active and reactive power**
(reactive power with fixed cos φ from the config), so no separate reactive-power step
is needed afterwards.

.. code-block:: python

    edisgo.set_time_series_worst_case_analysis()

A fictitious time index starting 1970-01-01 00:00 is set automatically (PyPSA needs
a time index). Each case is set up once for the MV and once for the LV level, so there
are four time steps in total; ``edisgo.timeseries.timeindex_worst_cases`` tells you
which time step maps to which case. Note that this **overwrites** any previously set
(non-worst-case) time series. The definition of load and feed-in case is explained in
:ref:`load-feedin-case`.

**Simultaneity factors**

Conventional loads and generators use fixed scale factors from the config file
:ref:`config_timeseries` in section ``worst_case_scale_factor``.

For **charging points (CP)** in the load case, the simultaneity factor is determined
dynamically based on the number of charging points *n* and their nominal charging power.
Three power classes are supported: 3.7 kW (slow AC), 11 kW (fast AC) and 22 kW (fast AC).
Support points are taken from VDE FNN (2021): *Ermittlung von Gleichzeitigkeitsfaktoren
fuer Ladevorgaenge an privaten Ladepunkten* (suburban residential evening scenario,
conservative worst case). Interpolation is performed in log-space over *n*; for *n > 150*
the factor is clamped to the value at *n = 150*. In the feed-in case the CP factor is
always 0.0 (no reverse power flow from charging assumed), consistent with the dena study
*Integrierte Energiewende* (p. 90).

For **heat pumps (HP)** in the load case, the simultaneity factor depends only on the
number of heat pumps *n*. Support points are derived from the Kerber simultaneity
function *g(n) = g_inf + (1 − g_inf) · n^(−3/4)* (Kerber 2011, eq. 3.2) with an
asymptote *g_inf = 0.9* based on the Consentec/E.ON distribution network study (peak
load cold winter day). The same log-space interpolation and clamping scheme as for
charging points is applied. In the feed-in case the HP factor is always 0.0.

Support point values for both CP and HP are defined in the config file
:ref:`config_timeseries` in section ``simultaneity_curves``.

Predefined
~~~~~~~~~~

Set series by component type, either from your own data or from public sources. Each
component group has its own argument; a call with **no arguments sets no time series
at all**, so you have to specify the desired source explicitly, e.g.:

.. code-block:: python

    edisgo.set_time_series_active_power_predefined(
        fluctuating_generators_ts="oedb",
        conventional_loads_ts="demandlib",
    )

* **Fluctuating generators** (``fluctuating_generators_ts``) — wind and solar feed-in
  from the ``oedb`` (OpenEnergy DataBase), or your own normalised profiles per
  technology (optionally per technology and weather cell) passed as a ``DataFrame``.
* **Dispatchable generators** (``dispatchable_generators_ts``) — your own normalised
  profiles per technology, where a column ``"other"`` acts as catch-all for all
  technologies not listed explicitly.
* **Conventional loads** (``conventional_loads_ts``) — standard load profiles per
  sector via ``demandlib``, scenario-specific per-building profiles from the ``oedb``
  (requires an ``engine`` and a ``scenario``), or your own sector profiles as a
  ``DataFrame`` (see :ref:`data-sources`).
* **Charging points** (``charging_points_ts``) — pass normalised profiles *per use
  case* (there is no ``oedb`` source for charging points, and only the use cases you
  supply are set). If you omit it, charging points — including ``public`` ones — are
  left **without a series**. To set them automatically instead, use
  :meth:`~edisgo.edisgo.EDisGo.apply_charging_strategy` (see :ref:`charging-strategies`),
  which covers all charging points (``public``/``hpc`` charged "dumb").

See :meth:`~edisgo.edisgo.EDisGo.set_time_series_active_power_predefined` for all
options. If no time index has been set, a default full-year index is set automatically.

Optimised
~~~~~~~~~

Optimise the operation of flexibilities (EV charging, heat pumps with thermal
storage, DSM, storage) with a multi-period optimal power flow, so that grid
expansion is minimised. This is :meth:`~edisgo.edisgo.EDisGo.pm_optimize`; it is
documented in detail in :ref:`flexibility-opf`.

Heuristic
~~~~~~~~~

Apply rule-based operation strategies. For electric vehicles use a
:ref:`charging strategy <charging-strategies>`:

.. code-block:: python

    edisgo.apply_charging_strategy(strategy="dumb")  # "dumb", "reduced" or "residual"

For heat pumps, the (currently uncontrolled) operating strategy serves the heat
demand directly from the heat pump:

.. code-block:: python

    edisgo.apply_heat_pump_operating_strategy()

See :ref:`heat-pumps-flex` for details.

Reactive power
--------------

Two options exist for setting reactive power explicitly (note that
:meth:`~edisgo.edisgo.EDisGo.set_time_series_worst_case_analysis` already sets
fixed-cos-φ reactive power itself, so neither is needed after a worst-case setup).

Manual
~~~~~~

Provide your own series, as for active power above.

Fixed cos φ
~~~~~~~~~~~

Derive reactive power from active power with a fixed power factor:

.. code-block:: python

    edisgo.set_time_series_reactive_power_control()

By default (``"default"``) the power factors and inductive/capacitive mode are taken
from the configuration files. You can override them per component group via the
``generators_parametrisation`` / ``loads_parametrisation`` /
``storage_units_parametrisation`` arguments (a ``DataFrame`` with columns
``components``, ``mode`` and ``power_factor``), or set a group to ``None`` to leave it
untouched — all three default to ``"default"``. Make sure the active-power series are
set first. The sign conventions and the formula ``Q = P · tan(arccos(cos φ))`` are
explained in :ref:`reactive-power-flex`.

Scaling existing series
-----------------------

Already-set series can be scaled up or down (e.g. for a sensitivity study) with
:meth:`~edisgo.network.timeseries.TimeSeries.scale_timeseries`, applied in place to all
generators, loads and storage units:

.. code-block:: python

    edisgo.timeseries.scale_timeseries(p_scaling_factor=1.1, q_scaling_factor=1.0)
