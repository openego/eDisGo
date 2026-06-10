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

.. code-block:: python

    edisgo.set_time_series_manual()

.. _worst-case-ts:

Worst-case
~~~~~~~~~~

Set feed-in and load for the two classic grid-planning situations — heavy load
(``load_case``) and reverse power flow (``feed-in_case``) — using simultaneity
factors from the configuration files.

.. code-block:: python

    edisgo.set_time_series_worst_case_analysis()

A fictitious time index starting 1970-01-01 00:00 is set automatically (PyPSA needs
a time index). ``edisgo.timeseries.timeindex_worst_cases`` tells you which time step
maps to which case. The definition of load and feed-in case is explained in
:ref:`load-feedin-case`.

Predefined
~~~~~~~~~~

Set series by component type, either from your own data or from public sources:

.. code-block:: python

    edisgo.set_time_series_active_power_predefined()

* **Fluctuating generators** — wind and solar feed-in from the
  ``oedb`` (OpenEnergy DataBase).
* **Conventional loads** — standard load profiles per sector via
  ``demandlib`` (see :ref:`data-sources`).

For all other components you provide your own series. See
:meth:`~edisgo.edisgo.EDisGo.set_time_series_active_power_predefined`.

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

Two options exist for reactive power; more controls (Q(U), cos φ(P)) are planned.

Manual
~~~~~~

Provide your own series, as for active power above.

Fixed cos φ
~~~~~~~~~~~

Derive reactive power from active power with a fixed power factor:

.. code-block:: python

    edisgo.set_time_series_reactive_power_control()

Make sure the active-power series are set first. The sign conventions and the
formula ``Q = P · tan(arccos(cos φ))`` are explained in :ref:`reactive-power-flex`.
