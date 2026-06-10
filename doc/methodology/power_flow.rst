.. _power-flow-methodology:

Power flow analysis
===================

In plain terms
--------------

A *power flow* computes the voltage at every bus and the current/loading on every
line and transformer for a given set of loads and generation. eDisGo uses it to find
where the grid is overloaded or where voltages leave the allowed band. It is the
measurement step that both :ref:`reinforcement <grid-reinforcement>` and the
:ref:`flexibility optimisation <flexibility-opf>` build on.

How it works
------------

:meth:`~edisgo.edisgo.EDisGo.analyze` runs a **non-linear AC power flow** using
`PyPSA <https://pypsa.org>`_. All loads and generators are modelled as **PQ nodes**
(fixed active and reactive power), and the **slack** is placed at the secondary side
of the HV/MV substation — i.e. the overlying grid is assumed to balance the
distribution grid at that point.

The power flow is solved for the time steps in
:attr:`~edisgo.network.timeseries.TimeSeries.timeindex`, or for a subset passed via
the ``timesteps`` argument. Internally the eDisGo object is converted to a PyPSA
network with :meth:`~edisgo.edisgo.EDisGo.to_pypsa`.

Physics
-------

At each bus the complex nodal power balance must hold,

.. math::

   S_i = P_i + \mathrm{j}\,Q_i = V_i \sum_k Y_{ik}^* \, V_k^*

where :math:`V_i` is the complex bus voltage and :math:`Y` the nodal admittance
matrix built from line and transformer impedances. PyPSA solves this non-linear
system with a Newton–Raphson iteration. The results — bus voltages
(:attr:`~edisgo.network.results.Results.v_res`), apparent powers
(:attr:`~edisgo.network.results.Results.s_res`) and currents
(:attr:`~edisgo.network.results.Results.i_res`) — are then compared against the
technical limits in :ref:`grid-reinforcement`.

.. _load-feedin-case:

Load case and feed-in case
--------------------------

Allowed voltage deviations and line/transformer load factors differ between two
situations commonly used in distribution-grid planning:

* **Load case** — high demand, low generation: power flows *from* the HV grid into
  the distribution grid.
* **Feed-in case** — high generation, low demand: power flows *back* to the HV grid
  (reverse power flow at the HV/MV substation).

When using worst-case time series (:ref:`worst-case-ts`) the two cases are built
from simultaneity (scale) factors in :ref:`config_timeseries`. When using real time
series, each time step is classified per grid by the sign of
(:math:`\sum \text{load} - \sum \text{generation}`): positive ⇒ load case, negative
⇒ feed-in case (grid losses are neglected for this classification). See
:meth:`~edisgo.network.timeseries.TimeSeries.timesteps_load_feedin_case`.

For the non-linear *optimal* power flow used for flexibility scheduling, see
:ref:`flexibility-opf`.
