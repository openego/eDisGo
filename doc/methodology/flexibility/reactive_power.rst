.. _reactive-power-flex:

Reactive power
==============

In plain terms
--------------

Beyond active power, generators and loads also exchange **reactive power**. Reactive
power does no useful work but it strongly influences voltages: injecting or absorbing
it at the right place can raise or lower local voltages and so relieve the grid
without any new hardware. eDisGo currently provides a fixed-power-factor control.

Fixed cos φ control
-------------------

:meth:`~edisgo.edisgo.EDisGo.set_time_series_reactive_power_control` derives a
reactive-power series from the active-power series using a fixed power factor cos φ,
per component type and voltage level (the values come from
:ref:`config_timeseries`). The core calculation is
:py:func:`~edisgo.flex_opt.q_control.fixed_cosphi`:

.. math::

   Q = P \cdot \tan(\arccos(\cos\varphi)) \cdot \text{sign} .

The **sign** encodes whether the component behaves inductively or capacitively and
follows eDisGo's (PyPSA-based) sign convention, which is *opposite* for generators and
loads. The helpers
:py:func:`~edisgo.flex_opt.q_control.get_q_sign_generator` and
:py:func:`~edisgo.flex_opt.q_control.get_q_sign_load` return the correct sign for a
given reactive-power mode (inductive/capacitive). See
:ref:`definitions and units <definitions-and-units>` for the full sign convention.

By default (``"default"``) the power factor and inductive/capacitive mode of every
component group come from the configuration files. They can be overridden per group by
passing a ``DataFrame`` (with columns ``components``, ``mode`` and ``power_factor``) to
``generators_parametrisation`` / ``loads_parametrisation`` /
``storage_units_parametrisation``, or a group can be set to ``None`` to leave it
untouched.

.. note::

   Reactive power must be set **after** all active-power series are in place (it is
   derived from them) — see the :ref:`ordering rules <ordering-rules>`.

Physics
-------

The voltage change along a feeder depends on both active and reactive power flow,
:math:`\Delta V \approx (R\,P + X\,Q)/V`. Because distribution lines have a sizeable
reactance :math:`X`, modulating :math:`Q` is an effective, hardware-free way to
support voltage — complementary to the reinforcement measures in
:ref:`grid-reinforcement`. Voltage-dependent controls (Q(U), cos φ(P)) are planned
for future releases.
