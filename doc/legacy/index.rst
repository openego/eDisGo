.. _legacy:

Experimental & legacy features
==============================

.. warning::

   The methods on this page are **not maintained** and were **not adapted to the
   refactored eDisGo code base**. They may not work out of the box and are kept here
   for reference only. For the maintained flexibility and optimisation features, see
   :ref:`flexibility-overview`.

Non-linear optimal power flow
-----------------------------

An older multi-period non-linear OPF (``perform_mp_opf``) optimised generator
dispatch, curtailment and storage with regard to grid-expansion costs, using a
different formulation than the maintained PowerModels-based flexibility optimisation
(:ref:`flexibility-opf`). It is no longer maintained.

.. _legacy-curtailment:

Curtailment heuristics
----------------------

Two heuristics (in ``edisgo.flex_opt.curtailment``) spatially distributed a given
curtailment target — obtained from an EHV/HV optimisation with
`eTraGo <https://github.com/openego/eTraGo>`_ — across the wind and solar generators
in the grid.

**feedin-proportional** (``feedin_proportional``) allocated curtailment in proportion
to each generator's share of total feed-in:

.. math::

   c_{g,t} = \frac{a_{g,t}}{\sum_{g} a_{g,t}} \cdot c_{\text{target},t}

where :math:`c_{g,t}` is the curtailed power of generator :math:`g`, :math:`a_{g,t}`
its weather-dependent availability and :math:`c_{\text{target},t}` the target.

**voltage-based** (``voltage_based``) allocated more curtailment to generators at
nodes with higher voltage-limit exceedance, solving a small linear program to set the
proportionality. See the git history for the full formulation.

.. _legacy-storage-positioning:

Storage-positioning heuristic
-----------------------------

A heuristic (``edisgo.flex_opt.storage_positioning.one_storage_per_feeder``) took a
given total storage capacity and distributed it over several smaller units to reduce
line overloading and voltage deviations, starting with the feeder with the highest
theoretical grid-expansion cost. Like the curtailment heuristics it was intended to
be used together with eTraGo, which optimised the storage capacity and operation.

For maintained storage handling see :ref:`storage-flex`.
