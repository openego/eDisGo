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
grid, such as:

* ``renewables_curtailment`` — required curtailment of renewable feed-in,
* ``storage_units_active_power`` / ``storage_units_soc`` — required central-storage
  dispatch and state of charge,
* ``dsm_active_power`` — required DSM utilisation,
* ``electromobility_active_power`` — required aggregate EV charging,
* ``heat_pump_decentral_active_power`` /
  ``thermal_storage_units_decentral_soc`` — required flexible heat-pump operation.

The helper
:py:func:`~edisgo.network.overlying_grid.distribute_overlying_grid_requirements`
distributes these aggregate requirements onto the individual components in the grid.

Use in the optimisation
-----------------------

The high-voltage requirements are honoured by the OPF versions that add HV
constraints — ``opf_version`` 3 and 4 of :meth:`~edisgo.edisgo.EDisGo.pm_optimize`
(see :ref:`flexibility-opf`). This keeps the local, distribution-level flexibility
schedule consistent with what the transmission-level planning expects, so the two
levels can be optimised coherently.
