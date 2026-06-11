.. _default_configs:

Configuration data
==================

eDisGo's default behaviour is controlled by configuration files shipped with
the package (``edisgo/config/``). You can inspect the current values below and
override them by editing the configuration of an :class:`~edisgo.edisgo.EDisGo`
object (``edisgo.config``). This page reproduces the most relevant default files
verbatim; the additional ``config_system.cfg`` only holds internal system paths and
is not meant to be edited.

.. _config_db:

config_db_tables
----------------

``config_db_tables.cfg`` selects which saved database connection and which
data-processing version to use when loading data from the OpenEnergy Platform /
egon-data (see :ref:`data-sources`).

.. include:: ../../edisgo/config/config_db_tables_default.cfg
   :literal:

.. _config_grid_expansion:

config_grid_expansion
---------------------

``config_grid_expansion.cfg`` holds everything needed to size and cost grid
reinforcement: the standard equipment used for expansion and its cost, the allowed
voltage deviations (``grid_expansion_allowed_voltage_deviations``) and the line/
transformer load factors (``grid_expansion_load_factors``), each per voltage level
and per load/feed-in case. These drive :ref:`grid-reinforcement` and
:ref:`grid-expansion-costs`.

.. include:: ../../edisgo/config/config_grid_expansion_default.cfg
   :literal:

.. _config_timeseries:

config_timeseries
-----------------

``config_timeseries.cfg`` defines the two worst-case situations (heavy load /
reverse feed-in) via simultaneity scale factors, the power factors and modes
(inductive/capacitive) used to generate reactive power (:ref:`reactive-power-flex`),
and the demandlib settings used when generating load profiles.

.. include:: ../../edisgo/config/config_timeseries_default.cfg
   :literal:

.. _config_grid:

config_grid
-----------

``config_grid.cfg`` specifies how new components are connected to the grid (the
voltage-level power thresholds used by
:meth:`~edisgo.edisgo.EDisGo.integrate_component_based_on_geolocation`) and where
disconnecting points are placed.

.. include:: ../../edisgo/config/config_grid_default.cfg
   :literal:

.. _config_opf_julia:

config_opf_julia
----------------

``config_opf_julia.cfg`` points to the Julia binary used by the multi-period optimal
power flow (:ref:`flexibility-opf`). It is only relevant when running
:meth:`~edisgo.edisgo.EDisGo.pm_optimize`; see :ref:`opf-requirements` for the full
Julia/Gurobi setup.

.. include:: ../../edisgo/config/config_opf_julia_default.cfg
   :literal:
