.. _default_configs:

Configuration data
==================

eDisGo's default behaviour is controlled by configuration files shipped with
the package (``edisgo/config/``). You can inspect the current values below and
override them in several ways: persistently via the user config directory
(``~/.edisgo/config``, where the defaults are copied on first use), by pointing the
``config_path`` argument (a path or a dict) at your own configs, or in memory by
editing ``edisgo.config[...]``. This page reproduces the most relevant default files
verbatim; the additional ``config_system.cfg`` holds internal system paths, the
equipment-file mapping and network parameters (e.g. the grid frequency, 50 Hz) and is
not meant to be edited.

.. _config_db:

config_db_tables
----------------

``config_db_tables.cfg`` selects which data source (``model_draft`` vs ``versioned``)
and data version to use, and maps dataset names to the corresponding OpenEnergy
Platform / egon-data table names (see :ref:`data-sources`).

.. include:: ../../edisgo/config/config_db_tables_default.cfg
   :literal:

.. _config_grid_expansion:

config_grid_expansion
---------------------

``config_grid_expansion.cfg`` holds everything needed to size and cost grid
reinforcement: the standard equipment used for expansion
(``grid_expansion_standard_equipment``), the allowed voltage deviations
(``grid_expansion_allowed_voltage_deviations``), the line/transformer load factors for
normal operation (``grid_expansion_load_factors``) and for n-1 operation
(``grid_expansion_load_factors_n_minus_one``), each per voltage level and per
load/feed-in case, and the unit costs (``costs_cables``, ``costs_transformers``).
These drive :ref:`grid-reinforcement` and :ref:`grid-expansion-costs`.

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
:meth:`~edisgo.edisgo.EDisGo.integrate_component_based_on_geolocation`, the geometric
connection parameters such as the branch detour factor and buffer radii) and the
allowed MV/LV voltage deviations for component connection, plus the coordinate
reference system (SRID 4326).

.. include:: ../../edisgo/config/config_grid_default.cfg
   :literal:

.. _config_opf_julia:

config_opf_julia
----------------

``config_opf_julia.cfg`` is a legacy configuration file. The multi-period optimal
power flow (:ref:`flexibility-opf`) resolves the ``julia`` executable from the system
``PATH``, so ``julia_bin`` is not read; :meth:`~edisgo.edisgo.EDisGo.pm_optimize`
needs ``julia`` to be on the ``PATH`` instead. See :ref:`opf-requirements` for the
Julia and Gurobi setup it actually requires.

.. include:: ../../edisgo/config/config_opf_julia_default.cfg
   :literal:
