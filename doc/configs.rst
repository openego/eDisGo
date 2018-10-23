.. _default_configs:

Default configuration data
============================

Following you find the default configuration files.

.. _config_db:

config_db_tables
-------------------

The config file ``config_db_tables.cfg`` holds data about which database connection
to use from your saved database connections and which dataprocessing version.

.. include:: config/config_db_tables_default.cfg
   :literal:

.. _config_grid_expansion:

config_grid_expansion
----------------------

The config file ``config_grid_expansion.cfg`` holds data mainly needed to determine
grid expansion needs and costs - these are standard equipment to use in grid expansion and
its costs, as well as allowed voltage deviations and line load factors.

.. include:: config/config_grid_expansion_default.cfg
   :literal:

.. _config_timeseries:

config_timeseries
----------------------

The config file ``config_timeseries.cfg`` holds data to define the two worst-case
scenarions heavy load flow ('load case') and reverse power flow ('feed-in case')
used in conventional grid expansion planning, power factors and modes (inductive
or capacitative) to generate reactive power time series, as well as configurations
of the demandlib in case load time series are generated using the oemof demandlib.

.. include:: config/config_timeseries_default.cfg
   :literal:

.. _config_grid:

config_grid
----------------------

The config file ``config_grid.cfg`` holds data to specify parameters used when 
connecting new generators to the grid and where to position disconnecting points.

.. include:: config/config_grid_default.cfg
   :literal:
