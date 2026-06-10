.. _data-model:

The data model
==============

Understanding eDisGo's data model makes every other feature easier to use. The
:class:`~edisgo.edisgo.EDisGo` object is the top-level API: you use it to import
data, run analyses, reinforce the grid and apply flexibility measures, and it gives
access to all data through a small number of container objects.

The EDisGo object and its containers
------------------------------------

In the examples below, ``edisgo`` is an :class:`~edisgo.edisgo.EDisGo` object.

.. code-block:: python

    edisgo.topology        # grid topology  -> Topology
    edisgo.timeseries      # time series     -> TimeSeries
    edisgo.results         # analysis results-> Results
    edisgo.electromobility # EV data         -> Electromobility
    edisgo.heat_pump       # heat-pump data  -> HeatPump
    edisgo.dsm             # DSM potential    -> DSM
    edisgo.overlying_grid  # constraints from the overlying grid -> OverlyingGrid
    edisgo.config          # configuration   -> Config

* :class:`~edisgo.network.topology.Topology` — buses, lines, transformers,
  switches, generators, loads and storage units.
* :class:`~edisgo.network.timeseries.TimeSeries` — active and reactive power time
  series of all components.
* :class:`~edisgo.network.results.Results` — power-flow results, equipment changes
  and grid-expansion costs (see :ref:`analysis-results`).
* :class:`~edisgo.network.electromobility.Electromobility`,
  :class:`~edisgo.network.heat.HeatPump`, :class:`~edisgo.network.dsm.DSM`,
  :class:`~edisgo.network.overlying_grid.OverlyingGrid` — flexibility data (see
  :ref:`flexibility-overview`).
* :class:`~edisgo.tools.config.Config` — configuration data (see
  :ref:`default_configs`).

Accessing grid data
-------------------

Topology data is stored in :pandas:`pandas.DataFrames<DataFrame>`, with one frame
per component type:

.. code-block:: python

    edisgo.topology.buses_df              # all buses (MV + LV)
    edisgo.topology.lines_df              # all lines
    edisgo.topology.transformers_df       # all MV/LV transformers
    edisgo.topology.transformers_hvmv_df  # HV/MV transformers
    edisgo.topology.switches_df           # switches
    edisgo.topology.generators_df         # generators
    edisgo.topology.loads_df              # loads (incl. heat pumps, charging points)
    edisgo.topology.storage_units_df      # storage units

Working with individual grids
-----------------------------

The grids can be accessed individually. The MV grid is an
:class:`~edisgo.network.grids.MVGrid`, each LV grid an
:class:`~edisgo.network.grids.LVGrid`:

.. code-block:: python

    # MV grid and its components (same DataFrame attributes as above)
    edisgo.topology.mv_grid
    edisgo.topology.mv_grid.buses_df
    edisgo.topology.mv_grid.generators_df

    # iterate over the LV grids (lv_grids returns a generator)
    list(edisgo.topology.mv_grid.lv_grids)

    # a single LV grid by id or name
    lv_grid = edisgo.topology.get_lv_grid("LVGrid_402945")
    lv_grid.buses_df
    lv_grid.loads_df

Single components can also be retrieved as objects
(:class:`~edisgo.network.components.Generator`,
:class:`~edisgo.network.components.Load`,
:class:`~edisgo.network.components.Storage`,
:class:`~edisgo.network.components.Switch`):

.. code-block:: python

    list(edisgo.topology.mv_grid.switch_disconnectors)
    list(lv_grid.generators)

Graph representation
--------------------

A :networkx:`networkx.Graph<>` representation is useful for path searches (e.g. from
a station to a generator). Lines become edges; buses and transformers become nodes:

.. code-block:: python

    edisgo.to_graph()              # whole topology
    edisgo.topology.mv_grid.graph  # MV grid
    lv_grid.graph                  # a single LV grid

For the sign and unit conventions used throughout these DataFrames, see
:ref:`definitions and units <definitions-and-units>`.
