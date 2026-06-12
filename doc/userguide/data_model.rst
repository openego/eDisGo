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

Each :class:`~edisgo.network.grids.Grid` can also assign **feeders** (the branches
leaving the grid's station) to its buses and lines and report per-feeder statistics:

.. code-block:: python

    edisgo.topology.mv_grid.assign_grid_feeder()   # tag buses/lines with their feeder
    edisgo.topology.mv_grid.get_feeder_stats()     # statistics per feeder

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

.. _pypsa-relationship:

Relationship to the PyPSA format
--------------------------------

eDisGo's static data model is a set of :pandas:`pandas.DataFrames<DataFrame>` that is
deliberately **close to, but not identical with, the** `PyPSA <https://pypsa.org>`_
**component model**. eDisGo does not store a PyPSA network internally; instead it
converts its topology on demand — :meth:`~edisgo.edisgo.EDisGo.to_pypsa` builds a
PyPSA network for the power flow, and :meth:`~edisgo.edisgo.EDisGo.to_powermodels`
builds the OPF input. A second, important difference: in eDisGo the **time series
live in a separate container** (:class:`~edisgo.network.timeseries.TimeSeries`), not
in the component frames — ``to_pypsa`` fills PyPSA's ``loads_t``/``generators_t`` etc.
from it.

Component mapping
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - eDisGo (``topology``)
     - PyPSA component
     - Notes / differing columns
   * - ``buses_df``
     - ``Bus``
     - ``v_nom``, ``x``, ``y`` as in PyPSA; eDisGo adds ``mv_grid_id`` /
       ``lv_grid_id`` (grid hierarchy) and ``in_building``.
   * - ``lines_df``
     - ``Line``
     - ``bus0``/``bus1``/``length``/``s_nom``/``num_parallel`` as in PyPSA; ``r`` and
       ``x`` are stored in **ohms** and passed straight through to PyPSA, which does the
       per-unit conversion internally from ``v_nom`` (``to_pypsa`` itself does *not*
       convert them). eDisGo uses ``type_info`` (cable/line type name) and ``kind``
       (``"cable"``/``"line"``) instead of PyPSA's standard-type ``type``. A ``b`` column
       exists but is always 0 and there is no ``g``, so **line shunt admittance is
       effectively neglected**.
   * - ``transformers_df``, ``transformers_hvmv_df``
     - ``Transformer``
     - MV/LV and HV/MV transformers; the HV/MV transformer's secondary side carries
       the **slack**.
   * - ``loads_df``
     - ``Load``
     - ``bus``/``p_set`` as in PyPSA; eDisGo adds ``type`` (``conventional_load`` /
       ``charging_point`` / ``heat_pump``), ``sector``, ``building_id``,
       ``annual_consumption``, ``number_households``.
   * - ``generators_df``
     - ``Generator``
     - ``bus``/``p_nom``/``control`` as in PyPSA; eDisGo adds ``type`` (technology),
       ``subtype`` and ``weather_cell_id``.
   * - ``storage_units_df``
     - ``StorageUnit``
     - ``p_nom``/``max_hours``/``efficiency_store``/``efficiency_dispatch``/``control``
       — these names are taken directly from PyPSA.
   * - ``switches_df``
     - —
     - **No PyPSA equivalent.** Switches are resolved into ``lines_df`` (open/closed
       state) before conversion, so PyPSA sees the radial operating state
       (:ref:`switches-explained`).

What eDisGo needs beyond PyPSA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compared with a bare PyPSA network, eDisGo additionally requires:

* the **grid hierarchy** — every bus tagged with its ``mv_grid_id`` and (for LV buses)
  ``lv_grid_id``, plus a defined HV/MV transformer — because eDisGo reasons per MV/LV
  grid (reinforcement, feeders, complexity reduction);
* **geo-coordinates** (``x``/``y``) for line lengths, plotting and spatial reduction;
* **component typing** — ``type``/``sector`` on loads, ``type``/``subtype``/
  ``weather_cell_id`` on generators, ``type_info``/``kind`` on lines — used for
  equipment limits, costs and time-series generation;
* the **flexibility containers** (:class:`~edisgo.network.electromobility.Electromobility`,
  :class:`~edisgo.network.heat.HeatPump`, :class:`~edisgo.network.dsm.DSM`,
  :class:`~edisgo.network.overlying_grid.OverlyingGrid`), which have no PyPSA counterpart.

Conversely, what PyPSA needs is assembled by ``to_pypsa``: the reactive-power series
``q_set`` from :ref:`reactive-power-flex`, and the slack at the HV/MV station. The
ohmic line ``r``/``x`` are handed to PyPSA unchanged — PyPSA converts them to per-unit
internally — while transformer per-unit values (``r_pu``/``x_pu``) are already computed
at ding0 import time, not by ``to_pypsa``.

Coming from a PyPSA network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is **no automatic PyPSA-to-eDisGo importer**: eDisGo loads grids from
ding0-format CSV files (:ref:`data-sources`). To represent an existing PyPSA grid in
eDisGo you therefore need to provide the data in eDisGo's schema — in practice:

#. assign every bus to an MV grid (and LV buses to an LV grid) via ``mv_grid_id`` /
   ``lv_grid_id``, and make sure there is an HV/MV transformer;
#. supply ``x``/``y`` coordinates for all buses;
#. give lines an ``s_nom``, ``length``, ``num_parallel``, ``type_info`` and ``kind``
   (``r``/``x`` in ohms);
#. classify loads (``type``, ``sector``) and generators (``type``, ``subtype``,
   ``weather_cell_id``);
#. add switches only if you want ring topology (otherwise the grid is purely radial);
#. set the time series through the :class:`~edisgo.network.timeseries.TimeSeries` API
   rather than on the components.
