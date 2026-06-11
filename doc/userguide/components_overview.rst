.. _components:

Components
==========

eDisGo represents a distribution grid as a set of typed components. This page lists
the component types and points to where each is described in detail.

Grid components
---------------

The physical grid elements are stored as :pandas:`pandas.DataFrames<DataFrame>` in the
:class:`~edisgo.network.topology.Topology` container — one frame per type (see
:ref:`data-model` for how to access them):

.. list-table::
   :header-rows: 1
   :widths: 30 38 32

   * - Component
     - Data (in ``edisgo.topology``)
     - Component object
   * - Bus
     - ``buses_df``
     - —
   * - Line
     - ``lines_df``
     - —
   * - Transformer (MV/LV)
     - ``transformers_df``
     - —
   * - Transformer (HV/MV)
     - ``transformers_hvmv_df``
     - —
   * - Switch
     - ``switches_df``
     - :class:`~edisgo.network.components.Switch`
   * - Generator
     - ``generators_df``
     - :class:`~edisgo.network.components.Generator`
   * - Load (incl. heat pumps and charging points)
     - ``loads_df``
     - :class:`~edisgo.network.components.Load`
   * - Storage unit
     - ``storage_units_df``
     - :class:`~edisgo.network.components.Storage`

To **add, remove or integrate** components, see :ref:`components-guide`.

Flexibility components
----------------------

Some loads and storage units can act as **flexibilities**. Their additional data lives
in dedicated containers; the modelling is described in :ref:`flexibility-overview`:

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - Flexibility
     - Container
     - Methodology
   * - Electric vehicles (charging points)
     - :class:`~edisgo.network.electromobility.Electromobility`
     - :ref:`electromobility-methodology`
   * - Heat pumps (+ thermal storage)
     - :class:`~edisgo.network.heat.HeatPump`
     - :ref:`heat-pumps-flex`
   * - Demand side management
     - :class:`~edisgo.network.dsm.DSM`
     - :ref:`dsm-flex`
   * - Battery / home storage
     - ``topology.storage_units_df``
     - :ref:`storage-flex`
   * - Overlying-grid requirements
     - :class:`~edisgo.network.overlying_grid.OverlyingGrid`
     - :ref:`overlying-grid-flex`

.. seealso::

   :ref:`data-model` for the data structure and how to access component data, and
   :ref:`components-guide` for adding and modifying components.
