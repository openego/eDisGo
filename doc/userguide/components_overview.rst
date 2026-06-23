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

.. _switches-explained:

Switches
--------

**What they are.** A switch is a **disconnecting point** in the grid — above all the
point where a **ring is left open**. Medium-voltage grids are built as rings (for
redundancy / n-1) but **operated radially**, open at one point; the switch marks that
open point. eDisGo so far only models *switch disconnectors*. They are held in
:attr:`~edisgo.network.topology.Topology.switches_df`, which is defined for the MV grid
and its underlying LV grids; in the ding0 grids eDisGo uses they are the MV ring
disconnectors, sitting at MV/LV stations.

**How they are represented.** A switch is a **branch** (a line in ``lines_df``) with two
candidate end buses and a ``state``:

* **open** (normal operation) — the branch is attached to a dedicated *virtual* bus
  ``bus_open``, so the ring is **broken** and the grid is radial;
* **closed** — the branch is attached to ``bus_closed`` instead, **closing the ring**.

``switches_df`` therefore records, per switch, the ``branch`` it controls and the two
candidate buses; opening or closing a switch just rewires that branch's bus in
``lines_df``.

**What they are used for.** Switches are not decorative — several parts of eDisGo depend
on them:

* **The operating point of every power flow.** Because the state is baked into
  ``lines_df``, :meth:`~edisgo.edisgo.EDisGo.analyze` and the optimisation run on the
  current (normally radial) configuration. Closing a switch makes the grid meshed and
  changes the result.
* **Ring detection.** :attr:`~edisgo.network.topology.Topology.rings` finds the grid's
  rings by temporarily closing all switches and looking for cycles — so it knows which
  buses form which ring.
* **Grid reinforcement.** When :meth:`~edisgo.edisgo.EDisGo.reinforce` fixes MV voltage
  problems by splitting a feeder (:ref:`grid-reinforcement`), the feeder can only be
  split **at an LV station**, because that is where a switch disconnector can be
  operated to run the line as two *half-rings*. LV stations thus decide *where* MV line
  reinforcement can act, and the resulting half-ring structure feeds the voltage-level
  cost allocation (:ref:`grid-expansion-costs`).

  .. note::

     The feeder is split at the LV station and connected **directly back to the MV
     station busbar**; a switch disconnector is **not yet inserted** at the new split
     point (``TODO`` in
     :py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_lines_voltage_issues`), so
     the half-rings created by reinforcement are not switchable in the model.
* **Plotting.** The MV topology plots mark the switch disconnectors.

**Working with switches.** Each switch is a
:class:`~edisgo.network.components.Switch` object:

.. code-block:: python

    # all switches, and the switch disconnectors of the MV grid
    edisgo.topology.switches_df
    list(edisgo.topology.mv_grid.switch_disconnectors)   # Switch objects

    sw = next(edisgo.topology.mv_grid.switch_disconnectors)
    sw.state          # "open" or "closed"
    sw.branch         # the line in lines_df that represents the switch
    sw.bus_open, sw.bus_closed
    sw.close()        # close the ring (rewires sw.branch in lines_df)
    sw.open()         # re-open it

Switches are imported from the ding0 ``switches.csv`` file, carried over through
:meth:`~edisgo.edisgo.EDisGo.spatial_complexity_reduction` (buses and branches are
remapped; switches that become unused or duplicated by the aggregation are
dropped/merged), and — being expressed as
ordinary branches — carried into the PyPSA network by
:meth:`~edisgo.edisgo.EDisGo.to_pypsa` without any special switch element (see
:ref:`pypsa-relationship`).

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
