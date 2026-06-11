.. _components-guide:

Adding and modifying components
===============================

Besides importing whole scenarios, you can add, remove and integrate individual
components — generators, loads, storage units, charging points or heat pumps. For an
overview of all component types, see :ref:`components`.

Adding a component
------------------

Use :meth:`~edisgo.edisgo.EDisGo.add_component`. If the bus is known, the component
is connected there; otherwise see geolocation-based integration below.

.. code-block:: python

    # add a storage unit to a known bus (worst-case study: no time series needed)
    edisgo.add_component(
        comp_type="storage_unit",
        add_ts=False,
        bus=edisgo.topology.buses_df.index[3],
        p_nom=4,
    )

In a time-series study you provide the series directly:

.. code-block:: python

    edisgo.add_component(
        comp_type="storage_unit",
        bus=edisgo.topology.buses_df.index[3],
        p_nom=4,
        ts_active_power=pd.Series([-3.4, 2.5, -3.4, 2.5], index=edisgo.timeseries.timeindex),
        ts_reactive_power=pd.Series([0.0, 0.0, 0.0, 0.0], index=edisgo.timeseries.timeindex),
    )

Removing a component
--------------------

.. code-block:: python

    edisgo.remove_component(comp_type="load", comp_name="Load_XYZ")

Integration based on geolocation
--------------------------------

:meth:`~edisgo.edisgo.EDisGo.integrate_component_based_on_geolocation` connects a
component to the most suitable point in the grid given its coordinates and nominal
power. The voltage level it is connected to depends on its power (the thresholds
are defined in :ref:`config_grid`); large components are connected directly to a
station, smaller ones to the nearest suitable line or bus.

Aggregating components
----------------------

:meth:`~edisgo.edisgo.EDisGo.aggregate_components` combines components of the same
type at a bus (e.g. all loads of a sector) into a single component, which can speed
up power flow and optimisation.

Manual heat-pump and storage integration
-----------------------------------------

Heat pumps are added as loads of ``type="heat_pump"`` and then equipped with COP and
heat-demand series:

.. code-block:: python

    bus = edisgo.topology.loads_df[
        edisgo.topology.loads_df.sector == "residential"].bus[0]
    hp_name = edisgo.add_component(
        "load", bus=bus, p_set=0.015, type="heat_pump")

    # cop and heat_demand are pandas Series you provide (indexed by the time index),
    # e.g. from measured or modelled data
    edisgo.heat_pump.set_cop(edisgo, cop.to_frame(name=hp_name))
    edisgo.heat_pump.set_heat_demand(edisgo, heat_demand.to_frame(name=hp_name))
    edisgo.apply_heat_pump_operating_strategy()

For the methodology behind heat-pump and storage operation see :ref:`heat-pumps-flex`
and :ref:`storage-flex`.
