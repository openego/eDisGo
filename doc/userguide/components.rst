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

    # add a storage unit to a known bus (no time series set — e.g. for a
    # subsequent worst-case study, where the series come from
    # set_time_series_worst_case_analysis)
    edisgo.add_component(
        comp_type="storage_unit",
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

.. _components-geolocation:

Integration based on geolocation
--------------------------------

:meth:`~edisgo.edisgo.EDisGo.integrate_component_based_on_geolocation` connects a
component to the most suitable point in the grid given its coordinates and nominal
power. The voltage level it is connected to depends on its power (the thresholds
are defined in :ref:`config_grid`); large components are connected directly to a
station, smaller ones to the nearest suitable line or bus.

If all LV buses have geolocations, components in the LV grid are connected to the
closest LV bus, or via a new bus and line if the closest bus is farther away than
``max_distance_from_target_bus`` (default 20 m). With ``connect_to_load_bus=True``
(off by default), a component in voltage level 7 is connected to the closest bus with
a matching load instead, e.g. a home charging point to a bus with a residential load
and a work charging point to a bus with a CTS, industrial or agricultural load. Small
generators and storage units up to 30 kW match residential loads, larger ones CTS,
industrial or agricultural loads, and heat pumps for individual heating match any
load. A bus only qualifies if it lies within ``max_distance_from_target_bus`` and no
more than ``allowed_number_of_comp_per_bus`` (default 2) components of the same type
are already connected to it. Otherwise, and for all other components, e.g. public
charging points, the component is connected as without the option.

In the following example with ding0 test grid 3, the charging point lies at a building
with a CTS load. Without the option, it is connected to the bus of this building; with
the option, it is connected to the bus of the neighbouring building with a residential
load, 11 m away.

.. code-block:: python

    edisgo.integrate_component_based_on_geolocation(
        comp_type="charging_point",
        geolocation=(10.018961, 47.553177),
        sector="home",
        p_set=0.011,
        add_ts=False,
        connect_to_load_bus=True,
    )

For all details see
:attr:`~edisgo.network.topology.Topology.connect_to_lv_based_on_geolocation`.

Aggregating components
----------------------

:meth:`~edisgo.edisgo.EDisGo.aggregate_components` combines the generators
respectively loads at the same bus into a single component, which can speed up power
flow and optimisation. By default all generators (resp. all loads) at a bus are
aggregated; you can instead group them by further columns — e.g. by type or sector —
via ``aggregate_generators_by_cols`` / ``aggregate_loads_by_cols``. Storage units are
not aggregated.

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
