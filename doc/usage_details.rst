.. _usage-details:

Usage details
=============

As eDisGo is designed to serve as a toolbox, it provides several methods to
analyze distribution grids for grid issues and evaluate measures responding these.
`Examples <https://github.com/openego/eDisGo/tree/dev/edisgo/examples>`_
are provided to show a typical workflow of how eDisGo can be used. See
the standard example or take a look at a
`script <https://gist.github.com/gplssm/e0fd6cb99f7e8c1eed3fd4a4e325dde0>`_
that is used to assess grid
extension costs for distribution grids in the upcoming two decades.
Further, we discuss how different features can be used in detail below.

The fundamental data structure
------------------------------

It's worth to understand how the fundamental data structure of eDisGo is
designed in order to make use of its entire features.

The class :class:`~.grid.network.EDisGo` serves as the top-level API for setting up your scenario,
invocation of data import, analysis of hosting capacity, grid reinforcement and flexibility measures.

If you want to set up a scenario to do a worst-case analysis of a ding0 grid (see :ref:`prerequisites`) you simply have
to provide a grid and set the :attr:`worst_case_analysis` parameter. The following example assums you have a file of a
ding0 grid named "ding0_grids__42.pkl" in current working directory.

.. code-block:: python

    from edisgo import EDisGo

    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    worst_case_analysis='worst-case-feedin')

You can also provide your own time series for load and feed-in for the analysis.

.. code-block:: python

    import pandas as pd

    # set up load and feed-in time series
    timeindex = pd.date_range('1/1/1970', periods=3, freq='H')
    feedin_renewables = pd.DataFrame(data={'solar': [10, 20, 30], 
	                                   'wind': [10, 10, 15]},
	                             index=timeindex)
    feedin_dispatchable = pd.DataFrame(data={'coal': [5, 10, 5],
	                                     'other': [3, 10, 7]},
	                               index=timeindex)
    load = pd.DataFrame(data={'residential': [10, 20, 20],
	                      'retail': [5, 5, 5],
	                      'industrial': [2, 2, 2],
	                      'agricultural': [10, 10, 10]},
	                index=timeindex)

    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    timeseries_generation_fluctuating=feedin_renewables,
		    timeseries_generation_dispatchable=feedin_dispatchable,
		    timeseries_load=load)

EDisGo also offers methods to generate load and feed-in time series. See :class:`~.grid.network.EDisGo` for
more information on which options to choose from and what other data can be provided.

All data is stored in the class :class:`~.grid.network.Network`. The network class serves as an overall 
data container in eDisGo holding the grid data for the :class:`~.grid.grid.MVGrid` and :class:`~.grid.grid.LVGrid` s, :class:`~.grid.network.Config` 
data, :class:`~.grid.network.Results`, :class:`~.grid.network.Timeseries`, etc. It is linked from multiple locations
and provides hierarchical access to all data. Network itself can be accessed via the EDisGo object.

.. code-block:: python

    # Access to Network data container object
    edisgo.network

The grid data and results can e.g. be accessed via

.. code-block:: python

    # MV grid instance
    edisgo.network.mv_grid

    # List of LV grid instances
    edisgo.network.lv_grids

    # Results of network analysis
    edisgo.network.results

The grid topology is represented by separate undirected graphs for the MV
grid and each of the LV grids. The :class:`~.grid.network.Graph` is subclassed from
:networkx:`networkx.Graph<graph>` and extended by extra-functionality.
Lines represent edges in the graph. Other equipment is represented by a node.


.. todo::

    Add more
     * Add examples on accessing particular data, i.e. generators


Identify grid issues
--------------------

Use PyPSA's non-linear power flow to perform a stationary power flow analysis.

As detailed in :ref:`edisgo-mwe`, once you set up your scenario by instantiating an
:class:`~.grid.network.EDisGo` object, you are ready for an analysis of grid
issues (line overloading or voltage band violations) respectively the hosting
capacity of the grid by :meth:`~.grid.network.EDisGo.analyze()`:

.. code-block:: python

    # Do non-linear power flow analysis for MV and LV grid
    edisgo.analyze()

The range of time analyzed by the power flow analysis is defined by :attr:`~.grid.network.TimeSeries.timeindex`
of :class:`~.grid.network.TimeSeries` class.

Grid extension
--------------

Grid extension can be invoked by :meth:`~.grid.network.EDisGo.reinforce()`:

.. code-block:: python

    # Reinforce grid due to overloading and overvoltage issues
    edisgo.reinforce()

Costs for the grid extension measures can be obtained as follows:

.. code-block:: python

    # Get costs of grid extension
    costs = edisgo.network.results.grid_expansion_costs

Further information on the grid reinforcement methodology can be found in section
:ref:`grid_expansion_methodology`.

Battery storages
----------------

Battery storages can be integrated into the grid as alternative to classical
grid extension. A battery in eDisGo is represented by the class
:class:`~.grid.components.Storage`. 
In order integrate a storage into the grid, start from the following exemplary code:

.. code-block:: python

    # define storage parameters
    storage_parameters = {'nominal_capacity': 10,
			  'soc_initial': 0,
                          'efficiency_in': .9,
                          'efficiency_out': .9,
                          'standing_loss': 0}

    # add storage instance to the grid
    edisgo.integrate_storage(battery_position='hvmv_substation_busbar',
                             battery_parameters=storage_parameters,
			     timeseries_battery='fifty-fifty')

Using the method :meth:`~.grid.network.EDisGo.integrate_storage()` provides a
high-level interface to define the position and storage operation at once,
based on predefined rules. Thus, a limited set of storage integration rules are
implemented. See :class:`~.grid.network.StorageControl` for
available storage integration strategies.

You can also integrate a storage directly upon defining your scenario. Assuming
you have the load and feed-in time series as well as the storage parameters defined
above you can do the following:

.. code-block:: python

    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    timeseries_generation_fluctuating=feedin_renewables,
		    timeseries_generation_dispatchable=feedin_dispatchable,
		    timeseries_load=load,
                    battery_position='hvmv_substation_busbar',
                    battery_parameters=storage_parameters,
		    timeseries_battery='fifty-fifty')

Curtailment
-----------

Curtailment can be specified per generation technology as factor between 0 and
1. Then, power output of a technology is cropped at its specific curtailment
factor.
For example define a curtailment for wind and PV power at 70 %.

.. code-block:: python

    # curtailment of each technology relative (0..1)
    curtailment = {'wind': 0.7, 'solar': 0.7}

    # define curtailment for power flow analysis through scenario
    scenario = Scenario(
                power_flow=(date(2011, 10, 10), date(2011, 10, 13)),
                mv_grid_id=42,
                scenario_name=['NEP 2035', 'Status Quo'],
                curtailment=curtailment)


Retrieve results
----------------

Results - voltage levels and line loading - from the power flow analysis are
provided through :class:`~.grid.network.Results`. Get voltage levels at nodes
from :meth:`~.grid.network.Results.v_res`
and line loading from :meth:`~.grid.network.Results.s_res` or
:attr:`~.grid.network.Results.i_res` respectively.
:attr:`~.grid.network.Results.equipment_changes` details about measures
performed during grid extension. Associated cost are determined by
:attr:`~.grid.network.Results.grid_expansion_costs`.
Flexibility measure may not entirely resolve all issues.
These are listed in :attr:`~.grid.network.Results.unresolved_issues`.
