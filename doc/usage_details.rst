.. _usage-details:

Usage details
=============

As eDisGo is designed to serve as a toolbox, it provides several methods to
analyze distribution grids for grid issues and to evaluate measures responding these.
We provide two examples, an 
:download:`example script <../edisgo/examples/example.py>`
and :download:`jupyter notebook <../edisgo/examples/edisgo_simple_example.ipynb>`.

Further, we discuss how different features can be used in detail below.

The fundamental data structure
------------------------------

It's worth to understand how the fundamental data structure of eDisGo is
designed in order to make use of its entire features.

The class :class:`~.grid.network.EDisGo` serves as the top-level API for setting up your scenario,
invocation of data import, analysis of hosting capacity, grid reinforcement and flexibility measures.

If you want to set up a scenario to do a worst-case analysis of a ding0 grid (see :ref:`prerequisites`) you simply have
to provide a grid and set the :attr:`worst_case_analysis` parameter. The following example assumes you have a file of a
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
    feedin_renewables = pd.DataFrame(data={'solar': [0.1, 0.2, 0.3], 
	                                   'wind': [0.3, 0.15, 0.15]},
	                             index=timeindex)
    feedin_dispatchable = pd.DataFrame(data={'coal': [0.5, 0.1, 0.5],
	                                     'other': [0.3, 0.1, 0.7]},
	                               index=timeindex)
    load = pd.DataFrame(data={'residential': [0.00001, 0.00002, 0.00002],
	                      'retail': [0.00005, 0.00005, 0.00005],
	                      'industrial': [0.00002, 0.00003, 0.00002],
	                      'agricultural': [0.00001, 0.000015, 0.00001]},
	                index=timeindex)

    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    timeseries_generation_fluctuating=feedin_renewables,
		    timeseries_generation_dispatchable=feedin_dispatchable,
		    timeseries_load=load)

EDisGo also offers methods to generate load time series and feed-in time series for fluctuating generators (see last :ref:`edisgo-mwe`).
See :class:`~.grid.network.EDisGo` for
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
    edisgo.network.mv_grid.lv_grids

    # Results of network analysis
    edisgo.network.results
  
    # MV grid generators
    edisgo.network.mv_grid.generators

The grid topology is represented by separate undirected graphs for the MV
grid and each of the LV grids. The :class:`~.grid.network.Graph` is subclassed from
:networkx:`networkx.Graph<graph>` and extended by extra-functionality.
Lines represent edges in the graph. Other equipment is represented by a node.


Identify grid issues
--------------------

As detailed in :ref:`edisgo-mwe`, once you set up your scenario by instantiating an
:class:`~.grid.network.EDisGo` object, you are ready for an analysis of grid
issues (line overloading or voltage band violations) respectively the hosting
capacity of the grid by :meth:`~.grid.network.EDisGo.analyze()`:

.. code-block:: python

    # Do non-linear power flow analysis for MV and LV grid
    edisgo.analyze()

The analyze function conducts a non-linear power flow using PyPSA.

The range of time analyzed by the power flow analysis is by default defined by the timeindex 
given to the EDisGo API but can also be specified by providing the parameter *timesteps* to analyze. 

Grid extension
--------------

Grid extension can be invoked by :meth:`~.grid.network.EDisGo.reinforce()`:

.. code-block:: python

    # Reinforce grid due to overloading and overvoltage issues
    edisgo.reinforce()

You can further specify e.g. if to conduct a combined analysis for MV and LV (regarding allowed voltage
deviations) or if to only calculate grid expansion needs without changing the topology of the graph. See
:meth:`~.grid.network.EDisGo.reinforce()` for more information.

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
Using the method :meth:`~.grid.network.EDisGo.integrate_storage()` provides a
high-level interface to define the position, size and storage operation,
based on user input and predefined rules. A limited set of storage integration rules are
implemented. See :class:`~.grid.network.StorageControl` for
available storage integration strategies.

Here is a small example on how to integrate a storage:

.. code-block:: python

    # define storage parameters
    storage_parameters = {'nominal_power': 200}

    # add storage instance to the grid
    edisgo.integrate_storage(position='hvmv_substation_busbar',
                             timeseries='fifty-fifty',
                             parameters=storage_parameters)

Further information on the storage integration methodology 'distribute_storages_mv' can be found in section
:ref:`storage-integration-label`.

Curtailment
-----------

The curtailment function is used to spatially distribute the power that is to be curtailed.
There are currently two options for doing this distribution:

* `feedin-proportional`
    Distributes the curtailed power to all the fluctuating generators depending on
    their weather-dependent availability. 
* `voltage-based`
    Distributes the curtailed power depending on the exceedance of the allowed voltage deviation at the nodes
    of the fluctuating generators.

The input to the curtailment function can be modified to curtail certain technologies or technologies by the weather cell they are in.
Opposite to the load and feed-in time series curtailment time series need to be given in kW.
Following are examples of the different options of how to specify curtailment requirements:

.. code-block:: python

    timeindex = pd.date_range('1/1/1970', periods=3, freq='H')

    # curtailment is allocated to all solar and wind generators
    curtailment = pd.Series(data=[0.0, 5000.0, 3000.0],
			    index=timeindex)

    # curtailment is allocated by generator type
    curtailment = pd.DataFrame(data={'wind': [0.0, 5000.0, 3000.0],
                                     'solar': [5500.0, 5400.0, 3200.0]},
                               index=timeindex)

    # curtailment is allocated by generator type and weather cell
    curtailment = pd.DataFrame(data={('wind', 1): [0.0, 5000.0, 3000.0],
                                     ('wind', 2): [100.0, 2000.0, 300.0],
    		                     ('solar', 1): [500.0, 5000.0, 300.0]},
    			       index=timeindex)

Set curtailment by calling the method :meth:`~.grid.network.EDisGo.curtail()`:

.. code-block:: python

    edisgo.curtail(curtailment_methodology='feedin-proportional',
                   timeseries_curtailment=curtailment)


or with

.. code-block:: python

    edisgo.curtail(curtailment_methodology='voltage-based',
                   timeseries_curtailment=curtailment)

Plots
----------------

EDisGo provides a bunch of predefined plots to e.g. plot the MV grid topology, and line loading and node voltages
in the MV grid or as a histogram.

.. code-block:: python

    # plot MV grid topology on a map
    edisgo.plot_mv_grid_topology()

    # plot grid expansion costs for lines in the MV grid and stations on a map
    edisgo.plot_grid_expansion_costs()

    # plot voltage histogram
    edisgo.histogram_voltage()

See :class:`~.grid.network.EDisGoRemiport` class for more plots and plotting options.

Results
----------------

Results such as voltage levels and line loading from the power flow analysis and 
grid extension costs are provided through the :class:`~.grid.network.Results` class
and can be accessed the following way:

.. code-block:: python

    edisgo.network.results

Get voltage levels at nodes from :meth:`~.grid.network.Results.v_res`
and line loading from :meth:`~.grid.network.Results.s_res` or
:attr:`~.grid.network.Results.i_res`.
:attr:`~.grid.network.Results.equipment_changes` holds details about measures
performed during grid extension. Associated costs are determined by
:attr:`~.grid.network.Results.grid_expansion_costs`.
Flexibility measures may not entirely resolve all issues.
These unresolved issues are listed in :attr:`~.grid.network.Results.unresolved_issues`.

Results can be saved to csv files with:

.. code-block:: python

    edisgo.network.results.save('path/to/results/directory/')

To reimport saved results you can use the :class:`~.grid.network.EDisGoRemiport` class.
After instantiating the class you can access results and plots the same way as you would
with the EDisGo class.

.. code-block:: python

    # import EDisGoReimport class
    from edisgo import EDisGoReimport

    # instantiate EDisGoReimport class
    edisgo = EDisGoReimport('path/to/results/directory/')

    # access results
    edisgo.network.results.grid_expansion_costs

    # plot MV grid topology on a map
    edisgo.plot_mv_grid_topology()


