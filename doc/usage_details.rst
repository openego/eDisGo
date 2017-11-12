.. _usage-details:

Usage details
=============

As eDisGo is designed to serve as a toolbox, it provides several methods to
analyze distribution grids for grid issues and evaluate measures responding these.
`Examples <https://github.com/openego/eDisGo/tree/dev/edisgo/examples>`_
are provided to show a typical workflow how eDisGo can be used. See
the standard example or take a look at a
`script <https://gist.github.com/gplssm/e0fd6cb99f7e8c1eed3fd4a4e325dde0>`_
that is used to assess grid
extension cost for distribution grids in the upcoming two decades.
Further, we discuss how different features can be used in detail below.

The fundamental data structure
------------------------------

It's worth to understand how the fundamental data structure of eDisGo is
designed in order to make use of its entire features.

The class :class:`~.grid.network.Network` serves as overall container in
eDisGo. It is linked from multiple locations and provides hierarchical access
to all data.
The grid data can be accessed via

.. code-block:: python

    # MV grid instance
    Network.mv_grid

    # List of LV grid instances
    Network.lv_grids

You may be interested to access more global data like
:class:`~.grid.network.Scenario`, :class:`~.grid.network.Parameters`,
:class:`~.grid.network.TimeSeries`, :class:`~.grid.network.ETraGoSpecs` or
:class:`~.grid.network.Results`. All these are accessible through
:class:`~.grid.network.Network`.

.. code-block:: python

    # Scenario
    Network.scenario

    # Parameters
    Network.scenario.parameters

    # ETraGoSpecs
    Network.scenario.etrago_specs

    # Results
    Network.results

The grid topology is represented by an undirected graph. Each one for the MV
grid and the LV grids. The :class:`~.grid.network.Graph` is subclassed from
:networkx:`networkx.Graph<graph>` and extended by extra-functionality.
Lines represent edges in the graph. Other equipment is represented by a node.


.. todo::

    Add more
     * Add examples on accessing particular data, i.e. generators


Identify grid issues
--------------------

Use PyPSA's non-linear power flow to perform a stationary power flow analysis.

As detailed in :ref:`edisgo-mwe`, once you imported a grid topology by
:code:`Network.import_from_ding0()`, you are ready for an analysis of grid
issues (line overloading or voltage band violations) respectively the hosting
capacity of the grid by :code:`Network.analyze()`.

The range of time analyzed by the power flow analysis is defined
:class:`~.grid.network.Network`'s :class:`~.grid.network.TimeSeries` class.

A worst-case analysis can be set up by passing :code:`power_flow='worst-case'`
when instantiating a :class:`~.grid.network.Scenario` obejct

.. code-block:: python

    scenario = Scenario(power_flow='worst-case', mv_grid_id=42)

Time series spanning a defined range from zero am on the 10th of October 2011 to
12 pm on the 13th of October 2011 is defined by

.. code-block:: python

    scenario = Scenario(
                power_flow=(date(2011, 10, 10), date(2011, 10, 13)),
                mv_grid_id=42,
                scenario_name=['NEP 2035', 'Status Quo'])

The :attr:`~.grid.network.Scenario.scenario_name` is used to distinguish time
series data for wind and PV power.

Grid extension
--------------

Battery storages
----------------

Battery storages can be integrated into the grid as alternative to classical
grid extension. A battery in eDisGo is represented by the class
:class:`~.grid.components.Storage`. Its operation is defined by the associated
class :class:`~.grid.components.StorageOperation`.
In order to a storage to the grid, start from the following exemplary code

.. code-block:: python

    # define storage parameters
    storage_parameters = {'soc_initial': 0,
                          'efficiency_in': .9,
                          'efficiency_out': .9,
                          'standing_loss': 0}

    # add storage instance to the grid
    network.integrate_storage(position='hvmv_substation_busbar',
                              parameters=storage_parameters)

Using the method :meth:`~.grid.network.Network.integrate_storage` provides a
high-level interface to define the position, size and storage operation at once,
based on predefined rules. Thus, a limited set of storage integration rules are
implemented. See :func:`~.flex_opt.storage_integration.integrate_storage` for
available storage integration strategies.

.. _storage-operation:

Modes of storage operation
^^^^^^^^^^^^^^^^^^^^^^^^^^

As already mentioned the operational mode of a storage is described by its
:class:`~.grid.components.StorageOperation` instance. The mode is defined by
:meth:`~.grid.components.StorageOperation.define_timeseries`. See there for
available modes.


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
