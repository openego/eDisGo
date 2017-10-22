.. _usage-details:

Usage details
=============

As eDisGo is designed to serve as a toolbox, it provides several methods to
analyze distribution grids for grid issues and evaluate measures responding these.
`Examples <https://github.com/openego/eDisGo/tree/dev/edisgo/examples>`_
are provided to show a typical workflow how eDisGo can be used. See
the standard example or take a look at a
`script <https://gist.github.com/gplssm/14d3f1305447ff91574cd89c53cbcd7c>`_
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
     * May add examples on accessing particular data, i.e. generators


Identify grid issues
--------------------

Use PyPSA's non-linear power flow to perform a stationary power flow analysis.

Once you

Grid extension
--------------

.. Battery storages
.. ----------------

.. Curtailment
.. -----------

Retrieve results
----------------
