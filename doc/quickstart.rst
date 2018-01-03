.. _quickstart:

Quickstart
==========

Installation
------------

Install latest eDisGo version through pip. Therefore, we highly recommend to
use a virtual environment and use its pip.

.. code-block:: bash

    pip3 install edisgo

Having trouble to install eDisGo dependencies, please also consult the `Ding0
installation notes <https://dingo.readthedocs.io/en/dev/getting_started.html>`_.

Consider to install a developer version as detailed in :ref:`dev-notes`.

Prerequisites
-------------

Beyond a running and up-to-date installation of eDisGo you need grid topology
data and eventually a dataset of future installations of RES.
You can retrieve data from `Zenodo <https://zenodo.org/record/890479>`_.
Make sure you choose latest data.
Aside from grid topology data stored in
`Pickles <https://docs.python.org/3/library/pickle.html>`_ (.pkl) the tar.gz
also includes a file with metadata (look for extension .meta).

A dataset on future installation of power plants (mainly extension of RES) is
`available at the OEP <https://oep.iks.cs.ovgu.de/>`_. It is linked to eDisGo
through the
`OEP API <https://oep-data-interface.readthedocs.io/en/latest/index.html>`_.

.. note::

    Currently, you still need an account on the OEDB itself. Access via the OEP
    is not implemented, yet.

.. todo::

    Describe oedb access here (mention: internet connection required)


.. _edisgo-mwe:

A minimum working example
-------------------------

Assuming you have file name "ding0_grids__42.pkl" in current working directory.

.. code-block:: python

    from edisgo.grid.network import Network, Scenario, TimeSeries

    # Define a scenario including a TimeSeries instance
    scenario = Scenario(power_flow='worst-case', mv_grid_id=grid_district)

    # Get the grid topology data
    network = Network.import_from_ding0(
        "ding0_grids__42.pkl",
        id='42',
        scenario=scenario)

    # Import future generators
    network.import_generators()

    # Do non-linear power flow analysis with PyPSA
    network.analyze()

    # Do grid reinforcement
    network.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = network.results.grid_expansion_costs


