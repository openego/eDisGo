.. _quickstart:

Quickstart
==========

Installation
------------

Install latest eDisGo version through pip. Therefore, we highly recommend using a virtual environment and use its pip.

.. code-block:: bash

    pip3 install edisgo

Having trouble to install eDisGo dependencies, please also consult the `Ding0
installation notes <https://dingo.readthedocs.io/en/dev/getting_started.html>`_.

Consider to install a developer version as detailed in :ref:`dev-notes`.

.. _prerequisites:

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
`available on the OEP <https://oep.iks.cs.ovgu.de/>`_. It is linked to eDisGo
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

Assuming you have file name "ding0_grids__42.pkl" in current working directory run a worst-case scenario as follows:

Using package included command-line script

.. code-block:: bash

    edisgo_run -f ding0_grids__42.pkl -wc

Or coding the script yourself with finer control of details

.. code-block:: python

    from edisgo import EDisGo

    # Set up the EDisGo object that will import the grid topology, set up
    # feed-in and load time series (here for a feed-in worst case analysis)
    # and other relevant data
    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    worst_case_analysis='worst-case-feedin')

    # Import future generators
    edisgo.import_generators(generator_scenario='nep2035')

    # Do non-linear power flow analysis with PyPSA
    edisgo.analyze()

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine costs for each line/transformer that was reinforced
    costs = edisgo.network.results.grid_expansion_costs


If you want to provide eTraGo specifications:

.. code-block:: python

    import pandas as pd
    from edisgo.grid.network import ETraGoSpecs

    # Define eTraGo specs
    timeindex = pd.date_range('1/1/1970', periods=4, freq='H')
    etrago_specs = ETraGoSpecs(
	conv_dispatch=pd.DataFrame({'biomass': [1] * len(timeindex),
				    'coal': [1] * len(timeindex),
				    'other': [1] * len(timeindex)},
			           index=timeindex),
	ren_dispatch=pd.DataFrame({'0': [0.2] * len(timeindex),
			           '1': [0.3] * len(timeindex),
			           '2': [0.4] * len(timeindex),
			           '3': [0.5] * len(timeindex)},
			          index=timeindex),
        curtailment=pd.DataFrame({'0': [0.0] * len(timeindex),
			          '1': [0.0] * len(timeindex),
			          '2': [0.1] * len(timeindex),
			          '3': [0.1] * len(timeindex)},
			         index=timeindex),
	renewables=pd.DataFrame({
	    'name': ['wind', 'wind', 'solar', 'solar'],
	    'w_id': ['1', '2', '1', '2'],
	    'ren_id': ['0', '1', '2', '3']}, index=[0, 1, 2, 3]),
	battery_capacity=100,
	battery_active_power=pd.Series(data=[50, 20, -10, 20],
			               index=timeindex),
        ding0_grid="ding0_grids__42.pkl")

    # Get EDisGo API object
    edisgo = etrago_specs.edisgo

    # Import future generators
    edisgo.import_generators(generator_scenario='nep2035')

    # Do non-linear power flow analysis with PyPSA
    edisgo.analyze()

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo.network.results.grid_expansion_costs

Parallelization
---------------

Try :func:`~.edisgo.tools.edisgo_run.run_edisgo_pool_flexible` for
parallelization of your custom function.