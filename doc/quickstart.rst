.. _quickstart:

Quickstart
==========

Installation
------------

Install latest eDisGo version through pip. Therefore, we highly recommend using a virtual environment and use its pip.

.. code-block:: bash

    pip3 install edisgo

In order to create plots with background maps you additionally need to install
the python package `contextily <https://github.com/darribas/contextily>`_.

Consider to install a developer version as detailed in :ref:`dev-notes`.

.. _prerequisites:

Prerequisites
-------------

Beyond a running and up-to-date installation of eDisGo you need **grid topology
data**. Currently synthetic grid data generated with the python project `Ding0 <https://github.com/openego/ding0>`_ 
is the only supported data source. You can retrieve data from `Zenodo <https://zenodo.org/record/890479>`_ 
(make sure you choose latest data) or check
out the ding0 documentation on how to generate grids yourself.

Aside from grid topology data you may eventually need a dataset on future installation of power plants. You
may therefore use the scenarios developed in the `open_eGo <https://openegoproject.wordpress.com>`_ that
are available in `OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/>`_ hosted on the `OpenEnergy Platform (OEP) <https://oep.iks.cs.ovgu.de/>`_. 
eDisGo provides an interface to the oedb using the package `ego.io <https://github.com/openego/ego.io>`_. ego.io gives you a python SQL-Alchemy representations of
the oedb and access to it by using the `oedialect <https://github.com/openego/oedialect>`_, an SQL-Alchemy dialect used by the OEP. 

To retrieve data from the oedb you need to create an account `here <http://openenergy-platform.org/login/>`_.
Upon retrieving data with eDisGo from the oedb for the first time you will be asked to type in your login information.
If you wish so, your login data will be saved to the folder ``.egoio`` to the file
``config.ini`` and your password stored in a keyring, so that you don't need to type it in every time you retrieve data. 
The ``config.ini`` holds the following information:

.. code-block:: bash

  [oedb]
  dialect  = oedialect
  username = <username>
  database = oedb
  host     = openenergy-platform.org
  port     = 80

.. _edisgo-mwe:

A minimum working example
-------------------------

Following you find short examples on how to use eDisGo. Further examples and details are provided in :ref:`usage-details`.

All following examples assume you have a ding0 grid topology file named "ding0_grids__42.pkl" in current working directory.
You can run a worst-case scenario as follows:

Using package included command-line script

.. code-block:: bash

    edisgo_run -f ding0_grids__42.pkl -wc

Or coding the script yourself with finer control of details

.. code-block:: python

    from edisgo import EDisGo

    # Set up the EDisGo object that will import the grid topology, set up
    # feed-in and load time series (here for a worst case analysis)
    # and other relevant data
    edisgo = EDisGo(ding0_grid="ding0_grids__42.pkl",
                    worst_case_analysis='worst-case')

    # Import future generators (OEP account needed!)
    edisgo.import_generators(generator_scenario='nep2035')

    # Conduct grid analysis (non-linear power flow)
    edisgo.analyze()

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine costs for each line/transformer that was reinforced
    costs = edisgo.network.results.grid_expansion_costs


Instead of conducting a worst-case analysis you can also provide specific time series:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object with your own time series 
    # (these are dummy time series!)
    # timeindex specifies which time steps to consider in power flow
    timeindex = pd.date_range('1/1/2011', periods=4, freq='H')
    # load time series (scaled by annual demand)
    timeseries_load = pd.DataFrame({'residential': [0.0001] * len(timeindex),
				    'commercial': [0.0002] * len(timeindex),
				    'industrial': [0.0015] * len(timeindex),
                                    'agricultural': [0.00005] * len(timeindex)},
			           index=timeindex)
    # feed-in time series of fluctuating generators (scaled by nominal power)
    timeseries_generation_fluctuating = \
        pd.DataFrame({'solar': [0.2] * len(timeindex),
		      'wind': [0.3] * len(timeindex)},
		     index=timeindex)
    # feed-in time series of dispatchable generators (scaled by nominal power)
    timeseries_generation_dispatchable = \
        pd.DataFrame({'biomass': [1] * len(timeindex),
		      'coal': [1] * len(timeindex),
		      'other': [1] * len(timeindex)},
		     index=timeindex)

    edisgo = EDisGo(
        ding0_grid="ding0_grids__42.pkl",
        timeseries_load=timeseries_load,
        timeseries_generation_fluctuating=timeseries_generation_fluctuating,
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex)

    # Import future generators for NEP2035 scenario
    edisgo.import_generators(generator_scenario='nep2035')

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo.network.results.grid_expansion_costs

Time series for load and fluctuating generators can also be automatically generated
using the provided API for the oemof demandlib and the OpenEnergy DataBase:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object using the OpenEnergy DataBase and the oemof
    # demandlib to set up time series for loads and fluctuating generators
    # (time series for dispatchable generators need to be provided)
    timeindex = pd.date_range('1/1/2011', periods=4, freq='H')
    timeseries_generation_dispatchable = \
        pd.DataFrame({'other': [1] * len(timeindex)},
		     index=timeindex)
    edisgo = EDisGo(
        ding0_grid="ding0_grids__42.pkl",
        timeseries_load='demandlib',
        timeseries_generation_fluctuating='oedb',
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex)

    # Import future generators for ego100 scenario
    edisgo.import_generators(generator_scenario='ego100')

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo.network.results.grid_expansion_costs

Parallelization
---------------

Try :func:`~.edisgo.tools.edisgo_run.run_edisgo_pool_flexible` for
parallelization of your custom function.