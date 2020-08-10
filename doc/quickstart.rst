.. _quickstart:

Quickstart
==========

Installation
------------

Install latest eDisGo version through pip. Therefore, we highly recommend using
a virtual environment and use its pip.

.. code-block:: bash

    pip3 install edisgo

The above will install all packages for the basic usage of eDisGo. To install
additional packages e.g. needed to create plots with background maps or to run
the jupyter notebook examples, we provide installation with extra packages:

.. code-block:: bash

    pip3 install edisgo[geoplot]  # for plotting with background maps
    pip3 install edisgo[examples]  # to run examples
    pip3 install edisgo[dev]  # developer packages
    pip3 install edisgo[full]  # combines all of the extras above

You may also consider installing a developer version as detailed in
:ref:`dev-notes`.

Requirements for edisgoOPF package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the multiperiod optimal power flow that is provided in the julia package
edisgoOPF in eDisGo you additionally need to install julia version 1.1.1.
Download julia from
`julia download page <https://julialang.org/downloads/oldreleases/>`_ and
add it to your path (see
`platform specific instructions <https://julialang.org/downloads/platform/>`_
for more information).

Before using the edisgoOPF julia package for the first time you need to
instantiate it. Therefore, in a terminal change directory to the edisgoOPF
package located in eDisGo/edisgo/opf/edisgoOPF and call julia from there.
Change to package mode by typing

.. code-block:: bash

    ]

Then activate the package:

.. code-block:: bash

    (v1.0) pkg> activate .

And finally instantiate it:

.. code-block:: bash

    (SomeProject) pkg> instantiate

.. _prerequisites:

Additional linear solver
^^^^^^^^^^^^^^^^^^^^^^^^^

As with the default linear solver in Ipopt (local solver used in the OPF)
the limit for prolem sizes is reached quite quickly, you may want to instead use
the solver HSL_MA97.
The steps required to set up HSL  are also described in the
`Ipopt Documentation <https://coin-or.github.io/Ipopt/INSTALL.html#DOWNLOAD_HSL>`_.
Here is a short version for reference:

First, you need to obtain an academic license for HSL Solvers.
Under http://www.hsl.rl.ac.uk/ipopt/ download the sources for Coin-HSL Full (Stable).
You will need to provide an institutional e-mail to gain access.

Unpack the tar.gz:

.. code-block:: bash

    tar -xvzf coinhsl-2014.01.10.tar.gz

To install the solver, clone the Ipopt Third Party HSL tools:

.. code-block:: bash

    git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
    cd ThirdParty-HSL


Under `ThirdParty-HSL`, create a folder for the HSL sources named `coinhsl` and
copy the contents of the HSL archive into it.
Under Ubuntu, you'll need BLAS, LAPACK and GCC for Fortran. If you don't have them, install them via:

.. code-block:: bash

    sudo apt-get install libblas-dev liblapack-dev gfortran

You can then configure and install your HSL Solvers:

.. code-block:: bash

    ./configure
    make
    sudo make install

To make Ipopt pick up the solver, you need to add it to your path.
During install, there will be an output that tells you where the libraries have
been put. Usually like this:

.. code-block:: bash

    Libraries have been installed in:
        /usr/local/lib


Add this path to the variable `LD_LIBRARY_PATH`:

.. code-block:: bash

    export LD_LIBRARY="/usr/local/bin":$LD_LIBRARY_PATH

You might also want to add this to your .bashrc to make it persistent.

For some reason, Ipopt looks for a library named `libhsl.so`, which is not what
the file is named, so we'll also need to provide a symlink:

.. code-block:: bash

    cd /usr/local/lib
    ln -s libcoinhsl.so libhsl.so

MA97 should now work and can be called from Julia with:

.. code-block:: julia

    JuMP.setsolver(pm.model,IpoptSolver(linear_solver="ma97"))

Prerequisites
-------------

Beyond a running and up-to-date installation of eDisGo you need **grid topology
data**. Currently synthetic grid data generated with the python project
`Ding0 <https://github.com/openego/ding0>`_
is the only supported data source. You can retrieve data from
`Zenodo <https://zenodo.org/record/890479>`_
(make sure you choose latest data) or check out the
`Ding0 documentation <https://dingo.readthedocs.io/en/dev/usage_details.html#ding0-examples>`_
on how to generate grids yourself.

.. _edisgo-mwe:

A minimum working example
-------------------------

Following you find short examples on how to use eDisGo. Further details are
provided in :ref:`usage-details`. Further examples can be found in the
`examples directory <https://github.com/openego/eDisGo/tree/features/refactoring/examples>`_.

All following examples assume you have a ding0 grid topology (directory containing
csv files, defining the grid topology) in a directory "ding0_example_grid" in
the directory from where you run your example.

Aside from grid topology data you may eventually need a dataset on future
installation of power plants. You may therefore use the scenarios developed in
the `open_eGo <https://openegoproject.wordpress.com>`_ project that
are available in the
`OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/>`_
hosted on the `OpenEnergy Platform (OEP) <https://oep.iks.cs.ovgu.de/>`_.
eDisGo provides an interface to the oedb using the package
`ego.io <https://github.com/openego/ego.io>`_. ego.io gives you a python
SQL-Alchemy representations of the oedb and access to it by using the
`oedialect <https://github.com/openego/oedialect>`_, an SQL-Alchemy dialect
used by the OEP.

You can run a worst-case scenario as follows:

.. code-block:: python

    from edisgo import EDisGo

    # Set up the EDisGo object that will import the grid topology, set up
    # feed-in and load time series (here for a worst case analysis)
    # and other relevant data
    edisgo = EDisGo(ding0_grid='ding0_example_grid',
                    worst_case_analysis='worst-case')

    # Import scenario for future generators from the oedb
    edisgo.import_generators(generator_scenario='nep2035')

    # Conduct grid analysis (non-linear power flow using PyPSA)
    edisgo.analyze()

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine costs for each line/transformer that was reinforced
    costs = edisgo.results.grid_expansion_costs


Instead of conducting a worst-case analysis you can also provide specific
time series:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object with your own time series 
    # (these are dummy time series!)
    # timeindex specifies which time steps to consider in power flow
    timeindex = pd.date_range('1/1/2011', periods=4, freq='H')
    # load time series (scaled by annual demand)
    timeseries_load = pd.DataFrame(
        {'residential': [0.0001] * len(timeindex),
         'retail': [0.0002] * len(timeindex),
         'industrial': [0.00015] * len(timeindex),
         'agricultural': [0.00005] * len(timeindex)
         },
        index=timeindex)
    # feed-in time series of fluctuating generators (scaled by nominal power)
    timeseries_generation_fluctuating = pd.DataFrame(
        {'solar': [0.2] * len(timeindex),
         'wind': [0.3] * len(timeindex)
         },
        index=timeindex)
    # feed-in time series of dispatchable generators (scaled by nominal power)
    timeseries_generation_dispatchable = pd.DataFrame(
        {'biomass': [1] * len(timeindex),
         'coal': [1] * len(timeindex),
         'other': [1] * len(timeindex)
         },
        index=timeindex)

    # Set up the EDisGo object with your own time series and generator scenario
    # NEP2035
    edisgo = EDisGo(
        ding0_grid='ding0_example_grid',
        generator_scenario='nep2035',
        timeseries_load=timeseries_load,
        timeseries_generation_fluctuating=timeseries_generation_fluctuating,
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex)

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo.results.grid_expansion_costs

Time series for loads and fluctuating generators can also be automatically generated
using the provided API for the oemof demandlib and the OpenEnergy DataBase:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object using the OpenEnergy DataBase and the oemof
    # demandlib to set up time series for loads and fluctuating generators
    # (time series for dispatchable generators need to be provided)
    timeindex = pd.date_range('1/1/2011', periods=4, freq='H')
    timeseries_generation_dispatchable = pd.DataFrame(
        {'biomass': [1] * len(timeindex),
         'coal': [1] * len(timeindex),
         'other': [1] * len(timeindex)
         },
        index=timeindex)

    edisgo = EDisGo(
        ding0_grid='ding0_example_grid',
        generator_scenario='ego100',
        timeseries_load='demandlib',
        timeseries_generation_fluctuating='oedb',
        timeseries_generation_dispatchable=timeseries_generation_dispatchable,
        timeindex=timeindex)

    # Do grid reinforcement
    edisgo.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo.results.grid_expansion_costs

Parallelization
---------------

Try :func:`~.edisgo.tools.edisgo_run.run_edisgo_pool_flexible` for
parallelization of your custom function.
