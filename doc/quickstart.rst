.. _quickstart:

Getting started
================

Installation using Linux
-------------------------

.. warning:: Make sure to use python 3.8 or higher!

Install latest eDisGo version through pip. Therefore, we highly recommend using
a virtual environment and its pip.

.. code-block:: bash

    python -m pip install edisgo

You may also consider installing a developer version as detailed in
:ref:`dev-notes`.

Installation using Windows
--------------------------

.. warning:: Make sure to use python 3.8 or higher!

For Windows users we recommend using Anaconda and to install the geo stack
using the conda-forge channel prior to installing eDisGo. You may use the provided
`eDisGo_env.yml file <https://github.com/openego/eDisGo/blob/dev/eDisGo_env.yml>`_
to do so. Download the file and create a virtual environment with:

.. code-block:: bash

    conda env create -f path/to/eDisGo_env.yml

Activate the newly created environment with:

.. code-block:: bash

    conda activate eDisGo_env

Installation using MacOS
--------------------------

We don't have any experience with our package on MacOS yet! If you try eDisGo on MacOS
we would be happy if you let us know about your experience!

Requirements for edisgoOPF package
----------------------------------

.. warning:: The non-linear optimal power flow is currently not maintained and might not work out of the box!

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
Under https://www.hsl.rl.ac.uk/ipopt/ download the sources for Coin-HSL Full (Stable).
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

Following you find short examples on how to use eDisGo to set up a network and time
series information for loads and generators in the network and afterwards conduct a
power flow analysis and determine possible grid expansion needs and costs. Further
details are provided in :ref:`usage-details`. Further examples can be found in the
`examples directory <https://github.com/openego/eDisGo/tree/dev/examples>`_.

All following examples assume you have a ding0 grid topology (directory containing
csv files, defining the grid topology) in a directory "ding0_example_grid" in
the directory from where you run your example. If you do not have an example grid, you
can download one `here <https://github.com/openego/eDisGo/tree/dev/tests/data/ding0_test_network_2/>`_.

Aside from grid topology data you may eventually need a dataset on future
installation of power plants. You may therefore use the scenarios developed in
the `open_eGo <https://openegoproject.wordpress.com>`_ project that
are available in the
`OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/schemas>`_
hosted on the `OpenEnergy Platform (OEP) <https://openenergy-platform.org/>`_.
eDisGo provides an interface to the oedb using the package
`ego.io <https://github.com/openego/ego.io>`_. ego.io gives you a python
SQL-Alchemy representations of the oedb and access to it by using the
`oedialect <https://github.com/OpenEnergyPlatform/oedialect>`_, an SQL-Alchemy dialect
used by the OEP.

You can run a worst-case scenario as follows:

.. code-block:: python

    from edisgo import EDisGo

    # Set up the EDisGo object - the EDisGo object provides the top-level API for
    # invocation of data import, power flow analysis, network reinforcement,
    # flexibility measures, etc..
    edisgo_obj = EDisGo(ding0_grid="ding0_example_grid")

    # Import scenario for future generator park from the oedb
    edisgo_obj.import_generators(generator_scenario="nep2035")

    # Set up feed-in and load time series (here for a worst case analysis)
    edisgo_obj.set_time_series_worst_case_analysis()

    # Conduct power flow analysis (non-linear power flow using PyPSA)
    edisgo_obj.analyze()

    # Do grid reinforcement
    edisgo_obj.reinforce()

    # Determine costs for each line/transformer that was reinforced
    costs = edisgo_obj.results.grid_expansion_costs


Instead of conducting a worst-case analysis you can also provide specific
time series:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object with generator park scenario NEP2035
    edisgo_obj = EDisGo(
        ding0_grid="ding0_example_grid",
        generator_scenario="nep2035"
    )

    # Set up your own time series by load sector and generator type (these are dummy
    # time series!)
    timeindex = pd.date_range("1/1/2011", periods=4, freq="H")
    # load time series (scaled by annual demand)
    timeseries_load = pd.DataFrame(
        {"residential": [0.0001] * len(timeindex),
         "cts": [0.0002] * len(timeindex),
         "industrial": [0.00015] * len(timeindex),
         "agricultural": [0.00005] * len(timeindex)
         },
        index=timeindex)
    # feed-in time series of fluctuating generators (scaled by nominal power)
    timeseries_generation_fluctuating = pd.DataFrame(
        {"solar": [0.2] * len(timeindex),
         "wind": [0.3] * len(timeindex)
         },
        index=timeindex)
    # feed-in time series of dispatchable generators (scaled by nominal power)
    timeseries_generation_dispatchable = pd.DataFrame(
        {"biomass": [1] * len(timeindex),
         "coal": [1] * len(timeindex),
         "other": [1] * len(timeindex)
         },
        index=timeindex)

    # Before you can set the time series to the edisgo_obj you need to set the time
    # index (this could also be done upon initialisation of the edisgo_obj) - the time
    # index specifies which time steps to consider in power flow analysis
    edisgo_obj.set_timeindex(timeindex)

    # Now you can set the active power time series of loads and generators in the grid
    edisgo_obj.set_time_series_active_power_predefined(
        conventional_loads_ts=timeseries_load,
        fluctuating_generators_ts=timeseries_generation_fluctuating,
        dispatchable_generators_ts=timeseries_generation_dispatchable
    )

    # Before you can now run a power flow analysis and determine grid expansion needs,
    # reactive power time series of the loads and generators also need to be set. If you
    # simply want to use default configurations, you can do the following.
    edisgo_obj.set_time_series_reactive_power_control()

    # Now you are ready to determine grid expansion needs
    edisgo_obj.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo_obj.results.grid_expansion_costs

Time series for loads and fluctuating generators can also be automatically generated
using the provided API for the oemof demandlib and the OpenEnergy DataBase:

.. code-block:: python

    import pandas as pd
    from edisgo import EDisGo

    # Set up the EDisGo object with generator park scenario NEP2035 and time index
    timeindex = pd.date_range("1/1/2011", periods=4, freq="H")
    edisgo_obj = EDisGo(
        ding0_grid="ding0_example_grid",
        generator_scenario="nep2035",
        timeindex=timeindex
    )

    # Set up your own time series by load sector and generator type (these are dummy
    # time series!)
    # Set up active power time series of loads and generators in the grid using prede-
    # fined profiles per load sector and technology type
    # (There are currently no predefined profiles for dispatchable generators, wherefore
    # their feed-in profiles need to be provided)
    timeseries_generation_dispatchable = pd.DataFrame(
        {"biomass": [1] * len(timeindex),
         "coal": [1] * len(timeindex),
         "other": [1] * len(timeindex)
         },
        index=timeindex
    )
    edisgo_obj.set_time_series_active_power_predefined(
        conventional_loads_ts="demandlib",
        fluctuating_generators_ts="oedb",
        dispatchable_generators_ts=timeseries_generation_dispatchable
    )

    # Before you can now run a power flow analysis and determine grid expansion needs,
    # reactive power time series of the loads and generators also need to be set. Here,
    # default configurations are again used.
    edisgo_obj.set_time_series_reactive_power_control()

    # Do grid reinforcement
    edisgo_obj.reinforce()

    # Determine cost for each line/transformer that was reinforced
    costs = edisgo_obj.results.grid_expansion_costs

LICENSE
-------

Copyright (C) 2018 Reiner Lemoine Institut gGmbH

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/.
