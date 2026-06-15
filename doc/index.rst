Welcome to the documentation of eDisGo!
========================================

.. figure:: images/edisgo_logo.png
   :align: right
   :scale: 70%

eDisGo (*electric Distribution Grid optimization*) is a Python toolbox to analyse
and plan **medium- and low-voltage distribution grids**. Its purpose is to evaluate
flexibility measures (controlled charging, heat-pump and storage operation, demand
side management, reactive power) as an economic alternative to — or in combination
with — conventional grid reinforcement.

What eDisGo can do
------------------

* **Import grid data and scenarios** from external sources:

  * `ding0 <https://github.com/openego/ding0>`_ — synthetic medium- and
    low-voltage grid topologies for all of Germany.
  * `OpenEnergy DataBase (oedb) / egon-data
    <https://openenergyplatform.org/database/>`_ — generator parks, load,
    heat-pump, DSM, storage and electric-vehicle data for future scenarios.
  * `demandlib <https://github.com/oemof/oemof-demand>`_ — standard electrical
    load profiles.
  * `SimBEV <https://github.com/rl-institut/simbev>`_ /
    `TracBEV <https://github.com/rl-institut/tracbev>`_ — electric-vehicle charging
    demand and potential charging-point locations.

* **Power flow analysis** — non-linear AC power flow via
  `PyPSA <https://pypsa.org>`_ to find voltage and loading problems.
* **Automatic grid reinforcement** — solves overloading and voltage issues with
  the measures German distribution grid operators commonly use, and reports the
  resulting grid-expansion costs.
* **Flexibility & optimisation** — represents electric vehicles, heat pumps,
  battery storage and demand side management as flexibilities and schedules them
  with a multi-period optimal power flow (PowerModels.jl) to minimise grid
  expansion. See :ref:`flexibility-overview`.
* **Spatial and temporal complexity reduction** for large grids.

How to read this documentation
------------------------------

The documentation is organised by how deep you want to go:

* :ref:`getting-started` — install eDisGo and run your first analysis.
* :ref:`user-guide` — task-oriented guide to the data model and every step of a
  study (data import, time series, analysis, reinforcement, results).
* :ref:`methodology` — the engineering and physics behind each method, function
  by function, including the **flexibility & optimisation** chapters.
* :ref:`tutorials` — runnable Jupyter notebooks.
* :ref:`reference` — conventions, units, configuration and equipment data, and the
  full auto-generated API reference.

eDisGo was initially developed in the
`open_eGo <https://openegoproject.wordpress.com>`_ research project as part of a
grid-planning tool spanning all voltage levels, documented in two project
publications:

* `Integrated Techno-Economic Power System Planning of Transmission and Distribution Grids <https://www.mdpi.com/1996-1073/12/11/2091>`_ (Müller et al., *Energies*, 2019)
* `Final report of the open_eGo project (in German) <https://www.uni-flensburg.de/fileadmin/content/abteilungen/industrial/dokumente/downloads/veroeffentlichungen/forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf>`_

Publications using eDisGo
-------------------------

eDisGo has since been used for the grid analysis and reinforcement-cost calculations
in the following peer-reviewed studies:

* `Challenges of top-down flexibility deployment for grid expansion across all voltage levels <https://doi.org/10.1088/2753-3751/ae2686>`_ — Büttner et al., *Environmental Research: Energy*, 2025.
* `Analyzing the Impact of Dynamic Tariff Adoption and Regulatory Options on Distribution Grids with an Open-Source Framework <https://doi.org/10.1145/3679240.3734590>`_ — Semmelmann et al., *ACM e-Energy*, 2025.
* `On the Integration of Electric Vehicles Into German Distribution Grids Through Smart Charging <https://doi.org/10.1109/tia.2024.3494777>`_ — Heider et al., *IEEE Transactions on Industry Applications*, 2024 (conference version: `SEST 2022 <https://doi.org/10.1109/sest53650.2022.9898464>`_).
* `Grid Reinforcement Costs with Increasing Penetrations of Distributed Energy Resources <https://doi.org/10.1109/powertech55446.2023.10202913>`_ — Heider et al., *IEEE PowerTech Belgrade*, 2023.
* `On the impact of heat pump installations and peak blocking strategies on grid expansion costs <https://doi.org/10.1109/isgteurope56780.2023.10407931>`_ — Semmelmann et al., *IEEE ISGT Europe*, 2023.
* `Assessing the impacts of market-oriented electric vehicle charging on German distribution grids <https://doi.org/10.1049/icp.2021.2515>`_ — Schachler et al., *CIRED 2021*.
* `Distribution System Planning with Battery Storage using Multiperiod Optimal Power Flow <https://doi.org/10.2991/ahe.k.210202.007>`_ — Pedersen et al., 2021.

The synthetic distribution grids analysed in these studies are generated with
`ding0 <https://github.com/openego/ding0>`_; an example dataset is published on
`Zenodo <https://doi.org/10.5281/zenodo.10405129>`_ (Amme et al., 2023).

.. _getting-started:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. _user-guide:

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   userguide/data_sources
   userguide/workflow
   userguide/data_model
   userguide/components_overview
   userguide/time_series
   userguide/components
   userguide/analysis_results

.. _methodology:

.. toctree::
   :maxdepth: 2
   :caption: Methodology & Physics

   methodology/power_flow
   methodology/grid_reinforcement
   methodology/flexibility/index
   methodology/complexity_reduction

.. _tutorials:

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/full_workflow_walkthrough
   tutorials/simple_example
   tutorials/electromobility_example
   tutorials/plot_example

.. _reference:

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/definitions_and_units
   reference/configs
   reference/equipment
   API Reference <autoapi/edisgo/index>
   reference/julia_api
   genindex

.. toctree::
   :maxdepth: 1
   :caption: Experimental & Legacy

   legacy/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   whatsnew
