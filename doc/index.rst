Welcome to the documentation of eDisGo!
========================================

.. figure:: images/edisgo_logo.png
   :align: right
   :scale: 70%

The python package eDisGo serves as a toolbox to evaluate flexibility measures
as an economic alternative to conventional grid expansion in
medium and low voltage grids.

The toolbox currently includes:

* Data import from external data sources

  * `ding0 <https://github.com/openego/ding0>`_ tool for synthetic medium and low
    voltage grid topologies for the whole of Germany
  * `OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/schemas>`_ for
    feed-in time series of fluctuating renewables and scenarios for future
    power plant park of Germany
  * `demandlib <https://github.com/oemof/demandlib>`_ for electrical load time series
  * `SimBEV <https://github.com/rl-institut/simbev>`_ and
    `TracBEV <https://github.com/rl-institut/tracbev>`_ for charging demand data of electric
    vehicles, respectively potential charging point locations

* Static, non-linear power flow analysis using `PyPSA <https://pypsa.org>`_ for
  grid issue identification
* Automatic grid reinforcement methodology solving overloading and voltage issues
  to determine grid expansion needs and costs based on measures most commonly
  taken by German distribution grid operators
* Implementation of different charging strategies of electric vehicles
* Multiperiod optimal power flow based on julia package PowerModels.jl optimizing
  storage positioning and/or operation (Currently not maintained)
  as well as generator dispatch with regard to minimizing grid expansion costs
* Temporal complexity reduction
* Heuristic for grid-supportive generator curtailment (Currently not maintained)
* Heuristic grid-supportive battery storage integration (Currently not maintained)

Currently, a method to optimize the flexibility that can be provided by electric
vehicles through controlled charging is being implemented.
Prospectively, demand side management and reactive power management will
be included.

See :ref:`quickstart` for the first steps.
A deeper guide is provided in :ref:`usage-details`.
Methodologies are explained in detail in :ref:`features-in-detail`.
For those of you who want to contribute see :ref:`dev-notes` and the
API reference.

eDisGo was initially developed in the
`open_eGo <https://openegoproject.wordpress.com>`_ research project as part of
a grid planning tool that can be used to determine the optimal grid and storage
expansion of the German power grid over all voltage levels and has been used in
two publications of the project:

* `Integrated Techno-Economic Power System Planning of Transmission and Distribution Grids <https://www.mdpi.com/1996-1073/12/11/2091>`_
* `Final report of the open_eGo project (in German) <https://www.uni-flensburg.de/fileadmin/content/abteilungen/industrial/dokumente/downloads/veroeffentlichungen/forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf>`_

Contents
==================

.. toctree::
   :maxdepth: 2

   quickstart
   usage_details
   features_in_detail
   dev_notes
   definitions_and_units
   configs
   equipment
   whatsnew
   genindex
