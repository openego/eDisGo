.. figure:: images/edisgo_logo.png
   :align: right
   :scale: 70%

eDisGo
======

The python package eDisGo serves as a toolbox to evaluate flexibility measures
as an economic alternative to conventional grid expansion in
medium and low voltage grids.

The toolbox currently includes:

* Data import from external data sources

  * `ding0 <https://github.com/openego/ding0>`_ tool for synthetic medium and low
    voltage grid topologies for the whole of Germany
  * `OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/>`_ for
    feed-in time series of fluctuating renewables and scenarios for future
    power plant park of Germany
  * `demandlib <https://github.com/oemof/demandlib>`_ for electrical load time series

* Static, non-linear power flow analysis using `PyPSA <https://pypsa.org>`_ for
  grid issue identification
* Automatic grid reinforcement methodology solving overloading and voltage issues
  to determine grid expansion needs and costs based on measures most commonly
  taken by German distribution grid operators
* Multiperiod optimal power flow based on julia package PowerModels.jl optimizing
  storage positioning and/or operation
  as well as generator dispatch with regard to minimizing grid expansion costs
* Temporal complexity reduction
* Heuristic for grid-supportive generator curtailment
* Heuristic grid-supportive battery storage integration

Currently, a method to optimize the flexibility that can be provided by electric
vehicles through controlled charging and V2G is implemented.
Prospectively, demand side management and reactive power management will
be included.

See :ref:`quickstart` for the first steps.
A deeper guide is provided in :ref:`usage-details`.
Methodologies are explained in detail in :ref:`features-in-detail`.
For those of you who want to contribute see :ref:`dev-notes` and the
:ref:`api` reference.

eDisGo was initially developed in the
`open_eGo <https://openegoproject.wordpress.com>`_ research project as part of
a grid planning tool that can be used to determine the optimal grid and storage
expansion of the German power grid over all voltage levels and has been used in
two publications of the project:

* `Integrated Techno-Economic Power System Planning of Transmission and Distribution Grids <https://www.mdpi.com/1996-1073/12/11/2091>`_
* `Final report of the open_eGo project (in German) <https://www.uni-flensburg.de/fileadmin/content/abteilungen/industrial/dokumente/downloads/veroeffentlichungen/forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf>`_

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