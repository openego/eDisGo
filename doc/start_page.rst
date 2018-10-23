eDisGo
======

The python package eDisGo provides a toolbox for analysis and optimization of
distribution grids. This software lives in the context of the research project
`open_eGo <https://openegoproject.wordpress.com>`_. It is closely related to the
python project `Ding0 <https://github.com/openego/ding0>`_ as this project
is currently the single data source for eDisGo providing synthetic grid data
for whole Germany.

The toolbox currently includes

* Data import from data sources of the open_eGo project
* Power flow analysis for grid issue identification (enabled by `PyPSA <https://pypsa.org>`_)
* Grid reinforcement solving overloading and voltage issues
* Curtailment methodologies
* Battery storage integration

See :ref:`quickstart` for the first steps. A deeper guide is provided in :ref:`usage-details`.
Methodologies are explained in detail in :ref:`features-in-detail`.
For those of you who want to contribute see :ref:`dev-notes` and the
:ref:`api` reference.


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