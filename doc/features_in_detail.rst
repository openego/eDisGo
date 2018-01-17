.. _features-in-detail:

Features in detail
==================

Data import
-----------

Power flow analysis
-------------------


Automatic grid expansion
-------------------------

General methodology
^^^^^^^^^^^^^^^^^^^^^^^^^^

The grid expansion methodology is conducted in :py:mod:`~edisgo.flex_opt.reinforce_grid`.

For now only a combined analysis of MV and LV grids is possible.
The order grid expansion measures are conducted is as follows:

* Reinforce transformers and lines due to over-loading issues
* Reinforce lines in MV grid due to over-voltage issues
* Reinforce lines in LV grid due to over-loading issues
* Reinforce transformers and lines due to over-loading issues

Reinforcement of transformers and lines due to over-loading issues is performed twice, once in the beginning and again after fixing over-voltage problems,
because the changed power flows after reinforcing the grid may lead to new over-loading issues.


Identification of over-voltage and over-loading issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identification of over-voltage and over-loading issues is conducted in :py:mod:`~edisgo.flex_opt.check_tech_constraints`.

Check LV and MV line load
"""""""""""""""""""""""""""""""""""""

  Uses a given load factor and the maximum allowed current given by the manufacturer to calculate the allowed
  line load of each LV and MV line. If the line load calculated in the power flow analysis exceeds the allowed line load the line is reinforced (see :ref:`grid-expansion-measure-line-load-label`).

Check HV/MV station load
"""""""""""""""""""""""""""""""""""""

  Uses a given load factor and the maximum allowed apparent power given by the manufacturer to calculate the allowed
  apparent power of the HV/MV station. If the apparent power calculated in the power flow analysis exceeds the allowed apparent power the station is reinforced (see :ref:`grid-expansion-measure-mv-station-load-label`).

Check MV/LV station load
"""""""""""""""""""""""""""""""""""""

  Uses a given load factor and the maximum allowed apparent power given by the manufacturer to calculate the allowed
  apparent power of each MV/LV station. If the apparent power calculated in the power flow analysis exceeds the allowed apparent power the station is reinforced (see :ref:`grid-expansion-measure-lv-station-load-label`).

Check line and station voltage deviation
""""""""""""""""""""""""""""""""""""""""""

  Uses a given allowed voltage deviation. If the voltage of an LV or MV line calculated in the power flow analysis exceeds the allowed voltage deviation the line is reinforced (see :ref:`grid-expansion-measure-lv-station-voltage-label` or
  :ref:`grid-expansion-measure-line-voltage-label`)


Grid expansion measures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Reinforcement measures are conducted in :py:mod:`~edisgo.flex_opt.check_tech_constraints`.

.. _grid-expansion-measure-line-load-label:

Reinforce lines due to over-loading
"""""""""""""""""""""""""""""""""""""

  In a first step a parallel line of the same line type is installed. If this does not solve the over-loading issue as many parallel standard lines as needed are installed.

.. _grid-expansion-measure-mv-station-load-label:

Reinforce HV/MV station due to over-loading issues
"""""""""""""""""""""""""""""""""""""""""""""""""""""
 
  In a first step a parallel transformer of the same type as the existing transformer is installed. If there is more than one transformer in the station the smallest transformer
  that will solve the over-loading issue is used. If this does not solve the over-loading issue as many parallel standard transformers as needed are installed.

.. _grid-expansion-measure-lv-station-load-label:

Reinforce MV/LV station due to over-loading issues
"""""""""""""""""""""""""""""""""""""""""""""""""""""

  In a first step a parallel transformer of the same type as the existing transformer is installed. If there is more than one transformer in the station the smallest transformer
  that will solve the over-loading issue is used. If this does not solve the over-loading issue as many parallel standard transformers as needed are installed.

.. _grid-expansion-measure-lv-station-voltage-label:

Reinforce MV/LV station due to over-voltage issues
"""""""""""""""""""""""""""""""""""""""""""""""""""""

  A parallel standard transformer is installed. Afterwards a power flow analysis is conducted and the voltage is rechecked. If there are still voltage issues the process of installing
  a parallel standard transformer and conducting a power flow analysis is repeated until voltage issues are solved.

.. _grid-expansion-measure-line-voltage-label:

Reinforce lines due to over-voltage
"""""""""""""""""""""""""""""""""""""""""""""""""""""

  In the case of several voltage problems the path to the node with the highest voltage deviation is reinforced first. Therefore, the line between the secondary side of the station and the 
  node with the highest voltage deviation is disconnected at a distribution substation after 2/3 of the path length. If there is no distribution substation where the line can be
  disconnected, the node is directly connected to the busbar. If the node is already directly connected to the busbar a parallel standard line is installed.
 
  Only one voltage problem for each main route is considered at a time since each measure effects the voltage of each node in that route.

  After each main route with voltage problems has been considered a power flow analysis is conducted and the voltage rechecked. The process of solving voltage issues is repeated until voltage issues are solved
  or until the maximum number of allowed iterations is reached.




Grid expansion costs
^^^^^^^^^^^^^^^^^^^^^^^^^^

Total grid expansion costs are the sum of costs for each added transformer and line.
Costs for lines and transformers are only distinguished by the voltage level they are installed in 
and not by the different types. 
In the case of lines it is further taken into account wether the line is installed in a rural or an urban area whereas rural areas
are areas with a population density smaller or equal to 500 people per km² and urban areas are defined as areas
with a population density higher than 500 people per km² [DENA]_. 
The population density is calculated by the population and area of the grid district the line is in (See :class:`~.grid.grids.Grid`).

Costs for lines of aggregated loads and generators are not considered in the costs calculation since grids of
aggregated areas are not modeled but aggregated loads and generators are directly connected to the MV busbar.

References
----------

.. [DENA] A.C. Agricola et al.:
    *dena-Verteilnetzstudie: Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland bis 2030*. 2012.
