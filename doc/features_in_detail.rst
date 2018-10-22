.. _features-in-detail:

Features in detail
==================

Power flow analysis
-------------------

In order to analyse voltages and line loadings a non-linear power flow analysis (PF) is conducted. 
All loads and generators are modelled as PQ nodes; the slack is modelled as a PV node with a set voltage of 1\,p.u.
and positioned at the substation's secondary side.

.. _grid_expansion_methodology:

Automatic grid expansion
-------------------------

General methodology
^^^^^^^^^^^^^^^^^^^^^^^^^^

The grid expansion methodology is conducted in :py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid`.

The order grid expansion measures are conducted is as follows:

* Reinforce stations and lines due to overloading issues
* Reinforce lines in MV grid due to voltage issues
* Reinforce distribution substations due to voltage issues
* Reinforce lines in LV grid due to voltage issues
* Reinforce stations and lines due to overloading issues

Reinforcement of stations and lines due to overloading issues is performed twice, once in the beginning and again after fixing voltage issues,
as the changed power flows after reinforcing the grid may lead to new overloading issues. How voltage and overloading issues are identified and
solved is explained in the following sections.

:py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid` offers a few additional options. It is e.g. possible to conduct grid 
reinforcement measures on a copy
of the graph so that the original grid topology is not changed. It is also possible to only identify necessary
reinforcement measures for two worst-case snapshots in order to save computing time and to set combined or separate
allowed voltage deviation limits for MV and LV.
See documentation of :py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid` for more information. 


Identification of overloading and voltage issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identification of overloading and voltage issues is conducted in 
:py:mod:`~edisgo.flex_opt.check_tech_constraints`.

Voltage issues are determined based on allowed voltage deviations set in the config file 
*config_grid_expansion.cfg* in section `grid_expansion_allowed_voltage_deviations`. It is possible
to set one allowed voltage deviation that is used for MV and LV or define separate allowed voltage deviations.
Which allowed voltage deviation is used is defined through the parameter *combined_analysis* of :py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid`.
By default *combined_analysis* is set to false, resulting in separate voltage limits for MV and LV, as a combined limit
may currently lead to problems if voltage deviation in MV grid is already close to the allowed limit, in which case the remaining allowed voltage deviation in the LV grids is close to zero.

Overloading is determined based on allowed load factors that are also defined in the config file
*config_grid_expansion.cfg* in section `grid_expansion_load_factors`.

Allowed voltage deviations as well as load factors are in most cases different for load and feed-in case. 
Load and feed-in case are commonly used worst-cases for grid expansion analyses. 
Load case defines a situation where all loads in the grid have a high demand while feed-in by generators is low
or zero. In this case power is flowing from the high-voltage grid to the distribution grid. 
In the feed-in case there is a high generator feed-in and a small energy demand leading to a reversed power flow.
Load and generation assumptions for the two worst-cases are definded in the config file
`config_timeseries.cfg` in section `worst_case_scale_factor` (scale factors describe actual power
to nominal power ratio of generators and loads).

When conducting grid reinforcement based on given time series instead of worst-case assumptions, load and feed-in
case also need to be definded to determine allowed voltage deviations and load factors. 
Therefore, the two cases are identified based on the generation and load time series of all loads and generators
in the grid and defined as follows:

* Load case: positive ( :math:`\sum load` - :math:`\sum generation` ) 
* Feed-in case: negative ( :math:`\sum load` - :math:`\sum generation` ) -> reverse power flow at HV/MV substation 

Grid losses are not taken into account. See :func:`~edisgo.tools.tools.assign_load_feedin_case` for more
details and implementation.

Check line load
""""""""""""""""""

  Exceedance of allowed line load of MV and LV lines is checked in :py:func:`~edisgo.flex_opt.check_tech_constraints.mv_line_load` and
  :py:func:`~edisgo.flex_opt.check_tech_constraints.lv_line_load`, respectively.
  The functions use the given load factor and the maximum allowed current given by the manufacturer (see *I_max_th* in tables :ref:`lv_cables_table`, 
  :ref:`mv_cables_table` and :ref:`mv_lines_table`) to calculate the allowed
  line load of each LV and MV line. If the line load calculated in the power flow analysis exceeds the allowed line 
  load, the line is reinforced (see :ref:`grid-expansion-measure-line-load-label`).
  

Check station load
""""""""""""""""""""

  Exceedance of allowed station load of HV/MV and MV/LV stations is checked in :py:func:`~edisgo.flex_opt.check_tech_constraints.hv_mv_station_load` and
  :py:func:`~edisgo.flex_opt.check_tech_constraints.mv_lv_station_load`, respectively.
  The functions use the given load factor and the maximum allowed apparent power given by the manufacturer (see *S_nom* in tables :ref:`lv_transformers_table`, 
  and :ref:`mv_transformers_table`) to calculate the allowed
  apparent power of the stations. If the apparent power calculated in the power flow analysis exceeds the allowed apparent power the station is reinforced 
  (see :ref:`grid-expansion-measure-station-load-label`).

Check line and station voltage deviation
""""""""""""""""""""""""""""""""""""""""""

  Compliance with allowed voltage deviation limits in MV and LV grids is checked in :py:func:`~edisgo.flex_opt.check_tech_constraints.mv_voltage_deviation` and
  :py:func:`~edisgo.flex_opt.check_tech_constraints.lv_voltage_deviation`, respectively.
  The functions check if the voltage deviation at a node calculated in the power flow analysis exceeds the allowed voltage deviation. If it does,
  the line is reinforced (see :ref:`grid-expansion-measure-lv-station-voltage-label` or
  :ref:`grid-expansion-measure-line-voltage-label`).


Grid expansion measures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Reinforcement measures are conducted in :py:mod:`~edisgo.flex_opt.reinforce_measures`. Whereas overloading issues can usually be solved in one step, except for 
some cases where the lowered grid impedance through reinforcement measures leads to new issues, voltage issues can only be solved iteratively. This means that after each reinforcement
step a power flow analysis is conducted and the voltage rechecked. An upper limit for how many iteration steps should be performed is set in order to avoid endless iteration. By
default it is set to 10 but can be changed using the parameter *max_while_iterations* of :py:func:`~edisgo.flex_opt.reinforce_grid.reinforce_grid`.

.. _grid-expansion-measure-line-load-label:

Reinforce lines due to overloading issues
"""""""""""""""""""""""""""""""""""""""""""""

  Line reinforcement due to overloading is conducted in :py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_branches_overloading`. 
  In a first step a parallel line of the same line type is installed. If this does not solve the overloading issue as many parallel standard lines as needed are installed.

.. _grid-expansion-measure-station-load-label:

Reinforce stations due to overloading issues
"""""""""""""""""""""""""""""""""""""""""""""""""""""
 
  Reinforcement of HV/MV and MV/LV stations due to overloading is conducted in :py:func:`~edisgo.flex_opt.reinforce_measures.extend_substation_overloading` and
  :py:func:`~edisgo.flex_opt.reinforce_measures.extend_distribution_substation_overloading`, respectively. 
  In a first step a parallel transformer of the same type as the existing transformer is installed. If there is more than one transformer in the station the smallest transformer
  that will solve the overloading issue is used. If this does not solve the overloading issue as many parallel standard transformers as needed are installed.

.. _grid-expansion-measure-lv-station-voltage-label:

Reinforce MV/LV stations due to voltage issues
"""""""""""""""""""""""""""""""""""""""""""""""""""""

  Reinforcement of MV/LV stations due to voltage issues is conducted in :py:func:`~edisgo.flex_opt.reinforce_measures.extend_distribution_substation_overvoltage`. 
  To solve voltage issues, a parallel standard transformer is installed. 

  After each station with voltage issues is reinforced, a power flow analysis is conducted and the voltage rechecked. If there are still voltage issues 
  the process of installing
  a parallel standard transformer and conducting a power flow analysis is repeated until voltage issues are solved or until the maximum number of allowed iterations is reached.

.. _grid-expansion-measure-line-voltage-label:

Reinforce lines due to voltage
"""""""""""""""""""""""""""""""""""""""""""""""""""""

  Reinforcement of lines due to voltage issues is conducted in :py:func:`~edisgo.flex_opt.reinforce_measures.reinforce_branches_overvoltage`. 
  In the case of several voltage issues the path to the node with the highest voltage deviation is reinforced first. Therefore, the line between the secondary side of the station and the 
  node with the highest voltage deviation is disconnected at a distribution substation after 2/3 of the path length. If there is no distribution substation where the line can be
  disconnected, the node is directly connected to the busbar. If the node is already directly connected to the busbar a parallel standard line is installed.
 
  Only one voltage problem for each feeder is considered at a time since each measure effects the voltage of each node in that feeder.

  After each feeder with voltage problems has been considered, a power flow analysis is conducted and the voltage rechecked. The process of solving voltage issues is repeated until voltage issues are solved
  or until the maximum number of allowed iterations is reached.


Grid expansion costs
^^^^^^^^^^^^^^^^^^^^^^^^^^

Total grid expansion costs are the sum of costs for each added transformer and line.
Costs for lines and transformers are only distinguished by the voltage level they are installed in 
and not by the different types. 
In the case of lines it is further taken into account wether the line is installed in a rural or an urban area, whereas rural areas
are areas with a population density smaller or equal to 500 people per km² and urban areas are defined as areas
with a population density higher than 500 people per km² [DENA]_. 
The population density is calculated by the population and area of the grid district the line is in (See :class:`~.grid.grids.Grid`).

Costs for lines of aggregated loads and generators are not considered in the costs calculation since grids of
aggregated areas are not modeled but aggregated loads and generators are directly connected to the MV busbar.

Curtailment
-----------

Implementation
^^^^^^^^^^^^^^
The Curtailment methodology is conducted in :py:mod:`~edisgo.flex_opt.curtailment`.

The curtailment function is essentially used to spatially distribute the power required to be curtailed (henceforth
referred to as 'curtailed power') to the various generating units inside the grid. This provides a simple interface
to curtailing the power of generators of either a certain type (eg. solar or wind) or generators in a give weather
cell or both.

The current implementations of this are:

* `curtail_all`
* `curtail_voltage`

with other more complex and finer methodologies in development. The focus of these methods is to try and reduce the
requirement for network reinforcement by alleviating either node over-voltage or line loading issues or both. While
it is possible to curtail specific generators internally, a user friendly implementation is still in the works.

Concept
^^^^^^^

.. _curtailment-basic-label:

Basic curtailment
"""""""""""""""""""""""
In each of the curtailment methods, first the feedin of each of the individual fluctuating generators are
calculated based on the normalized feedin time series per technology and weather cell id from the OpenEnergy
database and the individual generators' nominal capacities.

.. math::
    feedin = feedin_{\text{normalized per type or weather cell}} \times \\
    nominal\_power_{\text{single generators}}

Once the feedin is calculated, both the feedin per generator and curtailed power are normalized and multiplied to get
the curtailed power per generator.

.. math::
    curtailment =
        \frac{feedin}{\sum feedin} \times  total\_curtailment_{\text{normalized per type or weather cell}}

This curtailment is subtracted from the feedin of the generator to obtain the power output of the generator after
curtailment.
_{\text{single generators}}

Feedin factor
"""""""""""""
The case discussed in :ref:`curtailment-basic-label` is for equally curtailing all generators
of a given type or weather cell. To induce a bias in the curtailment of the generators based on a parameter
of our choosing like voltage or line loading, we use a feedin factor, which is essentially a scalar value which
is used to modify the feedin based on this parameter.

.. math::
    modified\_feedin = feedin \times feedin\_factor

and the resulting curtailment is:

.. math::
    curtailment = \frac{modified\_feedin}{\sum modified\_feedin} \times
            total\_curtailment_{\text{normalized per type or weather cell}}

The way this influences the curtailment is that when the the feedin for a particular generator is increased by
multiplication, it results in a higher curtailment of power in this specific generator. Similarly the converse,
where when the feedin for a particular generator is reduced the curtailment for this specific generator is also
reduced. The modified feedin also allows the total curtailed power to remain the same even with the inclusion of
the biasing due to the feedin factor.

The feedin factor is only used as a weighing factor to increase or decrease the curtailment and this in no way
affects the base feedin of the generator.

Spatially biased curtailment methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Curtailment biased by node voltage
""""""""""""""""""""""""""""""""""

This is implemented in the methodology using the keyword argument :py:mod:`edisgo.flex_opt.curtailment.curtail_voltage`.
Here the feedin factor is used to bias the curtailment such that there more power is curtailed at nodes with higher
voltages and lesser power is curtailed at nodes with lower voltages. This essentially is a linear characteristic
between curtailed power and voltage, the higher the voltage, the higher the curtailed power. The characteristic is
as shown in :numref:`curtailment_voltage_characteristic_label`

A lower voltage threshold is defined, where no curtailment is assigned if the voltage at the node is lower than this
threshold voltage. The assigned curtailment to the other nodes is directly proportional to the difference of the
voltage at the node to the lower voltage threshold.


.. _curtailment_voltage_characteristic_label:
.. figure:: images/curtailment_voltage_characteristic.png

    Per unit curtailment versus per unit node voltage characteristic used under the method
    :py:mod:`edisgo.flex_opt.curtailment.curtail_voltage`




References
----------

.. [DENA] A.C. Agricola et al.:
    *dena-Verteilnetzstudie: Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland bis 2030*. 2012.
