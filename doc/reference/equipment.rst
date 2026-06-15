.. _equipment:

Equipment data
==============

These tables list the cable, overhead-line and transformer types eDisGo knows about,
together with their electrical parameters. They serve two purposes: the **existing**
equipment defines the technical limits checked during analysis (e.g. ``I_max_th``,
``S_nom``), and the **standard** types are what :ref:`grid-reinforcement` installs
when it expands the grid. The cost of each type is configured in
:ref:`config_grid_expansion`.

Besides the thermal/rating limits, the tables carry the per-km impedance parameters
used to build the network model: nominal voltage ``U_n`` (kV), resistance ``R_per_km``
(Ω/km), inductance ``L_per_km`` (mH/km) and capacitance ``C_per_km`` (µF/km) for
cables and lines, and the short-circuit voltage ``u_kr`` (%) and copper losses ``P_k``
(MW) for transformers. The units are given in the second header row of each table.

.. _lv_cables_table:

.. csv-table:: LV cables
   :file: ../../edisgo/equipment/equipment-parameters_LV_cables.csv
   :delim: ,
   :header-rows: 2
   :widths: 3, 1, 1, 1, 1, 1
   :stub-columns: 0

.. _mv_cables_table:

.. csv-table:: MV cables
   :file: ../../edisgo/equipment/equipment-parameters_MV_cables.csv
   :delim: ,
   :header-rows: 2
   :widths: 4, 1, 1, 1, 1, 1
   :stub-columns: 0

.. _mv_lines_table:

.. csv-table:: MV overhead lines
   :file: ../../edisgo/equipment/equipment-parameters_MV_overhead_lines.csv
   :delim: ,
   :header-rows: 2
   :widths: 4, 1, 1, 1, 1, 1
   :stub-columns: 0

.. _lv_transformers_table:

.. csv-table:: LV transformers
   :file: ../../edisgo/equipment/equipment-parameters_LV_transformers.csv
   :delim: ,
   :header-rows: 2
   :widths: 5, 1, 1, 1
   :stub-columns: 0

.. _mv_transformers_table:

.. csv-table:: MV transformers
   :file: ../../edisgo/equipment/equipment-parameters_MV_transformers.csv
   :delim: ,
   :header-rows: 2
   :widths: 5, 1
   :stub-columns: 0
