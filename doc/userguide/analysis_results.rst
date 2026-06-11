.. _analysis-results:

Analysis, results, plots and I/O
================================

Once the grid and its time series are set up, you analyse the grid, reinforce it,
inspect the results, visualise them and save your work.

Power flow analysis
-------------------

:meth:`~edisgo.edisgo.EDisGo.analyze` runs a non-linear AC power flow with PyPSA and
identifies voltage and loading problems:

.. code-block:: python

    edisgo.analyze()                 # all time steps, MV and LV
    edisgo.analyze(timesteps=ts)     # a subset of the time index
    edisgo.analyze(mode="mv")        # only the MV grid

By default all time steps in ``edisgo.timeseries.timeindex`` are analysed. The
``troubleshooting_mode`` argument helps with convergence problems. The method is
explained in :ref:`power-flow-methodology`.

Grid reinforcement
------------------

:meth:`~edisgo.edisgo.EDisGo.reinforce` solves overloading and voltage problems by
applying the reinforcement measures described in :ref:`grid-reinforcement`:

.. code-block:: python

    edisgo.reinforce()                       # reinforce everything
    edisgo.reinforce(mode="mvlv")            # MV + stations
    edisgo.reinforce(copy_grid=True)         # only compute needs, keep topology
    edisgo.reinforce(catch_convergence_problems=True)

Results
-------

Results live in the :class:`~edisgo.network.results.Results` object:

.. code-block:: python

    edisgo.results.v_res                  # bus voltages from the power flow
    edisgo.results.s_res                  # apparent power on lines/transformers
    edisgo.results.i_res                  # currents
    edisgo.results.equipment_changes      # lines/transformers added during reinforcement
    edisgo.results.grid_expansion_costs   # cost per measure (kEUR)
    edisgo.results.unresolved_issues      # problems flexibility/expansion could not solve

How costs are computed is described in :ref:`grid-expansion-costs`.

Plots
-----

eDisGo ships a range of plots — static (matplotlib) and interactive
(plotly/dash). A few common ones:

.. code-block:: python

    edisgo.plot_mv_grid_topology()          # MV topology on a map
    edisgo.plot_mv_grid()                   # general MV plot (all mv_grid_topology options)
    edisgo.plot_mv_grid_expansion_costs()   # expansion costs on a map
    edisgo.plot_mv_line_loading()           # line loading
    edisgo.plot_mv_voltages()               # node voltages
    edisgo.plot_mv_storage_integration()    # positions of integrated storage units
    edisgo.histogram_voltage()              # voltage histogram
    edisgo.histogram_relative_line_load()   # line-loading histogram
    edisgo.plot_plotly()                    # interactive single-grid plot
    edisgo.plot_dash()                      # interactive comparison dashboard

To inspect the **voltage profile along a feeder**,
:meth:`~edisgo.edisgo.EDisGo.plot_voltage_over_dist` plots the LV voltage over the
distance to the MV/LV transformer for one LV grid, and
:meth:`~edisgo.edisgo.EDisGo.plot_voltage_over_dist_mv` does the same for the MV grid
relative to the HV/MV station.

See the :doc:`../tutorials/plot_example` notebook and the
:class:`~edisgo.edisgo.EDisGo` API for all plotting options.

Saving, loading and conversion
------------------------------

Save the whole object (topology, time series, results) to CSV, a ZIP archive,
pickle or JSON:

.. code-block:: python

    edisgo.save("path/to/dir", save_topology=True, save_timeseries=True, save_results=True)
    edisgo.save_edisgo_to_pickle()
    edisgo.save_edisgo_to_json()

Reload with :func:`~edisgo.edisgo.import_edisgo_from_files` or
:func:`~edisgo.edisgo.import_edisgo_from_pickle`.

The grid can also be converted to other representations:

.. code-block:: python

    edisgo.to_pypsa()        # PyPSA Network (power flow / LOPF)
    edisgo.to_powermodels()  # PowerModels.jl input (optimisation)
    edisgo.to_graph()        # networkx Graph

Memory use on large grids can be reduced with
:meth:`~edisgo.edisgo.EDisGo.reduce_memory` and
:meth:`~edisgo.edisgo.EDisGo.spatial_complexity_reduction` (see
:ref:`complexity-reduction`).
