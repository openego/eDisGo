.. _analysis-results:

Analysis, results, plots and I/O
================================

Once the grid and its time series are set up, you analyse the grid, reinforce it,
inspect the results, visualise them and save your work.

Power flow analysis
-------------------

:meth:`~edisgo.edisgo.EDisGo.analyze` runs a static, non-linear AC power flow with
PyPSA and writes the results (active, reactive and apparent power as well as currents
on lines and voltages at buses) to the :class:`~edisgo.network.results.Results`
object. The identification of overloading and voltage problems happens in
:meth:`~edisgo.edisgo.EDisGo.reinforce` (via ``flex_opt.check_tech_constraints``), not
in ``analyze``:

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
    # only compute needs, keep the original topology unchanged: the costs and
    # equipment changes are in the *returned* Results object, not in edisgo.results
    results = edisgo.reinforce(copy_grid=True)
    edisgo.reinforce(catch_convergence_problems=True)

Results
-------

Results live in the :class:`~edisgo.network.results.Results` object:

.. code-block:: python

    edisgo.results.v_res                  # bus voltages from the power flow (p.u.)
    edisgo.results.s_res                  # apparent power on lines/transformers (MVA)
    edisgo.results.i_res                  # currents (kA)
    edisgo.results.pfa_p                  # active power flow on lines/transformers (MW)
    edisgo.results.pfa_q                  # reactive power flow on lines/transformers (Mvar)
    edisgo.results.equipment_changes      # lines/transformers added, changed or removed
    edisgo.results.grid_expansion_costs   # cost per expanded line/transformer (kEUR)
    edisgo.results.unresolved_issues      # issues reinforcement could not solve in max. iterations

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

The interactive plots are **module-level functions** in
:mod:`edisgo.tools.plots`, not methods of the ``EDisGo`` object:

.. code-block:: python

    from edisgo.tools.plots import plot_plotly, plot_dash

    plot_plotly(edisgo)                          # interactive single-grid plot
    plot_dash({"scenario A": edisgo_a,           # interactive comparison dashboard
               "scenario B": edisgo_b})

To inspect the **voltage profile along a feeder**,
:meth:`~edisgo.edisgo.EDisGo.plot_voltage_over_dist` plots the LV voltage over the
distance to the MV/LV transformer for one LV grid, and
:meth:`~edisgo.edisgo.EDisGo.plot_voltage_over_dist_mv` does the same for the MV grid
relative to the HV/MV station. Both can compare two EDisGo objects:
``plot_voltage_over_dist`` compares against itself by default, while
``plot_voltage_over_dist_mv`` requires a second object as its ``other`` argument.

See the :doc:`../tutorials/plot_example` notebook and the
:class:`~edisgo.edisgo.EDisGo` API for all plotting options.

Saving, loading and conversion
------------------------------

Save the whole object (topology, time series, results) to a CSV directory (optionally
as a ZIP archive) or to a pickle:

.. code-block:: python

    edisgo.save("path/to/dir", save_topology=True, save_timeseries=True, save_results=True)
    edisgo.save_edisgo_to_pickle()

Reload with :func:`~edisgo.edisgo.import_edisgo_from_files` (CSV directory / ZIP) or
:func:`~edisgo.edisgo.import_edisgo_from_pickle`.

.. note::

   :meth:`~edisgo.edisgo.EDisGo.save_edisgo_to_json` does **not** serialise the full
   EDisGo object — it exports the grid in **PowerModels network-data format** (via
   :meth:`~edisgo.edisgo.EDisGo.to_powermodels`) for the Julia OPF, and cannot be
   reloaded as an EDisGo object.

The grid can also be converted to other representations:

.. code-block:: python

    edisgo.to_pypsa()        # PyPSA Network (power flow / LOPF)
    edisgo.to_powermodels()  # PowerModels.jl input (optimisation)
    edisgo.to_graph()        # networkx Graph

Memory use on large grids can be reduced with
:meth:`~edisgo.edisgo.EDisGo.reduce_memory` and
:meth:`~edisgo.edisgo.EDisGo.spatial_complexity_reduction` (see
:ref:`complexity-reduction`).
