.. _quickstart:

Quickstart
==========

This page gets you from a fresh installation to your first grid analysis in a few
minutes. It assumes eDisGo is installed (see :ref:`installation`) and that you have
a ding0 grid (a directory of CSV files describing the grid topology — see
:ref:`grid-data-prerequisite`). A small test grid ships with eDisGo under
``tests/data/ding0_test_network_2``.

.. _edisgo-mwe:

The five-minute example
-----------------------

The shortest meaningful study is a **worst-case analysis**: eDisGo builds the two
classic grid-planning situations (heavy load / reverse feed-in), runs a power flow
and reinforces the grid until all voltage and loading limits are met.

.. code-block:: python

    from edisgo import EDisGo

    # 1. Load a ding0 grid. The EDisGo object is the top-level API.
    edisgo = EDisGo(ding0_grid="path/to/ding0_grid")

    # 2. Add a future generator park (optional, from the OpenEnergy DataBase).
    edisgo.import_generators(generator_scenario="nep2035")

    # 3. Create worst-case load and feed-in time series.
    edisgo.set_time_series_worst_case_analysis()

    # 4. Run a non-linear power flow to find voltage and loading problems.
    edisgo.analyze()

    # 5. Reinforce the grid to solve those problems.
    edisgo.reinforce()

    # 6. Read the resulting grid-expansion costs (in kEUR).
    costs = edisgo.results.grid_expansion_costs

That is the whole core loop: **load → time series → analyze → reinforce → results**.
Every other feature plugs into this loop. The canonical order in which the steps
must be called (and the typical pitfalls) is summarised in :ref:`workflow`.

The guided walkthrough notebook
-------------------------------

For a complete, realistic study — loading a real ding0 grid, pulling scenario data
from the OpenEnergy Platform, adding electric vehicles, heat pumps, demand side
management and storage, optimising their operation and then reinforcing — work
through the guided notebook, which builds the full workflow up in seven stages:

* :doc:`tutorials/full_workflow_walkthrough`

If you prefer reading over running, the same steps are explained in prose in the
:ref:`user-guide`, and the engineering behind them in :ref:`methodology`.

Where to go next
----------------

* :ref:`data-sources` — where grid and scenario data come from and how to access
  the OpenEnergy Platform.
* :ref:`workflow` — the canonical call order and common-error checklist.
* :ref:`flexibility-overview` — using flexibilities instead of grid expansion.
* :doc:`tutorials/simple_example`, :doc:`tutorials/electromobility_example`,
  :doc:`tutorials/plot_example` — focused example notebooks.
