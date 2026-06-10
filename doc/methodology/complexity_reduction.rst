.. _complexity-reduction:

Complexity reduction
====================

Large grids analysed over many time steps can be slow and memory-hungry. eDisGo can
reduce both the **spatial** size (number of buses) and the **temporal** size (number
of time steps) of a problem.

Spatial complexity reduction
----------------------------

In plain terms
~~~~~~~~~~~~~~

Spatial reduction merges nearby buses into a smaller set of representative buses,
keeping the electrical behaviour as close as possible to the original grid. This
shrinks the grid for faster power flow and optimisation.

How it works
~~~~~~~~~~~~

Call :meth:`~edisgo.edisgo.EDisGo.spatial_complexity_reduction`. The procedure has
two steps: a *busmap* is built that maps every original bus to a clustered bus, and
the eDisGo object is then reduced according to that busmap (lines are recalculated
and sometimes merged). Parts of the method are based on the spatial clustering of
[PyPSA]_.

All buses must have coordinates so that line lengths can be derived from the
Euclidean distance and a detour factor. If your grid has no coordinates, set
``apply_pseudo_coordinates=True`` (the default) to compute coordinates from the
radial grid topology.

Parameters:

* ``mode`` — the clustering method:

  * ``"kmeans"`` — assign buses to K-Means cluster centres.
  * ``"kmeansdijkstra"`` — assign to the nearest cluster centre along the grid graph
    (Dijkstra distance).
  * ``"aggregate_to_main_feeder"`` — assign to the nearest node of the main feeder
    (the longest path in the feeder).
  * ``"equidistant_nodes"`` — place nodes equidistantly along the main feeder.

* ``cluster_area`` — where clustering is applied: ``"grid"``, ``"feeder"`` or
  ``"main_feeder"``.
* ``reduction_factor`` — :math:`n_\text{buses} = k_\text{reduction}\cdot
  n_\text{buses, cluster area}`; a smaller factor means a stronger reduction.
* ``reduction_factor_not_focused`` — reduce *non-critical* areas (no voltage or
  overloading problems in the worst case) more strongly than the focus areas.

For more control you can run the underlying functions directly:

.. code-block:: python

    from edisgo.tools.spatial_complexity_reduction import make_busmap, apply_busmap
    from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates

    edisgo_obj = make_pseudo_coordinates(edisgo_obj)
    busmap_df = make_busmap(
        edisgo_obj,
        mode="kmeans",
        cluster_area="feeder",
        reduction_factor=0.25,
    )
    edisgo_reduced, linemap_df = apply_busmap(edisgo_obj, busmap_df)

See [SCR]_ and [HoerschBrown]_ for the theory.

Temporal complexity reduction
-----------------------------

The number of analysed time steps can be reduced by selecting only the
grid-critical steps, or by clustering representative steps. This is used internally
by :meth:`~edisgo.edisgo.EDisGo.reinforce` (``reduced_analysis=True``) and by the
optimisation; the selection functions live in
``edisgo.opf.timeseries_reduction``.

Memory
------

:meth:`~edisgo.edisgo.EDisGo.reduce_memory` downcasts the stored time-series and
results DataFrames to smaller dtypes, lowering memory use without changing the grid.

References
----------

.. [PyPSA] `PyPSA — Spatial Clustering documentation
   <https://docs.pypsa.org/v0.35.1/examples/spatial-clustering.html>`_

.. [SCR] Malte Jahn: *Analysis of the effects of spatial complexity reduction on
   distribution network expansion planning with flexibilities* (master's thesis, in
   German).

.. [HoerschBrown] Jonas Hörsch, Tom Brown: *The role of spatial scale in joint
   optimisations of generation and transmission for European highly renewable
   scenarios*, `arXiv:1705.07617 <https://arxiv.org/abs/1705.07617>`_.
