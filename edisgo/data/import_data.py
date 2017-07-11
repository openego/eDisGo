from dingo.tools.results import load_nd_from_pickle
from dingo.core.network.stations import LVStationDingo
from ..grid.components import Load, Generator, MVDisconnectingPoint, BranchTee,\
    Station, Line, Transformer
from ..grid.grids import MVGrid, LVGrid, Graph
import pandas as pd


def import_from_dingo(file, network):
    """
    The actual code body of the Dingo data importer

    Notes
    -----
    Assumes `dingo.NetworkDingo` provided by `file` contains only data of one
    mv_grid_district.

    Parameters
    ----------
    file: str or dingo.NetworkDingo
        If a str is provided it is assumed it points to a pickle with Dingo
        grid data. This file will be read.
        If a object of the type `dingo.NetworkDingo` data will be used directly
        from this object.
    network: Network
        The eDisGo container object

    Returns
    -------

    """
    # when `file` is a string, it will be read by the help of pickle

    if isinstance(file, str):
        dingo_nd = load_nd_from_pickle(filename=file)
    # otherwise it is assumed the object is passed directly
    else:
        dingo_nd = file

    dingo_mv_grid = dingo_nd._mv_grid_districts[0].mv_grid

    # Import low-voltage grid data
    # TODO: implement this!
    # TODO: loop over LV grids and create list of LV grid object
    # TODO: consider for potentially aggregated LAs while looping

    # Import medium-voltage grid data
    _build_mv_grid(dingo_mv_grid, network)



def _build_lv_grid(lv_grid):
    """
    Build eDisGo LV grid from Dingo data

    Parameters
    ----------
    lv_grid: dingo.LVGridDingo

    Returns
    -------
    LVGrid
    """
    pass


def _build_mv_grid(dingo_grid, network):
    """

    Parameters
    ----------
    dingo_grid: dingo.MVGridDingo
        Dingo MV grid object
    network: Network
        The eDisGo container object

    Returns
    -------
    MVGrid
        A MV grid of class edisgo.grids.MVGrid is return. Data from the Dingo
        MV Grid object is translated to the new grid object.
    """

    # TODO: Why is the attribute population == 0?
    grid = MVGrid(
        network=network,
        grid_district={'geom': dingo_grid.grid_district.geo_data,
                       'population':
                           sum([_.zensus_sum
                                for _ in
                                dingo_grid.grid_district._lv_load_areas])},
        voltage_nom=dingo_grid.v_level)


    # TODO: add `consumption` to loads
    loads = [Load(
        id=_.id_db,
        geom=_.geo_data,
        grid=grid) for _ in dingo_grid.loads()]

    generators = [Generator(
        id=_.id_db,
        geom=_.geo_data,
        nominal_capacity=_.capacity,
        type=_.type,
        subtype=_.subtype,
        grid=grid) for _ in dingo_grid.generators()]

    disconnecting_points = [MVDisconnectingPoint(id=_.id_db,
                                                 geom=_.geo_data,
                                                 state=_.status,
                                                 grid=grid)
                   for _ in dingo_grid._circuit_breakers]

    branch_tees = [BranchTee(id=_.id_db, geom=_.geo_data, grid=grid)
                   for _ in dingo_grid._cable_distributors]

    stations = [Station(id=_.id_db,
                        geom=_.geo_data,
                        grid=grid,
                        transformers=[Transformer(
                            grid=grid,
                            id=t.grid.id_db,
                            geom=_.geo_data,
                            voltage_op=t.v_level,
                            type=pd.Series(dict(
                                s=t.s_max_a, x=t.x, r=t.r))
                        ) for t in _.transformers()])
                for _ in dingo_grid._graph.nodes()
                if isinstance(_, LVStationDingo)]

    # TODO: test if lines[0].geom() works once lines[0]._grid is set
    lines = [(_['adj_nodes'][0], _['adj_nodes'][1],
              {'line': Line(
                  id=_['branch'].id_db,
                  type=_['branch'].type,
                  length=_['branch'].length,
                  grid=grid)
              })
             for _ in dingo_grid.graph_edges()]

    # TODO: attach above data to the MVGrid object's _graph

    grid._station = Station(
        id=dingo_grid.station().id_db,
        geom=dingo_grid.station().geo_data,
        transformers=[Transformer(
            grid=grid,
            id=_.grid.id_db,
            geom=dingo_grid.station().geo_data,
            voltage_op=_.v_level,
            type=pd.Series(dict(
                s=_.s_max_a, x=_.x, r=_.r)))
                      for _ in dingo_grid.station().transformers()])
    return grid
