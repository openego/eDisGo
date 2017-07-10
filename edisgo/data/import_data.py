from dingo.tools.results import load_nd_from_pickle
from dingo.core.network.stations import LVStationDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
from ..grid.components import Load, Generator, MVDisconnectingPoint, BranchTee,\
    LVStation

def import_from_dingo(file):
    """
    The actual code body of the Dingo data importer

    Notes
    -----
    Assumes `dingo.NetworkDingo` provided by `file` contains only data of one
    mv_grid_district.

    Parameters
    ----------
    file: str or dingo.NetworkDingo


    Returns
    -------

    """
    # when `file` is a string, it will be read by the help of pickle

    if isinstance(file, str):
        dingo_nd = load_nd_from_pickle(filename=file)
    # otherwise it is assumed the object is passed directly
    else:
        dingo_nd = file

    mv_grid = dingo_nd._mv_grid_districts[0].mv_grid

    # Import low-voltage grid data
    # TODO: implement this!

    # Import medium-voltage grid data
    _build_mv_grid(mv_grid)



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


def _build_mv_grid(mv_grid):
    """

    Parameters
    ----------
    mv_grid: dingo.LVGridDingo

    Returns
    -------
    MVGrid
    """

    # TODO: add `consumption` to loads
    loads = [Load(
        id=_.id_db,
        geom=_.geo_data) for _ in mv_grid.loads()]

    generators = [Generator(
        id=_.id_db,
        geom=_.geo_data,
        nominal_capacity=_.capacity,
        type=_.type,
        subtype=_.subtype) for _ in mv_grid.generators()]

    disconnecting_points = [MVDisconnectingPoint(id=_.id_db,
                                                 geom=_.geo_data,
                                                 state=_.status)
                   for _ in mv_grid._circuit_breakers]

    branch_tees = [BranchTee(id=_.id_db, geom=_.geo_data)
                   for _ in mv_grid._cable_distributors]

    stations = [LVStation(id=_.id_db,
                          geom=_.geo_data,
                          transformers=_.transformers)
                for _ in mv_grid._graph.nodes()
                if isinstance(_, LVStationDingo)]

    # TODO: add LoadAreaCenter

    # TODO: add edges

    # TODO: attach above data to the LVGrid object
