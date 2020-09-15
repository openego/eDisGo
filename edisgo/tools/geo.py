import pyproj
from functools import partial
from geopy.distance import vincenty

import os

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import LineString, Point
    from shapely.ops import transform

import logging

logger = logging.getLogger("edisgo")


def proj2equidistant(srid):
    """
    Transforms to equidistant projection (epsg:3035).

    Parameters
    ----------
    srid : int
        Spatial reference identifier of geometry to transform.

    Returns
    -------
    :py:func:`functools.partial`

    """

    return partial(
        pyproj.transform,
        pyproj.Proj(init="epsg:{}".format(srid)),  # source CRS
        pyproj.Proj(init="epsg:3035"),  # destination CRS
    )


def proj2equidistant_reverse(srid):
    """
    Transforms back from equidistant projection to given projection.

    Parameters
    ----------
    srid : int
        Spatial reference identifier of geometry to transform.

    Returns
    -------
    :py:func:`functools.partial`

    """

    return partial(
        pyproj.transform,
        pyproj.Proj(init="epsg:3035"),  # source CRS
        pyproj.Proj(init="epsg:{}".format(srid)),  # destination CRS
    )


def proj_by_srids(srid1, srid2):
    """
    Transforms from specified projection to other specified projection.

    Parameters
    ----------
    srid1 : int
        Spatial reference identifier of geometry to transform.
    srid2 : int
        Spatial reference identifier of destination CRS.

    Returns
    -------
    :py:func:`functools.partial`

    Notes
    -----
    Projections often used are conformal projection (epsg:4326), equidistant
    projection (epsg:3035) and spherical mercator projection (epsg:3857).

    """

    return partial(
        pyproj.transform,
        pyproj.Proj(init="epsg:{}".format(srid1)),  # source CRS
        pyproj.Proj(init="epsg:{}".format(srid2)),  # destination CRS
    )


def calc_geo_lines_in_buffer(edisgo_object, bus, grid):
    """Determines lines in nodes' associated graph that are at least partly
    within buffer of radius from node. If there are no lines, the buffer is
    successively extended by radius_inc until lines are found.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus : pandas Series
        Data of origin bus the buffer is created around.
        Series has same rows as columns of topology.buses_df.
    grid : :class:`~.network.grids.Grid`
        Grid whose lines are searched

    Returns
    -------
    :obj:`list` of :class:`~.network.components.Line`
        Sorted (by repr()) list of lines

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/tools/geo.py#L53>`_.

    """

    buffer_radius = int(
        edisgo_object.config["grid_connection"]["conn_buffer_radius"]
    )
    buffer_radius_inc = int(
        edisgo_object.config["grid_connection"]["conn_buffer_radius_inc"]
    )

    lines = []
    srid = edisgo_object.topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    while not lines:
        buffer_zone_shp = bus_shp.buffer(buffer_radius)
        for line in grid.lines_df.index:
            line_bus0 = edisgo_object.topology.lines_df.loc[line, "bus0"]
            bus0 = edisgo_object.topology.buses_df.loc[line_bus0, :]
            line_bus1 = edisgo_object.topology.lines_df.loc[line, "bus1"]
            bus1 = edisgo_object.topology.buses_df.loc[line_bus1, :]
            line_shp = transform(
                proj2equidistant(srid),
                LineString([Point(bus0.x, bus0.y), Point(bus1.x, bus1.y)]),
            )
            if buffer_zone_shp.intersects(line_shp):
                lines.append(line)
        buffer_radius += buffer_radius_inc

    return sorted(lines)


def calc_geo_dist_vincenty(edisgo_object, bus_source, bus_target):
    """
    Calculates the geodesic distance between node_source and node_target in km.

    The detour factor in config is incorporated in the geodesic distance.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus_source : str
        Name of bus to connect as in topology.buses_df.
    bus_target : pandas Series
        Name of target bus as in topology.buses_df.

    Returns
    -------
    :obj:`float`
        Distance in km.

    """

    branch_detour_factor = edisgo_object.config["grid_connection"][
        "branch_detour_factor"
    ]

    bus_source = edisgo_object.topology.buses_df.loc[bus_source, :]
    bus_target = edisgo_object.topology.buses_df.loc[bus_target, :]

    # notice: vincenty takes (lat,lon)
    branch_length = (
        branch_detour_factor
        * vincenty(
            (bus_source.y, bus_source.x), (bus_target.y, bus_target.x)
        ).m
    )

    # ========= BUG: LINE LENGTH=0 WHEN CONNECTING GENERATORS ===========
    # When importing generators, the geom_new field is used as position. If it
    # is empty, EnergyMap's geom is used and so there are a couple of
    # generators at the same position => length of interconnecting
    # line is 0. See issue #76
    if branch_length == 0:
        branch_length = 1
        logger.debug(
            "Geo distance is zero, check objects' positions. "
            "Distance is set to 1m."
        )
    # ===================================================================

    return branch_length / 1e3


def find_nearest_bus(point, bus_target):
    """
    Finds the nearest bus in `bus_target` from a given point.

    Parameters
    ----------
    bus_source : shapely.geometry.Point
        Point to calculate distance from
    bus_target : pandas DataFrame
        List of candidate nodes with positions given as 'x' and 'y' columns

    Returns
    -------
    :tuple: (`str`, `float`)
        Tuple that contains the name of the nearest node in the list and its distance
    """

    bus_target["dist"] = [
        vincenty((point.y, point.x), (y, x)).km
        for (x, y) in zip(bus_target["x"], bus_target["y"])
    ]
    return bus_target["dist"].idxmin(), bus_target["dist"].min()
