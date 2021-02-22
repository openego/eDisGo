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


def find_nearest_conn_objects(grid_topology, bus, lines,
                              conn_diff_tolerance=0.0001):
    """
    Searches all lines for the nearest possible connection object per line.

    It picks out 1 object out of 3 possible objects: 2 line-adjacent buses
    and 1 potentially created branch tee on the line (using perpendicular
    projection). The resulting stack (list) is sorted ascending by distance
    from bus.

    Parameters
    ----------
    grid_topology : :class:`~.network.topology.Topology`
    bus : :pandas:`pandas.Series<Series>`
        Data of bus to connect.
        Series has same rows as columns of
        :attr:`~.network.topology.Topology.buses_df`.
    lines : list(str)
        List of line representatives from index of
        :attr:`~.network.topology.Topology.lines_df`.
    conn_diff_tolerance : float, optional
        Threshold which is used to determine if 2 objects are at the same
        position. Default: 0.0001.

    Returns
    -------
    list(dict)
        List of connection objects. Each object is represented by dict with
        representative, shapely object and distance to node.

    """

    conn_objects_min_stack = []

    srid = grid_topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    for line in lines:

        line_bus0 = grid_topology.buses_df.loc[
            grid_topology.lines_df.loc[line, "bus0"]
        ]
        line_bus1 = grid_topology.buses_df.loc[
            grid_topology.lines_df.loc[line, "bus1"]
        ]

        # create shapely objects for 2 buses and line between them,
        # transform to equidistant CRS
        line_bus0_shp = transform(
            proj2equidistant(srid), Point(line_bus0.x, line_bus0.y)
        )
        line_bus1_shp = transform(
            proj2equidistant(srid), Point(line_bus1.x, line_bus1.y)
        )
        line_shp = LineString([line_bus0_shp, line_bus1_shp])

        # create dict with line & 2 adjacent buses and their shapely objects
        # and distances
        conn_objects = {
            "s1": {
                "repr": line_bus0.name,
                "shp": line_bus0_shp,
                "dist": bus_shp.distance(line_bus0_shp) * 0.999,
            },
            "s2": {
                "repr": line_bus1.name,
                "shp": line_bus1_shp,
                "dist": bus_shp.distance(line_bus1_shp) * 0.999,
            },
            "b": {
                "repr": line,
                "shp": line_shp,
                "dist": bus_shp.distance(line_shp),
            },
        }

        # remove line from the dict of possible conn. objects if it is too
        # close to the bus (necessary to assure that connection target is
        # reproducible)
        if (
                abs(conn_objects["s1"]["dist"] - conn_objects["b"]["dist"])
                < conn_diff_tolerance
                or abs(conn_objects["s2"]["dist"] - conn_objects["b"]["dist"])
                < conn_diff_tolerance
        ):
            del conn_objects["b"]

        # remove MV station as possible connection point
        if (
                conn_objects["s1"]["repr"]
                == grid_topology.mv_grid.station.index[0]
        ):
            del conn_objects["s1"]
        elif (
                conn_objects["s2"]["repr"]
                == grid_topology.mv_grid.station.index[0]
        ):
            del conn_objects["s2"]

        # find nearest connection point in conn_objects
        conn_objects_min = min(
            conn_objects.values(), key=lambda v: v["dist"]
        )

        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [
        _ for _ in sorted(conn_objects_min_stack, key=lambda x: x["dist"])
    ]

    return conn_objects_min_stack
