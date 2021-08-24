from pyproj import Transformer
from geopy.distance import geodesic

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

    return Transformer.from_crs("EPSG:{}".format(srid), "EPSG:3035", always_xy=True).transform

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

    return Transformer.from_crs("EPSG:3035", "EPSG:{}".format(srid), always_xy=True).transform


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

    return Transformer.from_crs("EPSG:{}".format(srid1), "EPSG:{}".format(srid2), always_xy=True).transform


def calc_geo_lines_in_buffer(grid_topology, bus, grid,
                             buffer_radius=2000, buffer_radius_inc=1000):
    """
    Determines lines that are at least partly within buffer around given bus.

    If there are no lines, the buffer specified in `buffer_radius` is
    successively extended by `buffer_radius_inc` until lines are found.

    Parameters
    ----------
    grid_topology : :class:`~.network.topology.Topology`
    bus : :pandas:`pandas.Series<Series>`
        Data of origin bus the buffer is created around.
        Series has same rows as columns of
        :attr:`~.network.topology.Topology.buses_df`.
    grid : :class:`~.network.grids.Grid`
        Grid whose lines are searched.
    buffer_radius : float, optional
        Radius in m used to find connection targets. Default: 2000.
    buffer_radius_inc : float, optional
        Radius in m which is incrementally added to `buffer_radius` as long as
        no target is found. Default: 1000.

    Returns
    -------
    list(str)
        List of lines in buffer (meaning close to the bus) sorted by the
        lines' representatives.

    """

    lines = []
    srid = grid_topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))
    projection = proj2equidistant(srid)
    while not lines:
        buffer_zone_shp = bus_shp.buffer(buffer_radius)
        for line in grid.lines_df.index:
            line_bus0 = grid_topology.lines_df.loc[line, "bus0"]
            bus0 = grid_topology.buses_df.loc[line_bus0, :]
            line_bus1 = grid_topology.lines_df.loc[line, "bus1"]
            bus1 = grid_topology.buses_df.loc[line_bus1, :]
            line_shp = transform(
                projection,
                LineString([Point(bus0.x, bus0.y), Point(bus1.x, bus1.y)]),
            )
            if buffer_zone_shp.intersects(line_shp):
                lines.append(line)
        buffer_radius += buffer_radius_inc

    return sorted(lines)


def calc_geo_dist_vincenty(grid_topology, bus_source, bus_target,
                           branch_detour_factor=1.3):
    """
    Calculates the geodesic distance between two buses in km.

    The detour factor in config_grid is incorporated in the geodesic distance.

    Parameters
    ----------
    grid_topology : :class:`~.network.topology.Topology`
    bus_source : str
        Name of source bus as in index of
        :attr:`~.network.topology.Topology.buses_df`.
    bus_target : str
        Name of target bus as in index of
        :attr:`~.network.topology.Topology.buses_df`.
    branch_detour_factor : float
        Detour factor to consider that two buses can usually not be
        connected directly. Default: 1.3.

    Returns
    -------
    float
        Distance in km.

    """

    bus_source = grid_topology.buses_df.loc[bus_source, :]
    bus_target = grid_topology.buses_df.loc[bus_target, :]

    # notice: vincenty takes (lat,lon)
    branch_length = (
        branch_detour_factor
        * geodesic(
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
    Finds the nearest bus in `bus_target` to a given point.

    Parameters
    ----------
    point : :shapely:`shapely.Point<Point>`
        Point to find nearest bus for.
    bus_target : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with candidate buses and their positions given in 'x' and 'y'
        columns. The dataframe has the same format as
        :attr:`~.network.topology.Topology.buses_df`.

    Returns
    -------
    tuple(str, float)
        Tuple that contains the name of the nearest bus and its distance.

    """

    bus_target["dist"] = [
        geodesic((point.y, point.x), (y, x)).km
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
    repr = []

    srid = grid_topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))
    projection = proj2equidistant(srid)
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
            projection, Point(line_bus0.x, line_bus0.y)
        )
        line_bus1_shp = transform(
            projection, Point(line_bus1.x, line_bus1.y)
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
        # discard duplicates
        if not conn_objects_min["repr"] in repr:
            conn_objects_min_stack.append(conn_objects_min)
            repr.append(conn_objects_min["repr"])

    # sort all objects by distance from node
    conn_objects_min_stack = [
        _ for _ in sorted(conn_objects_min_stack, key=lambda x: x["dist"])
    ]

    return conn_objects_min_stack
