import pyproj
from functools import partial
from geopy.distance import vincenty

import os
if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

import logging
logger = logging.getLogger('edisgo')


def proj2equidistant():
    """Defines WGS84 (conformal) to ETRS (equidistant) projection

    Returns
    -------
    :functools:`partial`
    """

    return partial(pyproj.transform,
                   pyproj.Proj(init='epsg:4326'),  # source coordinate system
                   pyproj.Proj(init='epsg:3035'))  # destination coordinate system


def proj2conformal():
    """Defines ETRS (equidistant) to WGS84 (conformal) projection

    Returns
    -------
    :functools:`partial`
    """
    return partial(pyproj.transform,
                   pyproj.Proj(init='epsg:3035'),  # source coordinate system
                   pyproj.Proj(init='epsg:4326'))  # destination coordinate system


def calc_geo_lines_in_buffer(node, mv_grid, radius, radius_inc):
    """Determines branches in nodes' associated graph that are at least partly within buffer of `radius` from `node`.
    If there are no nodes, the buffer is successively extended by `radius_inc` until nodes are found.

    Args:
        node: origin node (e.g. LVStationDing0 object) with associated shapely object (attribute `geo_data`) in any CRS
              (e.g. WGS84)
        radius: buffer radius in m
        radius_inc: radius increment in m

    Returns:
        Sorted list of branches (NetworkX branch objects)

    """
    # TODO: Update docstring

    lines = []

    while not lines:
        node_shp = transform(proj2equidistant(), node.geom)
        buffer_zone_shp = node_shp.buffer(radius)
        for line in mv_grid.graph.lines():
            nodes = line['adj_nodes']
            branch_shp = transform(proj2equidistant(), LineString([nodes[0].geom, nodes[1].geom]))
            if buffer_zone_shp.intersects(branch_shp):
                lines.append(line)
        radius += radius_inc

    return sorted(lines, key=lambda _: repr(_))


def calc_geo_dist_vincenty(network, node_source, node_target):
    """ Calculates the geodesic distance between `node_source` and `node_target` incorporating the detour factor in
        config_calc.cfg.
    Args:
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    node_source:
        source node (Ding0 object), member of _graph
    node_target:
        target node (Ding0 object), member of _graph

    Returns:
        Distance in m
    """
    # TODO: Update docstring

    branch_detour_factor = network.config['connect']['branch_detour_factor']

    # notice: vincenty takes (lat,lon)
    branch_length = branch_detour_factor * vincenty((node_source.geom.y, node_source.geom.x),
                                                    (node_target.geom.y, node_target.geom.x)).m

    # ========= BUG: LINE LENGTH=0 WHEN CONNECTING GENERATORS ===========
    # When importing generators, the geom_new field is used as position. If it is empty, EnergyMap's geom
    # is used and so there are a couple of generators at the same position => length of interconnecting
    # line is 0. See issue #76
    if branch_length == 0:
        branch_length = 1
        logger.warning('Geo distance is zero, check objects\' positions. '
                       'Distance is set to 1m')
    # ===================================================================

    return branch_length
