from geopy.distance import vincenty

import os
if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

import logging
logger = logging.getLogger('edisgo')


def calc_geo_branches_in_polygon(mv_grid, polygon, mode, proj):
    # TODO: DOCSTRING

    branches = []
    polygon_shp = transform(proj, polygon)
    for branch in mv_grid.graph_edges():
        nodes = branch['adj_nodes']
        branch_shp = transform(proj, LineString([nodes[0].geo_data, nodes[1].geo_data]))

        # check if branches intersect with polygon if mode = 'intersects'
        if mode == 'intersects':
            if polygon_shp.intersects(branch_shp):
                branches.append(branch)
        # check if polygon contains branches if mode = 'contains'
        elif mode == 'contains':
            if polygon_shp.contains(branch_shp):
                branches.append(branch)
        # error
        else:
            raise ValueError('Mode is invalid!')
    return branches


def calc_geo_branches_in_buffer(node, mv_grid, radius, radius_inc, proj):
    """ Determines branches in nodes' associated graph that are at least partly within buffer of `radius` from `node`.
        If there are no nodes, the buffer is successively extended by `radius_inc` until nodes are found.

    Args:
        node: origin node (e.g. LVStationDing0 object) with associated shapely object (attribute `geo_data`) in any CRS
              (e.g. WGS84)
        radius: buffer radius in m
        radius_inc: radius increment in m
        proj: pyproj projection object: nodes' CRS to equidistant CRS (e.g. WGS84 -> ETRS)

    Returns:
        list of branches (NetworkX branch objects)

    """

    branches = []

    while not branches:
        node_shp = transform(proj, node.geo_data)
        buffer_zone_shp = node_shp.buffer(radius)
        for branch in mv_grid.graph_edges():
            nodes = branch['adj_nodes']
            branch_shp = transform(proj, LineString([nodes[0].geo_data, nodes[1].geo_data]))
            if buffer_zone_shp.intersects(branch_shp):
                branches.append(branch)
        radius += radius_inc

    return branches


def calc_geo_dist_vincenty(node_source, node_target):
    """ Calculates the geodesic distance between `node_source` and `node_target` incorporating the detour factor in
        config_calc.cfg.
    Args:
        node_source: source node (Ding0 object), member of _graph
        node_target: target node (Ding0 object), member of _graph

    Returns:
        Distance in m
    """

    branch_detour_factor = cfg_ding0.get('assumptions', 'branch_detour_factor')

    # notice: vincenty takes (lat,lon)
    branch_length = branch_detour_factor * vincenty((node_source.geo_data.y, node_source.geo_data.x),
                                                    (node_target.geo_data.y, node_target.geo_data.x)).m

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
