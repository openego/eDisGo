from dingo.tools.results import load_nd_from_pickle
from dingo.core.network.stations import LVStationDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
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
    #TODO: Consider that generators and loads in aggr. LAs are already included in MV
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
    # Instantiate a MV grid
    grid = MVGrid(
        network=network,
        grid_district={'geom': dingo_grid.grid_district.geo_data,
                       'population':
                           sum([_.zensus_sum
                                for _ in
                                dingo_grid.grid_district._lv_load_areas])},
        voltage_nom=dingo_grid.v_level)

    # Special treatment of LVLoadAreaCenters see ...
    # TODO: add a reference above for explanation of how these are treated
    la_centers = [_ for _ in dingo_grid._graph.nodes()
                  if isinstance(_, LVLoadAreaCentreDingo)]
    aggregated, aggr_stations = _determine_aggregated_nodes(la_centers)

    # Create list of load instances and add these to grid's graph
    # TODO: add `consumption` to loads
    loads = {_: Load(
        id=_.id_db,
        geom=_.geo_data,
        grid=grid) for _ in dingo_grid.loads()}
    grid.graph.add_nodes_from(loads.values(), type='load')

    # Create list of generator instances and add these to grid's graph
    generators = {_: Generator(
        id=_.id_db,
        geom=_.geo_data,
        nominal_capacity=_.capacity,
        type=_.type,
        subtype=_.subtype,
        grid=grid) for _ in dingo_grid.generators()}
    grid.graph.add_nodes_from(generators.values(), type='generator')

    # Create list of diconnection point instances and add these to grid's graph
    disconnecting_points = {_: MVDisconnectingPoint(id=_.id_db,
                                                 geom=_.geo_data,
                                                 state=_.status,
                                                 grid=grid)
                   for _ in dingo_grid._circuit_breakers}
    grid.graph.add_nodes_from(disconnecting_points.values(),
                              type='disconnection_point')

    # Create list of branch tee instances and add these to grid's graph
    branch_tees = {_: BranchTee(id=_.id_db, geom=_.geo_data, grid=grid)
                   for _ in dingo_grid._cable_distributors}
    grid.graph.add_nodes_from(branch_tees.values(), type='branch_tee')

    # Create list of LV station instances and add these to grid's graph
    stations = {_: Station(id=_.id_db,
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
                if isinstance(_, LVStationDingo) and _ not in aggr_stations}
    grid.graph.add_nodes_from(stations.values(), type='lv_station')

    # Create HV-MV station add to graph
    mv_station = Station(
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
    grid.graph.add_node(mv_station, type='mv_station')

    # Merge node above defined above to a single dict
    nodes = {**loads,
             **generators,
             **disconnecting_points,
             **branch_tees,
             **stations,
             **{dingo_grid.station(): mv_station}}

    # Create list of line instances and add these to grid's graph
    lines = [(nodes[_['adj_nodes'][0]], nodes[_['adj_nodes'][1]],
              {'line': Line(
                  id=_['branch'].id_db,
                  type=_['branch'].type,
                  length=_['branch'].length,
                  grid=grid)
              })
             for _ in dingo_grid.graph_edges()
             if not any([isinstance(_['adj_nodes'][0], LVLoadAreaCentreDingo),
                        isinstance(_['adj_nodes'][1], LVLoadAreaCentreDingo)])]
    grid.graph.add_edges_from(lines, type='line')

    # Assign reference to HV-MV station to MV grid
    grid._station = mv_station

    # Attach aggregated to MV station
    _attach_aggregated(grid, aggregated, dingo_grid)

    return grid


def _determine_aggregated_nodes(la_centers):
    """Determine generation and load within load areas

    Parameters
    ----------
    la_centers: list of LVLoadAreaCentre
        Load Area Centers are Dingo implementations for representating areas of
        high population density with high demand compared to DG potential.

    Notes
    -----
    Currently, MV grid loads are not considered in this aggregation function as
    Dingo data does not come with loads in the MV grid level.

    Returns
    -------
    :obj:`list` of dict
        aggregated
        Dict of the structure

        .. code:

            {'generation': {
                'v_level': {
                    'subtype': {
                        'ids': <ids of aggregated generator>,
                        'capacity'}
                    }
                },
            'load': {
                'consumption':
                    'residential': <value>,
                    'retail': <value>,
                    ...
                }
            'aggregates': {
                'population': int,
                'geom': `shapely.Polygon`
                }
            }
    :obj:`list`
        aggr_stations
        List of LV stations its generation and load is aggregated
    """

    def aggregate_generators(gen, aggr):
        """Aggregate generation capacity per voltage level

        Parameters
        ----------
        gen: dingo.core.GeneratorDingo
            Dingo Generator object
        aggr: dict
            Aggregated generation capacity. For structure see
            `_determine_aggregated_nodes()`.

        Returns
        -------

        """
        if gen.v_level not in aggr['generation']:
            aggr['generation'][gen.v_level] = {}
        if gen.subtype not in aggr['generation'][gen.v_level]:
            aggr['generation'][gen.v_level].update(
                {gen.subtype:
                     {'ids': [gen.id_db],
                      'capacity': gen.capacity,
                      'type': gen.type}})
        aggr['generation'][gen.v_level][gen.subtype]['ids'].append(gen.id_db)
        aggr['generation'][gen.v_level][gen.subtype]['capacity'] += gen.capacity

        return aggr

    aggregated = []
    aggr_stations = []

    for la_center in la_centers:
        aggr = {'generation': {}, 'load': {}, 'aggregates': []}

        # Determine aggregated generation in MV grid
        for gen in la_center.grid.generators():
            aggr = aggregate_generators(gen, aggr)

        # Determine aggregated generation in LV grid
        for lvgd in la_center.lv_load_area._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                aggr = aggregate_generators(gen, aggr)

        # Determine aggregated load in MV grid
        # -> Implement once laods in Dingo MV grids exist

        # Determine aggregated load in LV grid
        # TODO: implement, when sectoral consumption per load object is available
        # TODO: and add load object to graph
        # Note: la_center has cumulative peak load per sector

        # Collect metadata of aggregated load areas
        aggr['aggregates'] = {
            'population': la_center.lv_load_area.zensus_sum,
            'geom': la_center.lv_load_area.geo_area}

        # Determine LV grids/ stations that are aggregated
        stations = [_.lv_grid.station()
                    for _ in la_center.lv_load_area._lv_grid_districts]

        # add elements to lists
        aggregated.append(aggr)
        aggr_stations.append(stations)


    return aggregated, aggr_stations


def _attach_aggregated(grid, aggregated, dingo_grid):
    """Add Generators and Loads to MV station representing aggregated generation
    capacity and load

    Parameters
    ----------
    grid: MVGrid
        MV grid object
    aggregated: dict
        Information about aggregated load and generation capacity. For
        information about the structure of the dict see ... .
    dingo_grid: dingo.Network
        Dingo network container
    Returns
    -------
    MVGrid
        Altered instance of MV grid including aggregated load and generation
    """

    aggr_line_type = dingo_grid.network._static_data['MV_cables'].iloc[
        dingo_grid.network._static_data['MV_cables']['I_max_th'].idxmax()]

    for la in aggregated:
        # add aggregated generators
        for v_level, val in la['generation'].items():
            for subtype, val2 in val.items():
                gen = Generator(
                    id='_'.join(str(_) for _ in val2['ids']),
                    nominal_capacity=val2['capacity'],
                    type=val2['type'],
                    subtype=subtype,
                    geom=grid.station.geom,
                    grid=grid)
                grid.graph.add_node(gen, type='generator')

                # connect generator to MV station
                line = {'line': Line(
                         id='line_aggr_generator_vlevel_{v_level}_'
                            '{subtype}'.format(
                             v_level=v_level,
                             subtype=subtype),
                         type=aggr_line_type,
                         length=1,
                         grid=grid)
                     }
                grid.graph.add_edge(grid.station, gen, line, type='line')