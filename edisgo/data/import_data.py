from dingo.tools.results import load_nd_from_pickle
from dingo.core.network.stations import LVStationDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
from ..grid.components import Load, Generator, MVDisconnectingPoint, BranchTee,\
    Station, Line, Transformer
from ..grid.grids import MVGrid, LVGrid
import pandas as pd
import numpy as np
import networkx as nx


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

    # Import medium-voltage grid data
    network.mv_grid =_build_mv_grid(dingo_mv_grid, network)

    # Import low-voltage grid data
    lv_grids, lv_station_mapping, lv_grid_mapping  = _build_lv_grid(dingo_mv_grid)

    # Assign lv_grids to network
    network.mv_grid.lv_grids = lv_grids

    # Check data integrity
    _validate_dingo_grid_import(network.mv_grid, dingo_mv_grid, lv_grid_mapping)


def _build_lv_grid(dingo_grid):
    """
    Build eDisGo LV grid from Dingo data

    Parameters
    ----------
    dingo_grid: dingo.MVGridDingo
        Dingo MV grid object

    Returns
    -------
    list of LVGrid
        LV grids
    dict
        Dictionary containing a mapping of LV stations in Dingo to newly
        created eDisGo LV stations. This mapping is used to use the same
        instances of LV stations in the MV grid graph.
    """

    lv_station_mapping = {}
    lv_grids = []
    lv_grid_mapping = {}

    for la in dingo_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            dingo_lv_grid = lvgd.lv_grid
            if not dingo_lv_grid.grid_district.lv_load_area.is_aggregated:

                # Create LV grid instance
                lv_grid = LVGrid(
                    id=dingo_lv_grid.id_db,
                    geom=dingo_lv_grid.grid_district.geo_data,
                    grid_district={
                        'geom': dingo_lv_grid.grid_district.geo_data,
                        'population': dingo_lv_grid.grid_district.population},
                    voltage_nom=dingo_lv_grid.v_level)

                # Create LV station instances
                station = Station(id=dingo_lv_grid._station.id_db,
                                  geom=dingo_lv_grid._station.geo_data,
                                  grid=lv_grid,
                                  transformers=[Transformer(
                                      grid=lv_grid,
                                      id=t.grid.id_db,
                                      geom=dingo_lv_grid.grid_district.geo_data,
                                      voltage_op=t.v_level,
                                      type=pd.Series(dict(
                                          s=t.s_max_a, x=t.x, r=t.r))
                                  ) for t in dingo_lv_grid._station.transformers()])
                lv_grid.graph.add_node(station, type='lv_station')
                lv_station_mapping.update({dingo_lv_grid._station: station})

                # Create list of load instances and add these to grid's graph
                loads = {_: Load(
                    id=_.id_db,
                    geom=_.geo_data,
                    grid=lv_grid,
                    consumption=_.consumption) for _ in dingo_lv_grid.loads()}
                lv_grid.graph.add_nodes_from(loads.values(), type='load')

                # Create list of generator instances and add these to grid's graph
                generators = {_: Generator(
                    id=_.id_db,
                    geom=_.geo_data,
                    nominal_capacity=_.capacity,
                    type=_.type,
                    subtype=_.subtype,
                    grid=lv_grid) for _ in dingo_lv_grid.generators()}
                lv_grid.graph.add_nodes_from(generators.values(), type='generator')

                # Create list of branch tee instances and add these to grid's graph
                branch_tees = {
                    _: BranchTee(id=_.id_db, geom=_.geo_data, grid=lv_grid)
                    for _ in dingo_lv_grid._cable_distributors}
                lv_grid.graph.add_nodes_from(branch_tees.values(),
                                              type='branch_tee')

                # Merge node above defined above to a single dict
                nodes = {**loads,
                         **generators,
                         **branch_tees,
                         **{dingo_lv_grid._station: station}}

                edges = []
                edges_raw = list(nx.get_edge_attributes(
                    dingo_lv_grid._graph, 'branch').items())
                for edge in edges_raw:
                    edges.append({'adj_nodes': edge[0], 'branch': edge[1]})

                # Create list of line instances and add these to grid's graph
                lines = [(nodes[_['adj_nodes'][0]], nodes[_['adj_nodes'][1]],
                          {'line': Line(
                              id=_['branch'].id_db,
                              type=_['branch'].type,
                              length=_['branch'].length,
                              grid=lv_grid)
                          })
                         for _ in edges]
                lv_grid.graph.add_edges_from(lines, type='line')

                # Add LV station as association to LV grid
                lv_grid._station = station

                # Add to lv grid mapping
                lv_grid_mapping.update({lv_grid: dingo_lv_grid})

                # Put all LV grid to a list of LV grids
                lv_grids.append(lv_grid)

    # TODO: don't forget to adapt lv stations creation in MV grid
    return lv_grids, lv_station_mapping, lv_grid_mapping


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

    # Instantiate a MV grid
    grid = MVGrid(
        network=network,
        grid_district={'geom': dingo_grid.grid_district.geo_data,
                       'population':
                           sum([_.zensus_sum
                                for _ in
                                dingo_grid.grid_district._lv_load_areas
                                if not np.isnan(_.zensus_sum)])},
        voltage_nom=dingo_grid.v_level)

    # Special treatment of LVLoadAreaCenters see ...
    # TODO: add a reference above for explanation of how these are treated
    la_centers = [_ for _ in dingo_grid._graph.nodes()
                  if isinstance(_, LVLoadAreaCentreDingo)]
    aggregated, aggr_stations = _determine_aggregated_nodes(la_centers)

    # Create list of load instances and add these to grid's graph
    loads = {_: Load(
        id=_.id_db,
        geom=_.geo_data,
        grid=grid,
        consumption=_.consumption) for _ in dingo_grid.loads()}
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
        else:
            aggr['generation'][gen.v_level][gen.subtype]['ids'].append(gen.id_db)
            aggr['generation'][gen.v_level][gen.subtype]['capacity'] += gen.capacity

        return aggr

    def aggregate_loads(la_center, aggr):
        """Aggregate consumption in load area per sector

        Parameters
        ----------
        la_center: LVLoadAreaCentreDingo
            Load area center object from Dingo

        Returns
        -------

        """
        for s in ['retail', 'industrial', 'agricultural', 'residential']:
            if s not in aggr['load']:
                aggr['load'][s] = 0

        aggr['load']['retail'] += sum(
            [_.sector_consumption_retail
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['industrial'] += sum(
            [_.sector_consumption_industrial
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['agricultural'] += sum(
            [_.sector_consumption_agricultural
             for _ in la_center.lv_load_area._lv_grid_districts])
        aggr['load']['residential'] += sum(
            [_.sector_consumption_residential
             for _ in la_center.lv_load_area._lv_grid_districts])

        return aggr

    aggregated = []
    aggr_stations = []

    generation_aggr = {}
    for la in la_centers[0].grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                if la.is_aggregated:
                    generation_aggr.setdefault(gen.type, {})
                    generation_aggr[gen.type].setdefault(gen.subtype, {'dingo': 0})
                    generation_aggr[gen.type][gen.subtype].setdefault('dingo', 0)
                    generation_aggr[gen.type][gen.subtype]['dingo'] += gen.capacity

    for la_center in la_centers:
        aggr = {'generation': {}, 'load': {}, 'aggregates': []}

        # Determine aggregated generation in LV grid
        for lvgd in la_center.lv_load_area._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                aggr = aggregate_generators(gen, aggr)

        # Determine aggregated load in MV grid
        # -> Implement once laods in Dingo MV grids exist

        # Determine aggregated load in LV grid
        aggr = aggregate_loads(la_center, aggr)

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
                         length=.5,
                         grid=grid)
                     }
                grid.graph.add_edge(grid.station, gen, line, type='line')

        load = Load(
            geom=grid.station.geom,
            consumption=la['load'])

        grid.graph.add_node(load, type='load')
        grid.graph.add_edge(grid.station, load, line, type='line')


def _validate_dingo_grid_import(mv_grid, dingo_mv_grid, lv_grid_mapping):
    """Cross-check imported data with original data source

    Parameters
    ----------
    mv_grid: MVGrid
        eDisGo MV grid instance
    dingo_mv_grid: MVGridDingo
        Dingo MV grid instance
    lv_grid_mapping: dict
        Translates Dingo LV grids to associated, newly created eDisGo LV grids
    """

    # Check number of components in MV grid
    _validate_dingo_mv_grid_import(mv_grid, dingo_mv_grid)

    # Check number of components in LV grid
    _validate_dingo_lv_grid_import(mv_grid.lv_grids, dingo_mv_grid,
                                   lv_grid_mapping)

    # Check cumulative load and generation in MV grid district
    _validate_load_generation(mv_grid, dingo_mv_grid)



def _validate_dingo_mv_grid_import(grid, dingo_grid):
    """Verify imported data with original data from Dingo

    Parameters
    ----------
    grid: MVGrid
        MV Grid data (eDisGo)
    dingo_grid: dingo.MVGridDingo
        Dingo MV grid object

    Notes
    -----
    The data validation excludes grid components located in aggregated load
    areas as these are represented differently in eDisGo.

    Returns
    -------
    dict
        Dict showing data integrity for each type of grid component
    """

    integrity_checks = ['branch_tee',
                        'disconnection_point', 'mv_transformer',
                        'lv_station'#,'line',
                        ]

    data_integrity = {}
    data_integrity.update({_: {'dingo': None, 'edisgo': None, 'msg': None}
                           for _ in integrity_checks})

    # Check number of branch tees
    data_integrity['branch_tee']['dingo'] = len(dingo_grid._cable_distributors)
    data_integrity['branch_tee']['edisgo'] = len(
        grid.graph.nodes_by_attribute('branch_tee'))

    # Check number of disconnecting points
    data_integrity['disconnection_point']['dingo'] = len(
        dingo_grid._circuit_breakers)
    data_integrity['disconnection_point']['edisgo'] = len(
        grid.graph.nodes_by_attribute('disconnection_point'))

    # Check number of MV transformers
    data_integrity['mv_transformer']['dingo'] = len(
        list(dingo_grid.station().transformers()))
    data_integrity['mv_transformer']['edisgo'] = len(
        grid.station.transformers)

    # Check number of LV stations in MV grid (graph)
    data_integrity['lv_station']['edisgo'] = len(grid.graph.nodes_by_attribute(
        'lv_station'))
    data_integrity['lv_station']['dingo'] = len(
        [_ for _ in dingo_grid._graph.nodes()
         if isinstance(_, LVStationDingo)])

    # Check number of lines outside aggregated LA
    # edges_w_la = grid.graph.graph_edges()
    # data_integrity['line']['edisgo'] = len([_ for _ in edges_w_la
    #          if not (_['adj_nodes'][0] == grid.station or
    #                  _['adj_nodes'][1] == grid.station) and
    #          _['line']._length > .5])
    # data_integrity['line']['dingo'] = len(
    #     [_ for _ in dingo_grid.graph_edges()
    #      if not _['branch'].connects_aggregated])

    # raise an error if data does not match
    for c in integrity_checks:
        if data_integrity[c]['edisgo'] != data_integrity[c]['dingo']:
            raise ValueError(
                'Unequal number of objects for {c}. '
                '\n\tDingo:\t{dingo_no}'
                '\n\teDisGo:\t{edisgo_no}'.format(
                    c=c,
                    dingo_no=data_integrity[c]['dingo'],
                    edisgo_no=data_integrity[c]['edisgo']))

    return data_integrity


def _validate_dingo_lv_grid_import(grids, dingo_grid, lv_grid_mapping):
    """Verify imported data with original data from Dingo

    Parameters
    ----------
    grids: list of LVGrid
        LV Grid data (eDisGo)
    dingo_grid: dingo.MVGridDingo
        Dingo MV grid object
    lv_grid_mapping: dict
        Defines relationship between Dingo and eDisGo grid objects

    Notes
    -----
    The data validation excludes grid components located in aggregated load
    areas as these are represented differently in eDisGo.

    Returns
    -------
    dict
        Dict showing data integrity for each type of grid component
    """

    integrity_checks = ['branch_tee', 'lv_transformer',
                        'generator', 'load','line']

    data_integrity = {}

    for grid in grids:

        data_integrity.update({grid:{_: {'dingo': None, 'edisgo': None, 'msg': None}
                           for _ in integrity_checks}})

        # Check number of branch tees
        data_integrity[grid]['branch_tee']['dingo'] = len(
            lv_grid_mapping[grid]._cable_distributors)
        data_integrity[grid]['branch_tee']['edisgo'] = len(
            grid.graph.nodes_by_attribute('branch_tee'))

        # Check number of LV transformers
        data_integrity[grid]['lv_transformer']['dingo'] = len(
            list(lv_grid_mapping[grid].station().transformers()))
        data_integrity[grid]['lv_transformer']['edisgo'] = len(
            grid.station.transformers)

        # Check number of generators
        data_integrity[grid]['generator']['edisgo'] = len(
            grid.graph.nodes_by_attribute('generator'))
        data_integrity[grid]['generator']['dingo'] = len(
            list(lv_grid_mapping[grid].generators()))

        # Check number of loads
        data_integrity[grid]['load']['edisgo'] = len(
            grid.graph.nodes_by_attribute('load'))
        data_integrity[grid]['load']['dingo'] = len(
            list(lv_grid_mapping[grid].loads()))

        # Check number of lines outside aggregated LA
        data_integrity[grid]['line']['edisgo'] = len(
            list(grid.graph.graph_edges()))
        data_integrity[grid]['line']['dingo'] = len(
            [_ for _ in lv_grid_mapping[grid].graph_edges()
             if not _['branch'].connects_aggregated])

    # raise an error if data does not match
    for grid in grids:
        for c in integrity_checks:
            if data_integrity[grid][c]['edisgo'] != data_integrity[grid][c]['dingo']:
                raise ValueError(
                    'Unequal number of objects in grid {grid} for {c}. '
                    '\n\tDingo:\t{dingo_no}'
                    '\n\teDisGo:\t{edisgo_no}'.format(
                        grid=grid,
                        c=c,
                        dingo_no=data_integrity[grid][c]['dingo'],
                        edisgo_no=data_integrity[grid][c]['edisgo']))


def _validate_load_generation(mv_grid, dingo_mv_grid):
    """

    Parameters
    ----------
    mv_grid
    dingo_mv_grid

    Notes
    -----
    Only loads in LV grids are compared as currently Dingo does not have MV
    connected loads
    """

    decimal_places = 6
    tol = 10 ** -decimal_places

    sectors = ['retail', 'industrial', 'agricultural', 'residential']
    consumption = {_: {'edisgo': 0, 'dingo':0} for _ in sectors}


    # Collect eDisGo LV loads
    for lv_grid in mv_grid.lv_grids:
        for load in lv_grid.graph.nodes_by_attribute('load'):
            for s in sectors:
                consumption[s]['edisgo'] += load.consumption.get(s, 0)

    # Collect Dingo LV loads
    for la in dingo_mv_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for load in lvgd.lv_grid.loads():
                for s in sectors:
                    consumption[s]['dingo'] += load.consumption.get(s, 0)

    # Compare cumulative load
    for k, v in consumption.items():
            if v['edisgo'] != v['dingo']:
                raise ValueError(
                    'Consumption for {sector} does not match! '
                    '\n\tDingo:\t{dingo}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        sector=k,
                        dingo=v['dingo'],
                        edisgo=v['edisgo']))

    # Compare cumulative generation capacity
    mv_gens = mv_grid.graph.nodes_by_attribute('generator')
    lv_gens = []
    [lv_gens.extend(_.graph.nodes_by_attribute('generator'))
                    for _ in mv_grid.lv_grids]

    generation = {}
    generation_aggr = {}

    # collect eDisGo cumulative generation capacity
    for gen in mv_gens + lv_gens:
        if gen in mv_grid.graph.neighbors(mv_grid.station):
            generation_aggr.setdefault(gen.type, {})
            generation_aggr[gen.type].setdefault(gen.subtype, {'edisgo': 0})
            generation_aggr[gen.type][gen.subtype]['edisgo'] += gen.nominal_capacity
        generation.setdefault(gen.type, {})
        generation[gen.type].setdefault(gen.subtype, {'edisgo': 0})
        generation[gen.type][gen.subtype]['edisgo'] += gen.nominal_capacity

    # collect Dingo MV generation capacity
    for gen in dingo_mv_grid.generators():
        generation.setdefault(gen.type, {})
        generation[gen.type].setdefault(gen.subtype, {'dingo': 0})
        generation[gen.type][gen.subtype].setdefault('dingo', 0)
        generation[gen.type][gen.subtype]['dingo'] += gen.capacity

    # Collect Dingo LV generation capacity
    for la in dingo_mv_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                if la.is_aggregated:
                    generation_aggr.setdefault(gen.type, {})
                    generation_aggr[gen.type].setdefault(gen.subtype, {'dingo': 0})
                    generation_aggr[gen.type][gen.subtype].setdefault('dingo', 0)
                    generation_aggr[gen.type][gen.subtype]['dingo'] += gen.capacity
                generation.setdefault(gen.type, {})
                generation[gen.type].setdefault(gen.subtype, {'dingo': 0})
                generation[gen.type][gen.subtype].setdefault('dingo', 0)
                generation[gen.type][gen.subtype]['dingo'] += gen.capacity

    # Compare cumulative generation capacity
    for k1, v1 in generation.items():
        for k2, v2 in v1.items():
            if abs(v2['edisgo'] - v2['dingo']) > tol:
                raise ValueError(
                    'Generation capacity of {type} {subtype} does not match! '
                    '\n\tDingo:\t{dingo}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        type=k1,
                        subtype=k2,
                        dingo=v2['dingo'],
                        edisgo=v2['edisgo']))

    # Compare aggregated generation capacity
    for k1, v1 in generation_aggr.items():
        for k2, v2 in v1.items():
            if abs(v2['edisgo'] - v2['dingo']) > tol:
                raise ValueError(
                    'Aggregated generation capacity of {type} {subtype} does '
                    'not match! '
                    '\n\tDingo:\t{dingo}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        type=k1,
                        subtype=k2,
                        dingo=v2['dingo'],
                        edisgo=v2['edisgo']))