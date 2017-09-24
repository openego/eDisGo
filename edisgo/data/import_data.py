from ..grid.components import Load, Generator, MVDisconnectingPoint, BranchTee,\
    MVStation, Line, Transformer, LVStation
from ..grid.grids import MVGrid, LVGrid
import pandas as pd
import numpy as np
import networkx as nx
import os
if not 'READTHEDOCS' in os.environ:
    from ding0.tools.results import load_nd_from_pickle
    from ding0.core.network.stations import LVStationDing0
    from ding0.core.structure.regions import LVLoadAreaCentreDing0


def import_from_ding0(file, network):
    """
    Import a eDisGo grid topology from
    `Ding0 data <https://github.com/openego/ding0>`_.

    This import method is specifically designed to load grid topology data in
    the format as `Ding0 <https://github.com/openego/ding0>`_ provides it via
    pickles.

    The import of the grid topology includes

        * the topology itself
        * equipment parameter
        * generators incl. location, type, subtype and capacity
        * loads incl. location and sectoral consumption

    Parameters
    ----------
    file: :obj:`str` or :class:`ding0.core.NetworkDing0`
        If a str is provided it is assumed it points to a pickle with Ding0
        grid data. This file will be read.
        If a object of the type :class:`ding0.core.NetworkDing0` data will be
        used directly from this object.
    network: :class:`~.grid.network.Network`
        The eDisGo container object

    Examples
    --------
    Assuming you the Ding0 `ding0_data.pkl` in CWD

    >>> from edisgo.grid.network import Network
    >>> network = Network.import_from_ding0('ding0_data.pkl'))

    Notes
    -----
    Assumes :class:`ding0.core.NetworkDing0` provided by `file` contains
    only data of one mv_grid_district.

    """
    # when `file` is a string, it will be read by the help of pickle
    if isinstance(file, str):
        ding0_nd = load_nd_from_pickle(filename=file)
    # otherwise it is assumed the object is passed directly
    else:
        ding0_nd = file

    ding0_mv_grid = ding0_nd._mv_grid_districts[0].mv_grid

    # Import medium-voltage grid data
    network.mv_grid =_build_mv_grid(ding0_mv_grid, network)

    # Import low-voltage grid data
    lv_grids, lv_station_mapping, lv_grid_mapping  = _build_lv_grid(ding0_mv_grid, network)

    # Assign lv_grids to network
    network.mv_grid.lv_grids = lv_grids

    # Check data integrity
    _validate_ding0_grid_import(network.mv_grid, ding0_mv_grid, lv_grid_mapping)

    # Set data source
    network.set_data_source('grid', 'dingo')

    # Set more params
    network._id = network.mv_grid.id


def _build_lv_grid(ding0_grid, network):
    """
    Build eDisGo LV grid from Ding0 data

    Parameters
    ----------
    ding0_grid: ding0.MVGridDing0
        Ding0 MV grid object

    Returns
    -------
    list of LVGrid
        LV grids
    dict
        Dictionary containing a mapping of LV stations in Ding0 to newly
        created eDisGo LV stations. This mapping is used to use the same
        instances of LV stations in the MV grid graph.
    """

    lv_station_mapping = {}
    lv_grids = []
    lv_grid_mapping = {}

    for la in ding0_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            ding0_lv_grid = lvgd.lv_grid
            if not ding0_lv_grid.grid_district.lv_load_area.is_aggregated:

                # Create LV grid instance
                lv_grid = LVGrid(
                    id=ding0_lv_grid.id_db,
                    geom=ding0_lv_grid.grid_district.geo_data,
                    grid_district={
                        'geom': ding0_lv_grid.grid_district.geo_data,
                        'population': ding0_lv_grid.grid_district.population},
                    voltage_nom=ding0_lv_grid.v_level / 1e3,
                    network=network)

                station = {repr(_):_
                           for _ in network.mv_grid.graph.nodes_by_attribute('lv_station')} \
                            ['LVStation_' + str(ding0_lv_grid._station.id_db)]

                station.grid = lv_grid
                for t in station.transformers:
                    t.grid = lv_grid

                lv_grid.graph.add_node(station, type='lv_station')
                lv_station_mapping.update({ding0_lv_grid._station: station})

                # Create list of load instances and add these to grid's graph
                loads = {_: Load(
                    id=_.id_db,
                    geom=_.geo_data,
                    grid=lv_grid,
                    consumption=_.consumption) for _ in ding0_lv_grid.loads()}
                lv_grid.graph.add_nodes_from(loads.values(), type='load')

                # Create list of generator instances and add these to grid's graph
                generators = {_: Generator(
                    id=_.id_db,
                    geom=_.geo_data,
                    nominal_capacity=_.capacity,
                    type=_.type,
                    subtype=_.subtype,
                    grid=lv_grid,
                    v_level=_.v_level) for _ in ding0_lv_grid.generators()}
                lv_grid.graph.add_nodes_from(generators.values(), type='generator')

                # Create list of branch tee instances and add these to grid's graph
                branch_tees = {
                    _: BranchTee(id=_.id_db, geom=_.geo_data, grid=lv_grid)
                    for _ in ding0_lv_grid._cable_distributors}
                lv_grid.graph.add_nodes_from(branch_tees.values(),
                                              type='branch_tee')

                # Merge node above defined above to a single dict
                nodes = {**loads,
                         **generators,
                         **branch_tees,
                         **{ding0_lv_grid._station: station}}

                edges = []
                edges_raw = list(nx.get_edge_attributes(
                    ding0_lv_grid._graph, 'branch').items())
                for edge in edges_raw:
                    edges.append({'adj_nodes': edge[0], 'branch': edge[1]})

                # Create list of line instances and add these to grid's graph
                lines = [(nodes[_['adj_nodes'][0]], nodes[_['adj_nodes'][1]],
                          {'line': Line(
                              id=_['branch'].id_db,
                              type=_['branch'].type,
                              length=_['branch'].length,
                              kind=_['branch'].kind,
                              grid=lv_grid)
                          })
                         for _ in edges]
                lv_grid.graph.add_edges_from(lines, type='line')

                # Add LV station as association to LV grid
                lv_grid._station = station

                # Add to lv grid mapping
                lv_grid_mapping.update({lv_grid: ding0_lv_grid})

                # Put all LV grid to a list of LV grids
                lv_grids.append(lv_grid)

    # TODO: don't forget to adapt lv stations creation in MV grid
    return lv_grids, lv_station_mapping, lv_grid_mapping


def _build_mv_grid(ding0_grid, network):
    """

    Parameters
    ----------
    ding0_grid: ding0.MVGridDing0
        Ding0 MV grid object
    network: Network
        The eDisGo container object

    Returns
    -------
    MVGrid
        A MV grid of class edisgo.grids.MVGrid is return. Data from the Ding0
        MV Grid object is translated to the new grid object.
    """

    # Instantiate a MV grid
    grid = MVGrid(
        id=ding0_grid.id_db,
        network=network,
        grid_district={'geom': ding0_grid.grid_district.geo_data,
                       'population':
                           sum([_.zensus_sum
                                for _ in
                                ding0_grid.grid_district._lv_load_areas
                                if not np.isnan(_.zensus_sum)])},
        voltage_nom=ding0_grid.v_level)

    # Special treatment of LVLoadAreaCenters see ...
    # TODO: add a reference above for explanation of how these are treated
    la_centers = [_ for _ in ding0_grid._graph.nodes()
                  if isinstance(_, LVLoadAreaCentreDing0)]
    if la_centers:
        aggregated, aggr_stations = _determine_aggregated_nodes(la_centers)
    else:
        aggregated = {}
        aggr_stations = []

    # Create list of load instances and add these to grid's graph
    loads = {_: Load(
        id=_.id_db,
        geom=_.geo_data,
        grid=grid,
        consumption=_.consumption) for _ in ding0_grid.loads()}
    grid.graph.add_nodes_from(loads.values(), type='load')

    # Create list of generator instances and add these to grid's graph
    generators = {_: Generator(
        id=_.id_db,
        geom=_.geo_data,
        nominal_capacity=_.capacity,
        type=_.type,
        subtype=_.subtype,
        grid=grid,
        v_level=_.v_level) for _ in ding0_grid.generators()}
    grid.graph.add_nodes_from(generators.values(), type='generator')

    # Create list of diconnection point instances and add these to grid's graph
    disconnecting_points = {_: MVDisconnectingPoint(id=_.id_db,
                                                 geom=_.geo_data,
                                                 state=_.status,
                                                 grid=grid)
                   for _ in ding0_grid._circuit_breakers}
    grid.graph.add_nodes_from(disconnecting_points.values(),
                              type='disconnection_point')

    # Create list of branch tee instances and add these to grid's graph
    branch_tees = {_: BranchTee(id=_.id_db, geom=_.geo_data, grid=grid)
                   for _ in ding0_grid._cable_distributors}
    grid.graph.add_nodes_from(branch_tees.values(), type='branch_tee')

    # Create list of LV station instances and add these to grid's graph
    stations = {_: LVStation(id=_.id_db,
                        geom=_.geo_data,
                        mv_grid=grid,
                        grid=None,  # (this will be set during LV import)
                        transformers=[Transformer(
                            mv_grid=grid,
                            grid=None,  # (this will be set during LV import)
                            id='_'.join(['LVStation',
                                        str(_.id_db),
                                        'transformer',
                                        str(count)]),
                            geom=_.geo_data,
                            voltage_op=t.v_level,
                            type=pd.Series(dict(
                                S_nom=t.s_max_a, X=t.x, R=t.r))
                        ) for (count, t) in enumerate(_.transformers(), 1)])
                for _ in ding0_grid._graph.nodes()
                if isinstance(_, LVStationDing0) and _ not in aggr_stations}

    grid.graph.add_nodes_from(stations.values(), type='lv_station')

    # Create HV-MV station add to graph
    mv_station = MVStation(
        id=ding0_grid.station().id_db,
        geom=ding0_grid.station().geo_data,
        transformers=[Transformer(
            mv_grid=grid,
            grid=grid,
            id='_'.join(['MV_station',
                         str(ding0_grid.station().id_db),
                         'transformer',
                         str(count)]),
            geom=ding0_grid.station().geo_data,
            voltage_op=_.v_level,
            type=pd.Series(dict(
                s=_.s_max_a, x=_.x, r=_.r)))
            for (count, _) in enumerate(ding0_grid.station().transformers())])
    grid.graph.add_node(mv_station, type='mv_station')

    # Merge node above defined above to a single dict
    nodes = {**loads,
             **generators,
             **disconnecting_points,
             **branch_tees,
             **stations,
             **{ding0_grid.station(): mv_station}}

    # Create list of line instances and add these to grid's graph
    lines = [(nodes[_['adj_nodes'][0]], nodes[_['adj_nodes'][1]],
              {'line': Line(
                  id=_['branch'].id_db,
                  type=_['branch'].type,
                  length=_['branch'].length,
                  grid=grid)
              })
             for _ in ding0_grid.graph_edges()
             if not any([isinstance(_['adj_nodes'][0], LVLoadAreaCentreDing0),
                        isinstance(_['adj_nodes'][1], LVLoadAreaCentreDing0)])]
    grid.graph.add_edges_from(lines, type='line')

    # Assign reference to HV-MV station to MV grid
    grid._station = mv_station

    # Attach aggregated to MV station
    _attach_aggregated(grid, aggregated, ding0_grid)

    return grid


def _determine_aggregated_nodes(la_centers):
    """Determine generation and load within load areas

    Parameters
    ----------
    la_centers: list of LVLoadAreaCentre
        Load Area Centers are Ding0 implementations for representating areas of
        high population density with high demand compared to DG potential.

    Notes
    -----
    Currently, MV grid loads are not considered in this aggregation function as
    Ding0 data does not come with loads in the MV grid level.

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
        gen: ding0.core.GeneratorDing0
            Ding0 Generator object
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
        la_center: LVLoadAreaCentreDing0
            Load area center object from Ding0

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

    aggregated = {}
    aggr_stations = []

    # TODO: The variable generation_aggr is further used -> delete this code
    generation_aggr = {}
    for la in la_centers[0].grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                if la.is_aggregated:
                    generation_aggr.setdefault(gen.type, {})
                    generation_aggr[gen.type].setdefault(gen.subtype, {'ding0': 0})
                    generation_aggr[gen.type][gen.subtype].setdefault('ding0', 0)
                    generation_aggr[gen.type][gen.subtype]['ding0'] += gen.capacity

    for la_center in la_centers:
        aggr = {'generation': {}, 'load': {}, 'aggregates': []}

        # Determine aggregated generation in LV grid
        for lvgd in la_center.lv_load_area._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                aggr = aggregate_generators(gen, aggr)

        # Determine aggregated load in MV grid
        # -> Implement once laods in Ding0 MV grids exist

        # Determine aggregated load in LV grid
        aggr = aggregate_loads(la_center, aggr)

        # Collect metadata of aggregated load areas
        aggr['aggregates'] = {
            'population': la_center.lv_load_area.zensus_sum,
            'geom': la_center.lv_load_area.geo_area}

        # Determine LV grids/ stations that are aggregated
        for _ in la_center.lv_load_area._lv_grid_districts:
            aggr_stations.append(_.lv_grid.station())

        # add elements to lists
        aggregated.update({repr(la_center): aggr})


    return aggregated, aggr_stations


def _attach_aggregated(grid, aggregated, ding0_grid):
    """Add Generators and Loads to MV station representing aggregated generation
    capacity and load

    Parameters
    ----------
    grid: MVGrid
        MV grid object
    aggregated: dict
        Information about aggregated load and generation capacity. For
        information about the structure of the dict see ... .
    ding0_grid: ding0.Network
        Ding0 network container
    Returns
    -------
    MVGrid
        Altered instance of MV grid including aggregated load and generation
    """

    aggr_line_type = ding0_grid.network._static_data['MV_cables'].iloc[
        ding0_grid.network._static_data['MV_cables']['I_max_th'].idxmax()]

    for la_id, la in aggregated.items():
        # add aggregated generators
        for v_level, val in la['generation'].items():
            for subtype, val2 in val.items():
                gen = Generator(
                    id='_'.join([la_id] + [str(_) for _ in val2['ids']]),
                    nominal_capacity=val2['capacity'],
                    type=val2['type'],
                    subtype=subtype,
                    geom=grid.station.geom,
                    grid=grid,
                    v_level=4)
                grid.graph.add_node(gen, type='generator')

                # connect generator to MV station
                line = {'line': Line(
                         id='line_aggr_generator_{LA}_vlevel_{v_level}_'
                            '{subtype}'.format(
                             v_level=v_level,
                             subtype=subtype,
                             LA=la_id),
                         type=aggr_line_type,
                         length=.5,
                         grid=grid)
                     }
                grid.graph.add_edge(grid.station, gen, line, type='line')
        for sector, sectoral_load in la['load'].items():
            load = Load(
                geom=grid.station.geom,
                consumption={sector: sectoral_load},
                grid=grid,
                id='_'.join(['Load_aggregated', sector, repr(grid), la_id]))

            grid.graph.add_node(load, type='load')

            # connect aggregated load to MV station
            line = {'line': Line(
                id='_'.join(['line_aggr_load', sector, la_id]),
                type=aggr_line_type,
                length=.5,
                grid=grid)
            }
            grid.graph.add_edge(grid.station, load, line, type='line')


def _validate_ding0_grid_import(mv_grid, ding0_mv_grid, lv_grid_mapping):
    """Cross-check imported data with original data source

    Parameters
    ----------
    mv_grid: MVGrid
        eDisGo MV grid instance
    ding0_mv_grid: MVGridDing0
        Ding0 MV grid instance
    lv_grid_mapping: dict
        Translates Ding0 LV grids to associated, newly created eDisGo LV grids
    """

    # Check number of components in MV grid
    _validate_ding0_mv_grid_import(mv_grid, ding0_mv_grid)

    # Check number of components in LV grid
    _validate_ding0_lv_grid_import(mv_grid.lv_grids, ding0_mv_grid,
                                   lv_grid_mapping)

    # Check cumulative load and generation in MV grid district
    _validate_load_generation(mv_grid, ding0_mv_grid)



def _validate_ding0_mv_grid_import(grid, ding0_grid):
    """Verify imported data with original data from Ding0

    Parameters
    ----------
    grid: MVGrid
        MV Grid data (eDisGo)
    ding0_grid: ding0.MVGridDing0
        Ding0 MV grid object

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
    data_integrity.update({_: {'ding0': None, 'edisgo': None, 'msg': None}
                           for _ in integrity_checks})

    # Check number of branch tees
    data_integrity['branch_tee']['ding0'] = len(ding0_grid._cable_distributors)
    data_integrity['branch_tee']['edisgo'] = len(
        grid.graph.nodes_by_attribute('branch_tee'))

    # Check number of disconnecting points
    data_integrity['disconnection_point']['ding0'] = len(
        ding0_grid._circuit_breakers)
    data_integrity['disconnection_point']['edisgo'] = len(
        grid.graph.nodes_by_attribute('disconnection_point'))

    # Check number of MV transformers
    data_integrity['mv_transformer']['ding0'] = len(
        list(ding0_grid.station().transformers()))
    data_integrity['mv_transformer']['edisgo'] = len(
        grid.station.transformers)

    # Check number of LV stations in MV grid (graph)
    data_integrity['lv_station']['edisgo'] = len(grid.graph.nodes_by_attribute(
        'lv_station'))
    data_integrity['lv_station']['ding0'] = len(
        [_ for _ in ding0_grid._graph.nodes()
         if (isinstance(_, LVStationDing0) and
             not _.grid.grid_district.lv_load_area.is_aggregated)])

    # Check number of lines outside aggregated LA
    # edges_w_la = grid.graph.graph_edges()
    # data_integrity['line']['edisgo'] = len([_ for _ in edges_w_la
    #          if not (_['adj_nodes'][0] == grid.station or
    #                  _['adj_nodes'][1] == grid.station) and
    #          _['line']._length > .5])
    # data_integrity['line']['ding0'] = len(
    #     [_ for _ in ding0_grid.graph_edges()
    #      if not _['branch'].connects_aggregated])

    # raise an error if data does not match
    for c in integrity_checks:
        if data_integrity[c]['edisgo'] != data_integrity[c]['ding0']:
            raise ValueError(
                'Unequal number of objects for {c}. '
                '\n\tDing0:\t{ding0_no}'
                '\n\teDisGo:\t{edisgo_no}'.format(
                    c=c,
                    ding0_no=data_integrity[c]['ding0'],
                    edisgo_no=data_integrity[c]['edisgo']))

    return data_integrity


def _validate_ding0_lv_grid_import(grids, ding0_grid, lv_grid_mapping):
    """Verify imported data with original data from Ding0

    Parameters
    ----------
    grids: list of LVGrid
        LV Grid data (eDisGo)
    ding0_grid: ding0.MVGridDing0
        Ding0 MV grid object
    lv_grid_mapping: dict
        Defines relationship between Ding0 and eDisGo grid objects

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

        data_integrity.update({grid:{_: {'ding0': None, 'edisgo': None, 'msg': None}
                           for _ in integrity_checks}})

        # Check number of branch tees
        data_integrity[grid]['branch_tee']['ding0'] = len(
            lv_grid_mapping[grid]._cable_distributors)
        data_integrity[grid]['branch_tee']['edisgo'] = len(
            grid.graph.nodes_by_attribute('branch_tee'))

        # Check number of LV transformers
        data_integrity[grid]['lv_transformer']['ding0'] = len(
            list(lv_grid_mapping[grid].station().transformers()))
        data_integrity[grid]['lv_transformer']['edisgo'] = len(
            grid.station.transformers)

        # Check number of generators
        data_integrity[grid]['generator']['edisgo'] = len(
            grid.graph.nodes_by_attribute('generator'))
        data_integrity[grid]['generator']['ding0'] = len(
            list(lv_grid_mapping[grid].generators()))

        # Check number of loads
        data_integrity[grid]['load']['edisgo'] = len(
            grid.graph.nodes_by_attribute('load'))
        data_integrity[grid]['load']['ding0'] = len(
            list(lv_grid_mapping[grid].loads()))

        # Check number of lines outside aggregated LA
        data_integrity[grid]['line']['edisgo'] = len(
            list(grid.graph.graph_edges()))
        data_integrity[grid]['line']['ding0'] = len(
            [_ for _ in lv_grid_mapping[grid].graph_edges()
             if not _['branch'].connects_aggregated])

    # raise an error if data does not match
    for grid in grids:
        for c in integrity_checks:
            if data_integrity[grid][c]['edisgo'] != data_integrity[grid][c]['ding0']:
                raise ValueError(
                    'Unequal number of objects in grid {grid} for {c}. '
                    '\n\tDing0:\t{ding0_no}'
                    '\n\teDisGo:\t{edisgo_no}'.format(
                        grid=grid,
                        c=c,
                        ding0_no=data_integrity[grid][c]['ding0'],
                        edisgo_no=data_integrity[grid][c]['edisgo']))


def _validate_load_generation(mv_grid, ding0_mv_grid):
    """

    Parameters
    ----------
    mv_grid
    ding0_mv_grid

    Notes
    -----
    Only loads in LV grids are compared as currently Ding0 does not have MV
    connected loads
    """

    decimal_places = 6
    tol = 10 ** -decimal_places

    sectors = ['retail', 'industrial', 'agricultural', 'residential']
    consumption = {_: {'edisgo': 0, 'ding0':0} for _ in sectors}


    # Collect eDisGo LV loads
    for lv_grid in mv_grid.lv_grids:
        for load in lv_grid.graph.nodes_by_attribute('load'):
            for s in sectors:
                consumption[s]['edisgo'] += load.consumption.get(s, 0)

    # Collect Ding0 LV loads
    for la in ding0_mv_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for load in lvgd.lv_grid.loads():
                for s in sectors:
                    consumption[s]['ding0'] += load.consumption.get(s, 0)

    # Compare cumulative load
    for k, v in consumption.items():
            if v['edisgo'] != v['ding0']:
                raise ValueError(
                    'Consumption for {sector} does not match! '
                    '\n\tDing0:\t{ding0}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        sector=k,
                        ding0=v['ding0'],
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
        if gen in mv_grid.graph.neighbors(mv_grid.station) and \
            mv_grid.graph.get_edge_data(mv_grid.station,gen)['line'].length <= .5:
            generation_aggr.setdefault(gen.type, {})
            generation_aggr[gen.type].setdefault(gen.subtype, {'edisgo': 0})
            generation_aggr[gen.type][gen.subtype]['edisgo'] += gen.nominal_capacity
        generation.setdefault(gen.type, {})
        generation[gen.type].setdefault(gen.subtype, {'edisgo': 0})
        generation[gen.type][gen.subtype]['edisgo'] += gen.nominal_capacity

    # collect Ding0 MV generation capacity
    for gen in ding0_mv_grid.generators():
        generation.setdefault(gen.type, {})
        generation[gen.type].setdefault(gen.subtype, {'ding0': 0})
        generation[gen.type][gen.subtype].setdefault('ding0', 0)
        generation[gen.type][gen.subtype]['ding0'] += gen.capacity

    # Collect Ding0 LV generation capacity
    for la in ding0_mv_grid.grid_district._lv_load_areas:
        for lvgd in la._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                if la.is_aggregated:
                    generation_aggr.setdefault(gen.type, {})
                    generation_aggr[gen.type].setdefault(gen.subtype, {'ding0': 0})
                    generation_aggr[gen.type][gen.subtype].setdefault('ding0', 0)
                    generation_aggr[gen.type][gen.subtype]['ding0'] += gen.capacity
                generation.setdefault(gen.type, {})
                generation[gen.type].setdefault(gen.subtype, {'ding0': 0})
                generation[gen.type][gen.subtype].setdefault('ding0', 0)
                generation[gen.type][gen.subtype]['ding0'] += gen.capacity

    # Compare cumulative generation capacity
    for k1, v1 in generation.items():
        for k2, v2 in v1.items():
            if abs(v2['edisgo'] - v2['ding0']) > tol:
                raise ValueError(
                    'Generation capacity of {type} {subtype} does not match! '
                    '\n\tDing0:\t{ding0}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        type=k1,
                        subtype=k2,
                        ding0=v2['ding0'],
                        edisgo=v2['edisgo']))

    # Compare aggregated generation capacity
    for k1, v1 in generation_aggr.items():
        for k2, v2 in v1.items():
            if abs(v2['edisgo'] - v2['ding0']) > tol:
                raise ValueError(
                    'Aggregated generation capacity of {type} {subtype} does '
                    'not match! '
                    '\n\tDing0:\t{ding0}'
                    '\n\teDisGo:\t{edisgo}'.format(
                        type=k1,
                        subtype=k2,
                        ding0=v2['ding0'],
                        edisgo=v2['edisgo']))
