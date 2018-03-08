from ..grid.components import Load, Generator, BranchTee, MVStation, Line, \
    Transformer, LVStation, GeneratorFluctuating
from ..grid.grids import MVGrid, LVGrid
from ..grid.connect import connect_mv_generators, connect_lv_generators
from ..grid.tools import select_cable, position_switch_disconnectors
from ..tools.geo import proj2equidistant

from egoio.db_tables import model_draft, supply
from egoio.tools.db import connection

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from workalendar.europe import Germany
from demandlib import bdew as bdew, particular_profiles as profiles
import datetime

import pandas as pd
import numpy as np
import networkx as nx
from math import isnan
import random
import os

if not 'READTHEDOCS' in os.environ:
    from ding0.tools.results import load_nd_from_pickle
    from ding0.core.network.stations import LVStationDing0
    from ding0.core.structure.regions import LVLoadAreaCentreDing0
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


import logging
logger = logging.getLogger('edisgo')


def import_from_ding0(file, network):
    """
    Import an eDisGo grid topology from
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
        The eDisGo data container object

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

    # Make sure circuit breakers (respectively the rings) are closed
    ding0_mv_grid.close_circuit_breakers()

    # Import medium-voltage grid data
    network.mv_grid = _build_mv_grid(ding0_mv_grid, network)

    # Import low-voltage grid data
    lv_grids, lv_station_mapping, lv_grid_mapping = _build_lv_grid(
        ding0_mv_grid, network)

    # Assign lv_grids to network
    network.mv_grid.lv_grids = lv_grids

    # Integrate disconnecting points
    position_switch_disconnectors(network.mv_grid,
                                  mode=network.config['disconnecting_point'][
                                      'position'])

    # Check data integrity
    _validate_ding0_grid_import(network.mv_grid, ding0_mv_grid,
                                lv_grid_mapping)

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

                station = {repr(_): _ for _ in
                           network.mv_grid.graph.nodes_by_attribute(
                               'lv_station')}['LVStation_' + str(
                                    ding0_lv_grid._station.id_db)]

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

                # Create list of generator instances and add these to grid's
                # graph
                generators = {_: (GeneratorFluctuating(
                    id=_.id_db,
                    geom=_.geo_data,
                    nominal_capacity=_.capacity,
                    type=_.type,
                    subtype=_.subtype,
                    grid=lv_grid,
                    v_level=_.v_level) if _.type in ['wind', 'solar'] else
                                  Generator(
                                      id=_.id_db,
                                      geom=_.geo_data,
                                      nominal_capacity=_.capacity,
                                      type=_.type,
                                      subtype=_.subtype,
                                      grid=lv_grid,
                                      v_level=_.v_level))
                              for _ in ding0_lv_grid.generators()}
                lv_grid.graph.add_nodes_from(generators.values(),
                                             type='generator')

                # Create list of branch tee instances and add these to grid's
                # graph
                branch_tees = {
                    _: BranchTee(id=_.id_db,
                                 geom=_.geo_data,
                                 grid=lv_grid,
                                 in_building=_.in_building)
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
                              length=_['branch'].length / 1e3,
                              kind=_['branch'].kind,
                              grid=lv_grid)
                          })
                         for _ in edges]
                # convert voltage from V to kV
                for line in lines:
                    line[2]['line'].type['U_n'] = \
                        line[2]['line'].type['U_n'] / 1e3
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
        aggregated, aggr_stations, dingo_import_data = \
            _determine_aggregated_nodes(la_centers)
        network.dingo_import_data = dingo_import_data
    else:
        aggregated = {}
        aggr_stations = []

        # create empty DF for imported agg. generators
        network.dingo_import_data = pd.DataFrame(columns=('id',
                                                          'capacity',
                                                          'agg_geno')
                                                 )

    # Create list of load instances and add these to grid's graph
    loads = {_: Load(
        id=_.id_db,
        geom=_.geo_data,
        grid=grid,
        consumption=_.consumption) for _ in ding0_grid.loads()}
    grid.graph.add_nodes_from(loads.values(), type='load')

    # Create list of generator instances and add these to grid's graph
    generators = {_: (GeneratorFluctuating(
        id=_.id_db,
        geom=_.geo_data,
        nominal_capacity=_.capacity,
        type=_.type,
        subtype=_.subtype,
        grid=grid,
        v_level=_.v_level) if _.type in ['wind', 'solar'] else
                      Generator(
                          id=_.id_db,
                          geom=_.geo_data,
                          nominal_capacity=_.capacity,
                          type=_.type,
                          subtype=_.subtype,
                          grid=grid,
                          v_level=_.v_level))
                  for _ in ding0_grid.generators()}
    grid.graph.add_nodes_from(generators.values(), type='generator')

    # Create list of branch tee instances and add these to grid's graph
    branch_tees = {_: BranchTee(id=_.id_db,
                                geom=_.geo_data,
                                grid=grid,
                                in_building=False)
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
        grid=grid,
        transformers=[Transformer(
            mv_grid=grid,
            grid=grid,
            id='_'.join(['MVStation',
                         str(ding0_grid.station().id_db),
                         'transformer',
                         str(count)]),
            geom=ding0_grid.station().geo_data,
            voltage_op=_.v_level,
            type=pd.Series(dict(
                S_nom=_.s_max_a, X=_.x, R=_.r)))
            for (count, _) in enumerate(
                ding0_grid.station().transformers(), 1)])
    grid.graph.add_node(mv_station, type='mv_station')

    # Merge node above defined above to a single dict
    nodes = {**loads,
             **generators,
             **branch_tees,
             **stations,
             **{ding0_grid.station(): mv_station}}

    # Create list of line instances and add these to grid's graph
    lines = [(nodes[_['adj_nodes'][0]], nodes[_['adj_nodes'][1]],
              {'line': Line(
                  id=_['branch'].id_db,
                  type=_['branch'].type,
                  kind=_['branch'].kind,
                  length=_['branch'].length / 1e3,
                  grid=grid)
              })
             for _ in ding0_grid.graph_edges()
             if not any([isinstance(_['adj_nodes'][0], LVLoadAreaCentreDing0),
                        isinstance(_['adj_nodes'][1], LVLoadAreaCentreDing0)])]
    # set line name as series name
    for line in lines:
        line[2]['line'].type.name = line[2]['line'].type['name']
    grid.graph.add_edges_from(lines, type='line')

    # Assign reference to HV-MV station to MV grid
    grid._station = mv_station

    # Attach aggregated to MV station
    _attach_aggregated(network, grid, aggregated, ding0_grid)

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
        if gen.type not in aggr['generation'][gen.v_level]:
            aggr['generation'][gen.v_level][gen.type] = {}
        if gen.subtype not in aggr['generation'][gen.v_level][gen.type]:
            aggr['generation'][gen.v_level][gen.type].update(
                     {gen.subtype: {'ids': [gen.id_db],
                                'capacity': gen.capacity}})
        else:
            aggr['generation'][gen.v_level][gen.type][gen.subtype][
                'ids'].append(gen.id_db)
            aggr['generation'][gen.v_level][gen.type][gen.subtype][
                'capacity'] += gen.capacity

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

    dingo_import_data = pd.DataFrame(columns=('id',
                                              'capacity',
                                              'agg_geno')
                                     )

    for la_center in la_centers:
        aggr = {'generation': {}, 'load': {}, 'aggregates': []}

        # Determine aggregated generation in LV grid
        for lvgd in la_center.lv_load_area._lv_grid_districts:
            for gen in lvgd.lv_grid.generators():
                aggr = aggregate_generators(gen, aggr)

                dingo_import_data.loc[len(dingo_import_data)] = \
                    [int(gen.id_db),
                     gen.capacity,
                     None]

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
        aggregated.update({la_center.id_db: aggr})


    return aggregated, aggr_stations, dingo_import_data


def _attach_aggregated(network, grid, aggregated, ding0_grid):
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
            for type, val2 in val.items():
                for subtype, val3 in val2.items():
                    if type in ['solar', 'wind']:
                        gen = GeneratorFluctuating(
                            id='agg-' + str(la_id) + '-' + '_'.join(
                                [str(_) for _ in val3['ids']]),
                            nominal_capacity=val3['capacity'],
                            type=type,
                            subtype=subtype,
                            geom=grid.station.geom,
                            grid=grid,
                            v_level=4)
                    else:
                        gen = Generator(
                            id='agg-' + str(la_id) + '-' + '_'.join(
                                [str(_) for _ in val3['ids']]),
                            nominal_capacity=val3['capacity'],
                            type=type,
                            subtype=subtype,
                            geom=grid.station.geom,
                            grid=grid,
                            v_level=4)
                    grid.graph.add_node(gen, type='generator_aggr')

                    # backup reference of geno to LV geno list (save geno
                    # where the former LV genos are aggregated in)
                    network.dingo_import_data.set_value(network.dingo_import_data['id'].isin(val3['ids']),
                                                        'agg_geno',
                                                        gen)

                    # connect generator to MV station
                    line = Line(id='line_aggr_generator_la_' + str(la_id) + '_vlevel_{v_level}_'
                                '{subtype}'.format(
                                 v_level=v_level,
                                 subtype=subtype),
                                 type=aggr_line_type,
                                 kind='cable',
                                 length=1e-3,
                                 grid=grid)
                    grid.graph.add_edge(grid.station,
                                        gen,
                                        line=line,
                                        type='line_aggr')

        for sector, sectoral_load in la['load'].items():
            load = Load(
                geom=grid.station.geom,
                consumption={sector: sectoral_load},
                grid=grid,
                id='_'.join(['Load_aggregated', sector, repr(grid), str(la_id)]))

            grid.graph.add_node(load, type='load')

            # connect aggregated load to MV station
            line = Line(id='_'.join(['line_aggr_load_la_' + str(la_id), sector, str(la_id)]),
                        type=aggr_line_type,
                        kind='cable',
                        length=1e-3,
                        grid=grid)

            grid.graph.add_edge(grid.station,
                                load,
                                line=line,
                                type='line_aggr')


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
        grid.graph.nodes_by_attribute('mv_disconnecting_point'))

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
    # edges_w_la = grid.graph.lines()
    # data_integrity['line']['edisgo'] = len([_ for _ in edges_w_la
    #          if not (_['adj_nodes'][0] == grid.station or
    #                  _['adj_nodes'][1] == grid.station) and
    #          _['line']._length > .5])
    # data_integrity['line']['ding0'] = len(
    #     [_ for _ in ding0_grid.lines()
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
            grid.graph.nodes_by_attribute('generator') +
            grid.graph.nodes_by_attribute('generator_aggr'))
        data_integrity[grid]['generator']['ding0'] = len(
            list(lv_grid_mapping[grid].generators()))

        # Check number of loads
        data_integrity[grid]['load']['edisgo'] = len(
            grid.graph.nodes_by_attribute('load'))
        data_integrity[grid]['load']['ding0'] = len(
            list(lv_grid_mapping[grid].loads()))

        # Check number of lines outside aggregated LA
        data_integrity[grid]['line']['edisgo'] = len(
            list(grid.graph.lines()))
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
    gens_aggr = mv_grid.graph.nodes_by_attribute('generator_aggr')

    generation = {}
    generation_aggr = {}

    # collect eDisGo cumulative generation capacity
    for gen in mv_gens + lv_gens:
        generation.setdefault(gen.type, {})
        generation[gen.type].setdefault(gen.subtype, {'edisgo': 0})
        generation[gen.type][gen.subtype]['edisgo'] += gen.nominal_capacity
    for gen in gens_aggr:
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


def import_generators(network, data_source=None, file=None):
    """Import generator data from source.

    The generator data include

        * nom. capacity
        * type (TODO: specify!)
        * timeseries

    Additional data which can be processed (e.g. used in OEDB data) are

        * location
        * type
        * subtype
        * capacity

    Parameters
    ----------
    network: :class:`~.grid.network.Network`
        The eDisGo container object
    data_source: :obj:`str`
        Data source. Supported sources:

            * 'oedb'

    file: :obj:`str`
        File to import data from, required when using file-based sources.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        List of generators
    """

    if data_source == 'oedb':
        logging.warning('Right now only solar and wind generators can be '
                        'imported from the oedb.')
        _import_genos_from_oedb(network=network)
    elif data_source == 'pypsa':
        _import_genos_from_pypsa(network=network, file=file)
    else:
        logger.error("Invalid option {} for generator import. Must either be "
                     "'oedb' or 'pypsa'.".format(data_source))
        raise ValueError('The option you specified is not supported.')


def _import_genos_from_oedb(network):
    """Import generator data from the Open Energy Database (OEDB).

    The importer uses SQLAlchemy ORM objects.
    These are defined in ego.io,
    see https://github.com/openego/ego.io/tree/dev/egoio/db_tables

    Parameters
    ----------
    network: :class:`~.grid.network.Network`
        The eDisGo container object

    Notes
    ------
    Right now only solar and wind generators can be imported.

    """

    def _import_conv_generators():
        """Import conventional (conv) generators

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            List of medium-voltage generators

        Notes
        -----
        You can find a full list of columns in
        :func:`edisgo.data.import_data._update_grids`
        """

        # build query
        generators_sqla = session.query(
            orm_conv_generators.columns.id,
            orm_conv_generators.columns.subst_id,
            orm_conv_generators.columns.la_id,
            orm_conv_generators.columns.capacity,
            orm_conv_generators.columns.type,
            orm_conv_generators.columns.voltage_level,
            orm_conv_generators.columns.fuel,
            func.ST_AsText(func.ST_Transform(
                orm_conv_generators.columns.geom, srid))
        ). \
            filter(orm_conv_generators.columns.subst_id == network.mv_grid.id). \
            filter(orm_conv_generators.columns.voltage_level.in_([4, 5, 6, 7])). \
            filter(orm_conv_generators_version)

        # read data from db
        generators_mv = pd.read_sql_query(generators_sqla.statement,
                                          session.bind,
                                          index_col='id')

        return generators_mv

    def _import_res_generators(types_filter):
        """Import renewable (res) generators

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            List of medium-voltage generators
        :pandas:`pandas.DataFrame<dataframe>`
            List of low-voltage generators

        Notes
        -----
        You can find a full list of columns in
        :func:`edisgo.data.import_data._update_grids`

        If subtype is not specified it's set to 'unknown'.
        """

        # build basic query
        generators_sqla = session.query(
            orm_re_generators.columns.id,
            orm_re_generators.columns.subst_id,
            orm_re_generators.columns.la_id,
            orm_re_generators.columns.mvlv_subst_id,
            orm_re_generators.columns.electrical_capacity,
            orm_re_generators.columns.generation_type,
            orm_re_generators.columns.generation_subtype,
            orm_re_generators.columns.voltage_level,
            func.ST_AsText(func.ST_Transform(
                orm_re_generators.columns.rea_geom_new, srid)).label('geom'),
            func.ST_AsText(func.ST_Transform(
            orm_re_generators.columns.geom, srid)).label('geom_em')). \
                filter(orm_re_generators.columns.subst_id == network.mv_grid.id). \
            filter(orm_re_generators_version). \
            filter(types_filter)

        # extend basic query for MV generators and read data from db
        generators_mv_sqla = generators_sqla. \
            filter(orm_re_generators.columns.voltage_level.in_([4, 5]))
        generators_mv = pd.read_sql_query(generators_mv_sqla.statement,
                                          session.bind,
                                          index_col='id')

        # define generators with unknown subtype as 'unknown'
        generators_mv.loc[generators_mv[
                              'generation_subtype'].isnull(),
                          'generation_subtype'] = 'unknown'

        # extend basic query for LV generators and read data from db
        generators_lv_sqla = generators_sqla. \
            filter(orm_re_generators.columns.voltage_level.in_([6, 7]))
        generators_lv = pd.read_sql_query(generators_lv_sqla.statement,
                                       session.bind,
                                       index_col='id')

        # define generators with unknown subtype as 'unknown'
        generators_lv.loc[generators_lv[
                              'generation_subtype'].isnull(),
                          'generation_subtype'] = 'unknown'

        return generators_mv, generators_lv

    def _update_grids(network, generators_mv, generators_lv, remove_missing=True):
        """Update imported status quo DINGO-grid according to new generator dataset

        It
            * adds new generators to grid if they do not exist
            * updates existing generators if parameters have changed
            * removes existing generators from grid which do not exist in the imported dataset

        Steps:

            * Step 1: MV generators: Update existing, create new, remove decommissioned
            * Step 2: LV generators (single units): Update existing, remove decommissioned
            * Step 3: LV generators (in aggregated MV generators): Update existing,
                remove decommissioned
                (aggregated MV generators = originally LV generators from aggregated Load
                Areas which were aggregated during import from ding0.)
            * Step 4: LV generators (single units + aggregated MV generators): Create new

        Parameters
        ----------
        network: :class:`~.grid.network.Network`
            The eDisGo container object

        generators_mv: :pandas:`pandas.DataFrame<dataframe>`
            List of MV generators
            Columns:
                * id: :obj:`int` (index column)
                * electrical_capacity: :obj:`float` (unit: kW)
                * generation_type: :obj:`str` (e.g. 'solar')
                * generation_subtype: :obj:`str` (e.g. 'solar_roof_mounted')
                * voltage level: :obj:`int` (range: 4..7,)
                * geom: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)
                * geom_em: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)

        generators_lv: :pandas:`pandas.DataFrame<dataframe>`
            List of LV generators
            Columns:
                * id: :obj:`int` (index column)
                * mvlv_subst_id: :obj:`int` (id of MV-LV substation in grid
                  = grid which the generator will be connected to)
                * electrical_capacity: :obj:`float` (unit: kW)
                * generation_type: :obj:`str` (e.g. 'solar')
                * generation_subtype: :obj:`str` (e.g. 'solar_roof_mounted')
                * voltage level: :obj:`int` (range: 4..7,)
                * geom: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)
                * geom_em: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)

        remove_missing: :obj:`bool`
            If true, remove generators from grid which are not included in the imported dataset.
        """

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -4

        # get existing generators in MV and LV grids
        g_mv, g_lv, g_mv_agg = _build_generator_list(network=network)

        # print current capacity
        capacity_grid = 0
        capacity_grid += sum([row['obj'].nominal_capacity for id, row in g_mv.iterrows()])
        capacity_grid += sum([row['obj'].nominal_capacity for id, row in g_lv.iterrows()])
        capacity_grid += sum([row['obj'].nominal_capacity for id, row in g_mv_agg.iterrows()])
        logger.debug('Cumulative generator capacity (existing): {} kW'
                     .format(str(round(capacity_grid, 1)))
                     )

        # ======================================
        # Step 1: MV generators (existing + new)
        # ======================================
        logger.debug('==> MV generators')
        logger.debug('{} generators imported.'
                     .format(str(len(generators_mv))))
        # get existing genos (status quo DF format)
        g_mv_existing = g_mv[g_mv['id'].isin(list(generators_mv.index.values))]
        # get existing genos (new genos DF format)
        generators_mv_existing = generators_mv[generators_mv.index.isin(list(g_mv_existing['id']))]
        # remove existing ones from grid's geno list
        g_mv = g_mv[~g_mv.isin(g_mv_existing)].dropna()

        # TEMP: BACKUP 1 GENO FOR TESTING
        #temp_geno = generators_mv_existing.iloc[0]
        #temp_geno['geom_em'] = temp_geno['geom_em'].replace('10.667', '10.64')

        # iterate over exiting generators and check whether capacity has changed
        log_geno_count = 0
        log_geno_cap = 0
        for id, row in generators_mv_existing.iterrows():

            geno_existing = g_mv_existing[g_mv_existing['id'] == id]['obj'].iloc[0]

            # check if capacity equals; if not: update capacity
            if abs(row['electrical_capacity'] - \
                           geno_existing.nominal_capacity) < cap_diff_threshold:
                continue
            else:
                log_geno_cap += row['electrical_capacity'] - geno_existing.nominal_capacity
                log_geno_count += 1
                geno_existing.nominal_capacity = row['electrical_capacity']

                # check if cap=0 (this may happen if dp is buggy)
                if row['electrical_capacity'] <= 0:
                    geno_existing.grid.graph.remove_node(geno_existing)
                    logger.warning('Capacity of generator {} is <=0, generator removed. '
                                   'Check your data source.'
                                   .format(repr(geno_existing))
                                   )

        logger.debug('Capacities of {} of {} existing generators updated ({} kW).'
                     .format(str(log_geno_count),
                             str(len(generators_mv_existing) - log_geno_count),
                             str(round(log_geno_cap, 1))
                             )
                     )

        # new genos
        log_geno_count = 0
        log_geno_cap = 0
        generators_mv_new = generators_mv[~generators_mv.index.isin(
            list(g_mv_existing['id']))]

        # remove them from grid's geno list
        g_mv = g_mv[~g_mv.isin(list(generators_mv_new.index.values))].dropna()

        # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        #generators_mv_new = generators_mv_new.append(temp_geno)

        # iterate over new generators and create them
        for id, row in generators_mv_new.iterrows():
            # check if geom is available, skip otherwise
            geom = _check_geom(id, row)
            if not geom:
                logger.warning('Generator {} has no geom entry at all and will'
                               'not be imported!'.format(id))
                continue

            # create generator object and add it to MV grid's graph
            if row['generation_type'] in ['solar', 'wind']:
                network.mv_grid.graph.add_node(
                    GeneratorFluctuating(
                        id=id,
                        grid=network.mv_grid,
                        nominal_capacity=row['electrical_capacity'],
                        type=row['generation_type'],
                        subtype=row['generation_subtype'],
                        v_level=int(row['voltage_level']),
                        geom=wkt_loads(geom)),
                    type='generator')
            else:
                network.mv_grid.graph.add_node(
                    Generator(id=id,
                              grid=network.mv_grid,
                              nominal_capacity=row['electrical_capacity'],
                              type=row['generation_type'],
                              subtype=row['generation_subtype'],
                              v_level=int(row['voltage_level']),
                              geom=wkt_loads(geom)
                              ),
                    type='generator')
            log_geno_cap += row['electrical_capacity']
            log_geno_count += 1

        logger.debug('{} of {} new generators added ({} kW).'
                     .format(str(log_geno_count),
                             str(len(generators_mv_new)),
                             str(round(log_geno_cap, 1))
                             )
                     )

        # remove decommissioned genos
        # (genos which exist in grid but not in the new dataset)
        log_geno_cap = 0
        if not g_mv.empty and remove_missing:
            log_geno_count = 0
            for _, row in g_mv.iterrows():
                log_geno_cap += row['obj'].nominal_capacity
                row['obj'].grid.graph.remove_node(row['obj'])
                log_geno_count += 1
            logger.debug('{} of {} decommissioned generators removed ({} kW).'
                         .format(str(log_geno_count),
                                 str(len(g_mv)),
                                 str(round(log_geno_cap, 1))
                                 )
                         )

        # =============================================
        # Step 2: LV generators (single existing units)
        # =============================================
        logger.debug('==> LV generators')
        logger.debug('{} generators imported.'.format(str(len(generators_lv))))
        # get existing genos (status quo DF format)
        g_lv_existing = g_lv[g_lv['id'].isin(list(generators_lv.index.values))]
        # get existing genos (new genos DF format)
        generators_lv_existing = generators_lv[generators_lv.index.isin(list(g_lv_existing['id']))]

        # TEMP: BACKUP 1 GENO FOR TESTING
        # temp_geno = g_lv.iloc[0]

        # remove existing ones from grid's geno list
        g_lv = g_lv[~g_lv.isin(g_lv_existing)].dropna()

        # iterate over exiting generators and check whether capacity has changed
        log_geno_count = 0
        log_geno_cap = 0
        for id, row in generators_lv_existing.iterrows():

            geno_existing = g_lv_existing[g_lv_existing['id'] == id]['obj'].iloc[0]

            # check if capacity equals; if not: update capacity
            if abs(row['electrical_capacity'] - \
                           geno_existing.nominal_capacity) < cap_diff_threshold:
                continue
            else:
                log_geno_cap += row['electrical_capacity'] - geno_existing.nominal_capacity
                log_geno_count += 1
                geno_existing.nominal_capacity = row['electrical_capacity']

        logger.debug('Capacities of {} of {} existing generators (single units) updated ({} kW).'
                     .format(str(log_geno_count),
                             str(len(generators_lv_existing) - log_geno_count),
                             str(round(log_geno_cap, 1))
                             )
                     )

        # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        # g_lv.loc[len(g_lv)] = temp_geno

        # remove decommissioned genos
        # (genos which exist in grid but not in the new dataset)
        log_geno_cap = 0
        if not g_lv.empty and remove_missing:
            log_geno_count = 0
            for _, row in g_lv.iterrows():
                log_geno_cap += row['obj'].nominal_capacity
                row['obj'].grid.graph.remove_node(row['obj'])
                log_geno_count += 1
            logger.debug('{} of {} decommissioned generators (single units) removed ({} kW).'
                         .format(str(log_geno_count),
                                 str(len(g_lv)),
                                 str(round(log_geno_cap, 1))
                                 )
                         )

        # ====================================================================================
        # Step 3: LV generators (existing in aggregated units (originally from aggregated LA))
        # ====================================================================================
        g_lv_agg = network.dingo_import_data
        g_lv_agg_existing = g_lv_agg[g_lv_agg['id'].isin(list(generators_lv.index.values))]
        generators_lv_agg_existing = generators_lv[generators_lv.index.isin(list(g_lv_agg_existing['id']))]

        # TEMP: BACKUP 1 GENO FOR TESTING
        # temp_geno = g_lv_agg.iloc[0]

        g_lv_agg = g_lv_agg[~g_lv_agg.isin(g_lv_agg_existing)].dropna()

        log_geno_count = 0
        log_agg_geno_list = []
        log_geno_cap = 0
        for id, row in generators_lv_agg_existing.iterrows():

            # check if capacity equals; if not: update capacity off agg. geno
            cap_diff = row['electrical_capacity'] - \
                       g_lv_agg_existing[g_lv_agg_existing['id'] == id]['capacity'].iloc[0]
            if abs(cap_diff) < cap_diff_threshold:
                continue
            else:
                agg_geno = g_lv_agg_existing[g_lv_agg_existing['id'] == id]['agg_geno'].iloc[0]
                agg_geno.nominal_capacity += cap_diff
                log_geno_cap += cap_diff

                log_geno_count += 1
                log_agg_geno_list.append(agg_geno)

        logger.debug('Capacities of {} of {} existing generators (in {} of {} aggregated units) '
                     'updated ({} kW).'
                     .format(str(log_geno_count),
                             str(len(generators_lv_agg_existing) - log_geno_count),
                             str(len(set(log_agg_geno_list))),
                             str(len(g_lv_agg_existing['agg_geno'].unique())),
                             str(round(log_geno_cap, 1))
                             )
                     )

        # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        # g_lv_agg.loc[len(g_lv_agg)] = temp_geno

        # remove decommissioned genos
        # (genos which exist in grid but not in the new dataset)
        log_geno_cap = 0
        if not g_lv_agg.empty and remove_missing:
            log_geno_count = 0
            for _, row in g_lv_agg.iterrows():
                row['agg_geno'].nominal_capacity -= row['capacity']
                log_geno_cap += row['capacity']
                # remove LV geno id from id string of agg. geno
                id = row['agg_geno'].id.split('-')
                ids = id[2].split('_')
                ids.remove(str(int(row['id'])))
                row['agg_geno'].id = '-'.join([id[0], id[1], '_'.join(ids)])

                # after removing the LV geno from agg geno, is the agg. geno empty?
                # if yes, remove it from grid
                if not ids:
                    row['agg_geno'].grid.graph.remove_node(row['agg_geno'])

                log_geno_count += 1
            logger.debug('{} of {} decommissioned generators in aggregated generators removed ({} kW).'
                         .format(str(log_geno_count),
                                 str(len(g_lv_agg)),
                                 str(round(log_geno_cap, 1))
                                 )
                         )

        # ====================================================================
        # Step 4: LV generators (new single units + genos in aggregated units)
        # ====================================================================
        # new genos
        log_geno_count =\
            log_agg_geno_new_count =\
            log_agg_geno_upd_count = 0

        # TEMP: BACKUP 1 GENO FOR TESTING
        #temp_geno = generators_lv[generators_lv.index == g_lv_existing.iloc[0]['id']]

        generators_lv_new = generators_lv[~generators_lv.index.isin(list(g_lv_existing['id'])) &
                                          ~generators_lv.index.isin(list(g_lv_agg_existing['id']))]

        # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        #generators_lv_new = generators_lv_new.append(temp_geno)

        # dict for new agg. generators
        agg_geno_new = {}
        # get LV grid districts
        lv_grid_dict = _build_lv_grid_dict(network)

        # get predefined random seed and initialize random generator
        seed = int(network.config['grid_connection']['random_seed'])
        random.seed(a=seed)

        # check if none of new generators can be allocated to an existing  LV grid
        if not any([_ in lv_grid_dict.keys()
                    for _ in list(generators_lv_new['mvlv_subst_id'])]):
            logger.warning('None of the imported generators can be allocated '
                           'to an existing LV grid. Check compatibility of grid '
                           'and generator datasets.')

        # iterate over new (single unit or part of agg. unit) generators and create them
        log_geno_cap = 0
        for id, row in generators_lv_new.iterrows():
            lv_geno_added_to_agg_geno = False

            # new unit is part of agg. LA (mvlv_subst_id is different from existing
            # ones in LV grids of non-agg. load areas)
            if (row['mvlv_subst_id'] not in lv_grid_dict.keys() and
                    row['la_id'] and not isnan(row['la_id']) and
                    row['mvlv_subst_id'] and not isnan(row['mvlv_subst_id'])):

                # check if new unit can be added to existing agg. generator
                # (LA id, type and subtype match) -> update existing agg. generator.
                # Normally, this case should not occur since `subtype` of new genos
                # is set to a new value (e.g. 'solar')
                for _, agg_row in g_mv_agg.iterrows():
                    if (agg_row['la_id'] == int(row['la_id']) and
                            agg_row['obj'].type == row['generation_type'] and
                            agg_row['obj'].subtype == row['generation_subtype']):

                        agg_row['obj'].nominal_capacity += row['electrical_capacity']
                        agg_row['obj'].id += '_{}'.format(str(id))
                        log_agg_geno_upd_count += 1
                        lv_geno_added_to_agg_geno = True

                if not lv_geno_added_to_agg_geno:
                    la_id = int(row['la_id'])
                    if la_id not in agg_geno_new:
                        agg_geno_new[la_id] = {}
                    if row['voltage_level'] not in agg_geno_new[la_id]:
                        agg_geno_new[la_id][row['voltage_level']] = {}
                    if row['generation_type'] not in agg_geno_new[la_id][row['voltage_level']]:
                        agg_geno_new[la_id][row['voltage_level']][row['generation_type']] = {}
                    if row['generation_subtype'] not in \
                            agg_geno_new[la_id][row['voltage_level']][row['generation_type']]:
                        agg_geno_new[la_id][row['voltage_level']][row['generation_type']]\
                            .update({row['generation_subtype']: {'ids': [int(id)],
                                                                 'capacity': row['electrical_capacity']
                                                                 }
                             }
                        )
                    else:
                        agg_geno_new[la_id][row['voltage_level']][row['generation_type']] \
                            [row['generation_subtype']]['ids'].append(int(id))
                        agg_geno_new[la_id][row['voltage_level']][row['generation_type']] \
                            [row['generation_subtype']]['capacity'] += row['electrical_capacity']

            # new generator is a single (non-aggregated) unit
            else:
                # check if geom is available
                geom = _check_geom(id, row)

                if row['generation_type'] in ['solar', 'wind']:
                    gen = GeneratorFluctuating(
                        id=id,
                        grid=None,
                        nominal_capacity=row['electrical_capacity'],
                        type=row['generation_type'],
                        subtype=row['generation_subtype'],
                        v_level=int(row['voltage_level']),
                        geom=wkt_loads(geom) if geom else geom)
                else:
                    gen = Generator(id=id,
                                    grid=None,
                                    nominal_capacity=row[
                                        'electrical_capacity'],
                                    type=row['generation_type'],
                                    subtype=row['generation_subtype'],
                                    v_level=int(row['voltage_level']),
                                    geom=wkt_loads(geom) if geom else geom)

                # TEMP: REMOVE MVLV SUBST ID FOR TESTING
                #row['mvlv_subst_id'] = None

                # check if MV-LV substation id exists. if not, allocate to
                # random one
                lv_grid = _check_mvlv_subst_id(
                    generator=gen,
                    mvlv_subst_id=row['mvlv_subst_id'],
                    lv_grid_dict=lv_grid_dict)

                gen.grid = lv_grid

                lv_grid.graph.add_node(gen, type='generator')

                log_geno_count += 1
            log_geno_cap += row['electrical_capacity']

        # there are new agg. generators to be created
        if agg_geno_new:

            pfac_mv_gen = network.config['reactive_power_factor']['mv_gen']

            # add aggregated generators
            for la_id, val in agg_geno_new.items():
                for v_level, val2 in val.items():
                    for type, val3 in val2.items():
                        for subtype, val4 in val3.items():
                            gen = Generator(
                                id='agg-' + str(la_id) + '-' + '_'.join([
                                    str(_) for _ in val4['ids']]),
                                nominal_capacity=val4['capacity'],
                                type=type,
                                subtype=subtype,
                                geom=network.mv_grid.station.geom,
                                grid=network.mv_grid,
                                v_level=4)

                            network.mv_grid.graph.add_node(
                                gen, type='generator_aggr')

                            # select cable type
                            line_type, line_count = select_cable(
                                network=network,
                                level='mv',
                                apparent_power=gen.nominal_capacity /
                                pfac_mv_gen)

                            # connect generator to MV station
                            line = Line(id='line_aggr_generator_la_' + str(la_id) + '_vlevel_{v_level}_'
                                        '{subtype}'.format(
                                         v_level=v_level,
                                         subtype=subtype),
                                        type=line_type,
                                        kind='cable',
                                        quantity=line_count,
                                        length=1e-3,
                                        grid=network.mv_grid)

                            network.mv_grid.graph.add_edge(network.mv_grid.station,
                                                           gen,
                                                           line=line,
                                                           type='line_aggr')

                            log_agg_geno_new_count += len(val4['ids'])
                            log_geno_cap += val4['capacity']

        logger.debug('{} of {} new generators added ({} single units, {} to existing '
                     'agg. generators and {} units as new aggregated generators) '
                     '(total: {} kW).'
                     .format(str(log_geno_count +
                                 log_agg_geno_new_count +
                                 log_agg_geno_upd_count),
                             str(len(generators_lv_new)),
                             str(log_geno_count),
                             str(log_agg_geno_upd_count),
                             str(log_agg_geno_new_count),
                             str(round(log_geno_cap, 1))
                             )
                     )

    def _check_geom(id, row):
        """Checks if a valid geom is available in dataset

        If yes, this geom will be used.
        If not:

            * MV generators: use geom from EnergyMap.
            * LV generators: set geom to None. It is re-set in
                :func:`edisgo.data.import_data._check_mvlv_subst_id`
                to MV-LV station's geom. EnergyMap's geom is not used
                since it is more inaccurate than the station's geom.

        Parameters
        ----------
        id : :obj:`int`
            Id of generator
        row : :pandas:`pandas.Series<series>`
            Generator dataset

        Returns
        -------
        :shapely:`Shapely Point object<points>` or None
            Geom of generator. None, if no geom is available.
        """

        geom = None

        # check if geom is available
        if row['geom']:
            geom = row['geom']
        else:
            # MV generators: set geom to EnergyMap's geom, if available
            if int(row['voltage_level']) in [4,5]:
                # check if original geom from Energy Map is available
                if row['geom_em']:
                    geom = row['geom_em']
                    logger.debug('Generator {} has no geom entry, EnergyMap\'s geom entry will be used.'
                                 .format(id)
                                 )

        return geom

    def _check_mvlv_subst_id(generator, mvlv_subst_id, lv_grid_dict):
        """Checks if MV-LV substation id of single LV generator is missing or invalid.
        If so, a random one from existing stations in LV grids will be assigned.

        Parameters
        ----------
        generator : :class:`~.grid.components.Generator`
            LV generator
        mvlv_subst_id : :obj:`int`
            MV-LV substation id
        lv_grid_dict : :obj:`dict`
            Dict of existing LV grids
            Format: {:obj:`int`: :class:`~.grid.grids.LVGrid`}

        Returns
        -------
        :class:`~.grid.grids.LVGrid`
            LV grid of generator
        """

        if mvlv_subst_id and not isnan(mvlv_subst_id):
            # assume that given LA exists
            try:
                # get LV grid
                lv_grid = lv_grid_dict[mvlv_subst_id]

                # if no geom, use geom of station
                if not generator.geom:
                    generator.geom = lv_grid.station.geom
                    logger.debug('Generator {} has no geom entry, stations\' geom will be used.'
                                 .format(generator.id)
                                 )

                return lv_grid

            # if LA/LVGD does not exist, choose random LVGD and move generator to station of LVGD
            # this occurs due to exclusion of LA with peak load < 1kW
            except:
                lv_grid = random.choice(list(lv_grid_dict.values()))

                generator.geom = lv_grid.station.geom

                logger.warning('Generator {} cannot be assigned to '
                               'non-existent LV Grid and was '
                               'allocated to a random LV Grid ({}); '
                               'geom was set to stations\' geom.'
                               .format(repr(generator),
                                       repr(lv_grid)))
                pass
                return lv_grid

        else:
            lv_grid = random.choice(list(lv_grid_dict.values()))

            generator.geom = lv_grid.station.geom

            logger.warning('Generator {} has no mvlv_subst_id and was '
                           'allocated to a random LV Grid ({}); '
                           'geom was set to stations\' geom.'
                           .format(repr(generator),
                                   repr(lv_grid)))
            pass
            return lv_grid

    def _validate_generation():
        """Validate generators in updated grids

        The validation uses the cumulative capacity of all generators.
        """
        # TODO: Valdate conv. genos too!

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -4

        capacity_imported = generators_res_mv['electrical_capacity'].sum() + \
                            generators_res_lv['electrical_capacity'].sum() #+ \
                            #generators_conv_mv['capacity'].sum()

        capacity_grid = 0
        # MV genos
        for geno in network.mv_grid.graph.nodes_by_attribute('generator') +\
            network.mv_grid.graph.nodes_by_attribute('generator_aggr'):
            capacity_grid += geno.nominal_capacity

        # LV genos
        for lv_grid in network.mv_grid.lv_grids:
            for geno in lv_grid.graph.nodes_by_attribute('generator'):
                capacity_grid += geno.nominal_capacity

        logger.debug('Cumulative generator capacity (updated): {} kW'
                     .format(str(round(capacity_imported, 1)))
                     )

        if abs(capacity_imported - capacity_grid) > cap_diff_threshold:
            raise ValueError('Cumulative capacity of imported generators ({} kW) '
                             'differ from cumulative capacity of generators '
                             'in updated grid ({} kW) by {} kW.'
                             .format(str(round(capacity_imported, 1)),
                                     str(round(capacity_grid, 1)),
                                     str(round(capacity_imported - capacity_grid, 1))
                                     )
                             )
        else:
            logger.debug('Cumulative capacity of imported generators validated.')

    def _validate_sample_geno_location():
        if all(generators_res_lv['geom'].notnull()) \
                and all(generators_res_mv['geom'].notnull()) \
                and not generators_res_lv['geom'].empty \
                and not generators_res_mv['geom'].empty:
            # get geom of 1 random MV and 1 random LV generator and transform
            sample_mv_geno_geom_shp = transform(proj2equidistant(network),
                                                wkt_loads(generators_res_mv['geom']
                                                          .dropna()
                                                          .sample(n=1)
                                                          .item())
                                                )
            sample_lv_geno_geom_shp = transform(proj2equidistant(network),
                                                wkt_loads(generators_res_lv['geom']
                                                          .dropna()
                                                          .sample(n=1)
                                                          .item())
                                                )

            # get geom of MV grid district
            mvgd_geom_shp = transform(proj2equidistant(network),
                                      network.mv_grid.grid_district['geom']
                                      )

            # check if MVGD contains geno
            if not (mvgd_geom_shp.contains(sample_mv_geno_geom_shp) and
                        mvgd_geom_shp.contains(sample_lv_geno_geom_shp)):
                raise ValueError('At least one imported generator is not located '
                                 'in the MV grid area. Check compatibility of '
                                 'grid and generator datasets.')

    # make DB session
    conn = connection(section=network.config['db_connection']['section'])
    Session = sessionmaker(bind=conn)
    session = Session()

    srid = int(network.config['geo']['srid'])

    oedb_data_source = network.config['data_source']['oedb_data_source']
    scenario = network.generator_scenario

    if oedb_data_source == 'model_draft':

        # load ORM names
        orm_conv_generators_name = network.config['model_draft']['conv_generators_prefix'] + \
                                   scenario + \
                                   network.config['model_draft']['conv_generators_suffix']
        orm_re_generators_name = network.config['model_draft']['re_generators_prefix'] + \
                                 scenario + \
                                 network.config['model_draft']['re_generators_suffix']

        # import ORMs
        orm_conv_generators = model_draft.__getattribute__(orm_conv_generators_name)
        orm_re_generators = model_draft.__getattribute__(orm_re_generators_name)

        # set dummy version condition (select all generators)
        orm_conv_generators_version = 1 == 1
        orm_re_generators_version = 1 == 1

    elif oedb_data_source == 'versioned':

        # load ORM names
        orm_conv_generators_name = network.config['versioned']['conv_generators_prefix'] + \
                                   scenario + \
                                   network.config['versioned']['conv_generators_suffix']
        orm_re_generators_name = network.config['versioned']['re_generators_prefix'] + \
                                 scenario + \
                                 network.config['versioned']['re_generators_suffix']
        data_version = network.config['versioned']['version']

        # import ORMs
        orm_conv_generators = supply.__getattribute__(orm_conv_generators_name)
        orm_re_generators = supply.__getattribute__(orm_re_generators_name)

        # set version condition
        orm_conv_generators_version = orm_conv_generators.columns.version == data_version
        orm_re_generators_version = orm_re_generators.columns.version == data_version

    # Create filter for generation technologies
    # ToDo: This needs to be removed when all generators can be imported
    # (all generators in a scenario should be imported)
    types_condition = orm_re_generators.columns.generation_type.in_(
        ['solar', 'wind'])

    # get conventional and renewable generators
    #generators_conv_mv = _import_conv_generators()
    generators_res_mv, generators_res_lv = _import_res_generators(
        types_condition)

    #generators_mv = generators_conv_mv.append(generators_res_mv)

    _validate_sample_geno_location()

    _update_grids(network=network,
                  #generators_mv=generators_mv,
                  generators_mv=generators_res_mv,
                  generators_lv=generators_res_lv)

    _validate_generation()

    connect_mv_generators(network=network)
    connect_lv_generators(network=network)


def _import_genos_from_pypsa(network, file):
    """Import generator data from a pyPSA file.

    TBD

    Parameters
    ----------
    network: :class:`~.grid.network.Network`
        The eDisGo container object
    file: :obj:`str`
        File including path
    """
    raise NotImplementedError

    # generators = pd.read_csv(file,
    #                          comment='#',
    #                          index_col='name',
    #                          delimiter=',',
    #                          decimal='.'
    #                          )


def _build_generator_list(network):
    """Builds DataFrames with all generators in MV and LV grids

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
            A DataFrame with id of and reference to MV generators
    :pandas:`pandas.DataFrame<dataframe>`
            A DataFrame with id of and reference to LV generators
    :pandas:`pandas.DataFrame<dataframe>`
            A DataFrame with id of and reference to aggregated LV generators
    """

    genos_mv = pd.DataFrame(columns=
                            ('id', 'obj'))
    genos_lv = pd.DataFrame(columns=
                            ('id', 'obj'))
    genos_lv_agg = pd.DataFrame(columns=
                                ('la_id', 'id', 'obj'))

    # MV genos
    for geno in network.mv_grid.graph.nodes_by_attribute('generator'):
            genos_mv.loc[len(genos_mv)] = [int(geno.id), geno]
    for geno in network.mv_grid.graph.nodes_by_attribute('generator_aggr'):
            la_id = int(geno.id.split('-')[1].split('_')[-1])
            genos_lv_agg.loc[len(genos_lv_agg)] = [la_id, geno.id, geno]

    # LV genos
    for lv_grid in network.mv_grid.lv_grids:
        for geno in lv_grid.graph.nodes_by_attribute('generator'):
            genos_lv.loc[len(genos_lv)] = [int(geno.id), geno]

    return genos_mv, genos_lv, genos_lv_agg


def _build_lv_grid_dict(network):
    """Creates dict of LV grids

    LV grid ids are used as keys, LV grid references as values.

    Parameters
    ----------
    network: :class:`~.grid.network.Network`
        The eDisGo container object

    Returns
    -------
    :obj:`dict`
        Format: {:obj:`int`: :class:`~.grid.grids.LVGrid`}
    """

    lv_grid_dict = {}
    for lv_grid in network.mv_grid.lv_grids:
        lv_grid_dict[lv_grid.id] = lv_grid
    return lv_grid_dict


def import_feedin_timeseries(config_data, mv_grid_id, generator_scenario):
    """
    Import RES feed-in time series data and process

    Parameters
    ----------
    config_data : dict
        Dictionary containing config data from config files.
    mv_grid_id : :obj:`str`
        MV grid ID as used in oedb.
    generator_scenario : None or :obj:`str`
        Defines which scenario of future generator park to use.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Feedin time series
    """

    def _retrieve_timeseries_from_oedb(config_data, mv_grid_id,
                                       generator_scenario):
        """Retrieve time series from oedb

        Parameters
        ----------
        config_data : dict
            Dictionary containing config data from config files.
        mv_grid_id : :obj:`str`
            MV grid ID as used in oedb.
        generator_scenario : None or :obj:`str`
            Defines which scenario of future generator park to use.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Feedin time series
        """
        if config_data['versioned']['version'] == 'model_draft':
            orm_feedin_name = config_data['model_draft']['res_feedin_data']
            orm_feedin = model_draft.__getattribute__(orm_feedin_name)
            orm_feedin_version = 1 == 1
        else:
            orm_feedin_name = config_data['versioned']['res_feedin_data']
            # orm_feedin = supply.__getattribute__(orm_feedin_name)
            # TODO: remove workaround
            orm_feedin = model_draft.__getattribute__(orm_feedin_name)
            orm_feedin_version = 1 == 1
            # orm_feedin_version = orm_feedin.columns.version == scenario.config.data['versioned']['version']

        conn = connection(section=config_data['db_connection']['section'])
        Session = sessionmaker(bind=conn)
        session = Session()

        # TODO: add option to retrieve subset of time series
        feedin_sqla = session.query(
            orm_feedin.hour,
            orm_feedin.coastdat_id,
            orm_feedin.sub_id.label('subst_id'),
            orm_feedin.generation_type,
            orm_feedin.scenario,
            orm_feedin.feedin). \
            filter(orm_feedin.sub_id == mv_grid_id). \
            filter(orm_feedin.scenario.in_(generator_scenario)). \
            filter(orm_feedin_version)

        feedin = pd.read_sql_query(feedin_sqla.statement,
                                   session.bind,
                                   index_col='subst_id')

        # rename 'windonshore' to 'wind'
        feedin = feedin.replace({'generation_type': {'windonshore': 'wind'}})

        # average across different weather cells in grid district
        # TODO: replace this by using the specific time series for each generator when input tables are replaced are information on weather cells is available
        feedin = feedin.groupby(['hour', 'generation_type'],
                                as_index=False).mean()
        feedin.index = pd.date_range('1/1/2011', periods=8760, freq='H')
        return feedin

    feedin = _retrieve_timeseries_from_oedb(config_data, mv_grid_id,
                                            generator_scenario)
    gen_dict = {}
    for gen_type in feedin.generation_type.unique():
        gen_dict[gen_type] = feedin[
            feedin.generation_type==gen_type].sort_values(by='hour').set_index(
            'hour').feedin
    if gen_dict:
        return pd.DataFrame(gen_dict, index=gen_dict[gen_type].index)
    else:
        return None


def import_load_timeseries(config_data, data_source, mv_grid_id=None,
                           year=None):
    """
    Import load time series

    Parameters
    ----------
    config_data : dict
        Dictionary containing config data from config files.
    data_source : str
        Specify type of data source. Available data sources are

         * 'demandlib'
            Determine a load time series with the use of the demandlib.
            This calculates standard load profiles for 4 different sectors.

    mv_grid_id : :obj:`str`
        MV grid ID as used in oedb. Provide this if `data_source` is 'oedb'.
        Default: None.
    year : int
        Year for which to generate load time series. Provide this if
        `data_source` is 'demandlib'. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Load time series

    """

    def _import_load_timeseries_from_oedb(config_data, mv_grid_id):
        """
        Retrieve load time series from oedb

        Parameters
        ----------
        config_data : dict
            Dictionary containing config data from config files.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Load time series

        Notes
        ------
        This is currently not a valid option to retrieve load time series
        since time series in the oedb are not differentiated by sector. An
        issue concerning this has been created.

        """

        if config_data['versioned']['version'] == 'model_draft':
            orm_load_name = config_data['model_draft']['load_data']
            orm_load = model_draft.__getattribute__(orm_load_name)
            orm_load_areas_name = config_data['model_draft']['load_areas']
            orm_load_areas = model_draft.__getattribute__(orm_load_areas_name)
            orm_load_version = 1 == 1
        else:
            orm_load_name = config_data['versioned']['load_data']
            # orm_load = supply.__getattribute__(orm_load_name)
            # TODO: remove workaround
            orm_load = model_draft.__getattribute__(orm_load_name)
            # orm_load_version = orm_load.version == config.data['versioned']['version']

            orm_load_areas_name = config_data['versioned']['load_areas']
            # orm_load_areas = supply.__getattribute__(orm_load_areas_name)
            # TODO: remove workaround
            orm_load_areas = model_draft.__getattribute__(orm_load_areas_name)
            # orm_load_areas_version = orm_load.version == config.data['versioned']['version']

            orm_load_version = 1 == 1

        conn = connection(section=config_data['db_connection']['section'])
        Session = sessionmaker(bind=conn)
        session = Session()

        load_sqla = session.query(  # orm_load.id,
            orm_load.p_set,
            orm_load.q_set,
            orm_load_areas.subst_id). \
            join(orm_load_areas, orm_load.id == orm_load_areas.otg_id). \
            filter(orm_load_areas.subst_id == mv_grid_id). \
            filter(orm_load_version). \
            distinct()

        load = pd.read_sql_query(load_sqla.statement,
                                 session.bind,
                                 index_col='subst_id')
        return load

    def _load_timeseries_demandlib(config_data, year):
        """
        Get normalized sectoral load time series

        Time series are normalized to 1 kWh consumption per year

        Parameters
        ----------
        config_data : dict
            Dictionary containing config data from config files.
        year : int
            Year for which to generate load time series.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Load time series

        """

        sectoral_consumption = {'h0': 1, 'g0': 1, 'i0': 1, 'l0': 1}

        cal = Germany()
        holidays = dict(cal.holidays(year))

        e_slp = bdew.ElecSlp(year, holidays=holidays)

        # multiply given annual demand with timeseries
        elec_demand = e_slp.get_profile(sectoral_consumption)

        # Add the slp for the industrial group
        ilp = profiles.IndustrialLoadProfile(e_slp.date_time_index,
                                             holidays=holidays)

        # Beginning and end of workday, weekdays and weekend days, and scaling
        # factors by default
        elec_demand['i0'] = ilp.simple_profile(
            sectoral_consumption['i0'],
            am=datetime.time(config_data['demandlib']['day_start'].hour,
                             config_data['demandlib']['day_start'].minute, 0),
            pm=datetime.time(config_data['demandlib']['day_end'].hour,
                             config_data['demandlib']['day_end'].minute, 0),
            profile_factors=
            {'week': {'day': config_data['demandlib']['week_day'],
                      'night': config_data['demandlib']['week_night']},
             'weekend': {'day': config_data['demandlib']['weekend_day'],
                         'night': config_data['demandlib']['weekend_night']}})

        # Resample 15-minute values to hourly values and sum across sectors
        elec_demand = elec_demand.resample('H').mean()

        return elec_demand

    if data_source == 'oedb':
        load = _import_load_timeseries_from_oedb(config_data, mv_grid_id)
    elif data_source == 'demandlib':
        load = _load_timeseries_demandlib(config_data, year)
        load.rename(columns={'g0': 'retail', 'h0': 'residential',
                             'l0': 'agricultural', 'i0': 'industrial'},
                    inplace=True)
    return load
