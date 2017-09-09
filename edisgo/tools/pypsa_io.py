"""
This modules provides tools to convert graph based representation of the grid
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA grid
container.
"""


import pandas as pd
from math import pi, sqrt, floor
from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe
from networkx import connected_component_subgraphs


def to_pypsa(network, mode):
    """
    Translate graph based grid representation to PyPSA Network

    For details from a user perspective see API documention of
    :meth:`~.grid.network.Network.analyze` of the grid container
    :class:`~.grid.network.Network`.

    Translating eDisGo's grid topology to PyPSA representation is structured
    into tranlating the topology and adding time series for components of the
    grid. In both cases translation of MV grid only (`mode='mv'`), LV grid only
    (`mode='lv'`), MV and LV (`mode=None`) share some code. The
    code is organized as follows

    * Medium-voltage only (`mode='mv'`): All medium-voltage grid components are
      exported by :func:`mv_to_pypsa` including the LV station. LV grid load
      and generation is considered using :func:`add_aggregated_lv_components`.
      Time series are collected by `_pypsa_load_timeseries` (as example
      for loads, generators and buses) specifying `mode='mv'). Timeseries
      for aggregated load/generation at substations are determined individually.
    * Low-voltage only (`mode='lv'`): LV grid topology including the MV-LV
      transformer is exported. The slack is defind at primary side of the MV-LV
      transformer.
    * Both level MV+LV (`mode=None`): The entire grid topology is translated to
      PyPSA in order to perform a complete power flow analysis in both levels
      together. First, both grid levels are translated seperately using
      :func:`mv_to_pypsa` and :func:`lv_to_pypsa`. Those are merge by
      :func:`combine_mv_and_lv`. Time series are obtained at once for both grid
      levels.

    This PyPSA interface is aware of translation errors and performs so checks
    on integrity of data converted to PyPSA grid representation

    * Sub-graphs/ Sub-networks: It is ensured the grid has no islanded parts
    * Completeness of time series: It is ensured each component has a time
      series
    * Buses available: Each component (load, generator, line, transformer) is
      connected to a bus. The PyPSA representation is check for completeness of
      buses.

    Parameters
    ----------
    network : Network
        eDisGo grid container
    mode : str
        Determines grid levels that are translated to
        `PyPSA grid representation
        <https://www.pypsa.org/doc/components.html#network>`_. Specify

        * 'mv' to export MV grid level only. This includes cumulative load and
          generation from underlying LV grid aggregated at respective LV
          station.
        * 'lv' to export LV grid level only

    Returns
    -------

        PyPSA Network
    """

    # get topology and time series data
    if mode is None:
        mv_components = mv_to_pypsa(network)
        lv_components = lv_to_pypsa(network)
        components = combine_mv_and_lv(mv_components,
                                       lv_components)
    elif mode is 'mv':
        mv_components = mv_to_pypsa(network)
        components = add_aggregated_lv_components(
            network,
            mv_components)

        timeseries_load_p, timeseries_load_q = _pypsa_load_timeseries(
            network,
            mode=mode)

        timeseries_gen_p, timeseries_gen_q = _pypsa_generator_timeseries(
            network,
            mode=mode)

        timeseries_bus_v_set = _pypsa_bus_timeseries(
            network,
            components['Bus'].index.tolist())

        ts_gen_lv_aggr_p, \
        ts_gen_lv_aggr_q, \
        ts_load_lv_aggr_p, ts_load_lv_aggr_q = _pypsa_timeseries_aggregated_at_lv_station(
            network)

        # Concat MV and LV (aggregated) time series
        timeseries_load_p = pd.concat([timeseries_load_p, ts_load_lv_aggr_p],
                                      axis=1)
        timeseries_load_q = pd.concat([timeseries_load_q, ts_load_lv_aggr_q],
                                      axis=1)
        timeseries_gen_p = pd.concat([timeseries_gen_p, ts_gen_lv_aggr_p],
                                     axis=1)
        timeseries_gen_q = pd.concat([timeseries_gen_q, ts_gen_lv_aggr_q],
                                     axis=1)
    elif mode is 'lv':
        raise NotImplementedError
        lv_to_pypsa(network)
    else:
        raise ValueError("Provide proper mode or leave it empty to export "
                         "entire grid topology.")

    # check topology
    _check_topology(components)

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.set_snapshots(network.scenario.timeseries.timeindex)

    # import grid topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(components['Bus'], 'Bus')

    for k, comps in components.items():
        if k is not 'Bus':
            pypsa_network.import_components_from_dataframe(comps, k)

    # import time series to PyPSA network
    import_series_from_dataframe(pypsa_network,
                                 timeseries_gen_p,
                                 'Generator',
                                 'p_set')
    import_series_from_dataframe(pypsa_network,
                                 timeseries_gen_q,
                                 'Generator',
                                 'q_set')
    import_series_from_dataframe(pypsa_network,
                                 timeseries_load_p,
                                 'Load',
                                 'p_set')
    import_series_from_dataframe(pypsa_network,
                                 timeseries_load_q,
                                 'Load',
                                 'q_set')

    import_series_from_dataframe(pypsa_network,
                                 timeseries_bus_v_set,
                                 'Bus',
                                 'v_mag_pu_set')

    _check_integrity_of_pypsa(pypsa_network)

    return pypsa_network


def mv_to_pypsa(network):
    """Translate MV grid topology representation to PyPSA format

    MV grid topology translated here includes

    * MV station (no transformer, see :meth:`~.grid.network.Network.analyze`)
    * Loads, Generators, Lines, Branch Tees of MV grid level as well as LV
      stations. LV stations do not have load and generation of LV level.

    Parameters
    ----------
    network : Network
        eDisGo grid container
    """

    generators = network.mv_grid.graph.nodes_by_attribute('generator')
    loads = network.mv_grid.graph.nodes_by_attribute('load')
    branch_tees = network.mv_grid.graph.nodes_by_attribute('branch_tee')
    lines = network.mv_grid.graph.graph_edges()
    lv_stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    mv_stations = network.mv_grid.graph.nodes_by_attribute('mv_station')

    omega = 2 * pi * 50

    # define required dataframe columns for components
    generator = {'name': [],
                 'bus': [],
                 'control': [],
                 'p_nom': [],
                 'type': []}

    bus = {'name': [], 'v_nom': []}

    load = {'name': [], 'bus': []}

    line = {'name': [],
            'bus0': [],
            'bus1': [],
            'type': [],
            'x': [],
            'r': [],
            's_nom': [],
            'length': []}

    transformer = {'name': [],
                   'bus0': [],
                   'bus1': [],
                   'type': [],
                   'model': [],
                   'x': [],
                   'r': [],
                   's_nom': [],
                   'tap_ratio': []}

    # create dataframe representing generators and associated buses
    for gen in generators:
        bus_name = '_'.join(['Bus', repr(gen)])
        generator['name'].append(repr(gen))
        generator['bus'].append(bus_name)
        # TODO: revisit 'control' for dispatchable power plants
        generator['control'].append('PQ')
        generator['p_nom'].append(gen.nominal_capacity)
        generator['type'].append('_'.join([gen.type, gen.subtype]))

        bus['name'].append(bus_name)
        bus['v_nom'].append(gen.grid.voltage_nom)

    # create dataframe representing branch tees
    for bt in branch_tees:
        bus['name'].append('_'.join(['Bus', repr(bt)]))
        bus['v_nom'].append(bt.grid.voltage_nom)

    # create dataframes representing loads and associated buses
    for lo in loads:
        bus_name = '_'.join(['Bus', repr(lo)])
        load['name'].append(repr(lo))
        load['bus'].append(bus_name)

        bus['name'].append(bus_name)
        bus['v_nom'].append(lo.grid.voltage_nom)

    # create dataframe for lines
    for l in lines:
        line['name'].append(repr(l['line']))

        if l['adj_nodes'][0] in lv_stations:
            line['bus0'].append(
                '_'.join(['Bus', l['adj_nodes'][0].__repr__(side='mv')]))
        elif l['adj_nodes'][0] is network.mv_grid.station:
            line['bus0'].append(
                '_'.join(['Bus', l['adj_nodes'][0].__repr__(side='lv')]))
        else:
            line['bus0'].append('_'.join(['Bus', repr(l['adj_nodes'][0])]))

        if l['adj_nodes'][1] in lv_stations:
            line['bus1'].append(
                '_'.join(['Bus', l['adj_nodes'][1].__repr__(side='mv')]))
        elif l['adj_nodes'][1] is network.mv_grid.station:
            line['bus1'].append(
                '_'.join(['Bus', l['adj_nodes'][1].__repr__(side='lv')]))
        else:
            line['bus1'].append('_'.join(['Bus', repr(l['adj_nodes'][1])]))

        line['type'].append("")
        line['x'].append(
            l['line'].type['L'] * omega / 1e3 * l['line'].length / 1e3)
        line['r'].append(l['line'].type['R'] * l['line'].length / 1e3)
        line['s_nom'].append(
            sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
        line['length'].append(l['line'].length / 1e3)

    # create dataframe for LV stations incl. primary/secondary side bus
    for lv_st in lv_stations:
        transformer_count = 1
        # add primary side bus (bus0)
        bus0_name = '_'.join(['Bus', lv_st.__repr__(side='mv')])
        bus['name'].append(bus0_name)
        bus['v_nom'].append(lv_st.grid.voltage_nom)

        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', lv_st.__repr__(side='lv')])
        bus['name'].append(bus1_name)
        bus['v_nom'].append(lv_st.transformers[0].voltage_op)

        for tr in lv_st.transformers:
            transformer['name'].append(
                '_'.join([repr(lv_st), 'transformer', str(transformer_count)]))
            transformer['bus0'].append(bus0_name)
            transformer['bus1'].append(bus1_name)
            transformer['type'].append("")
            transformer['model'].append('pi')
            transformer['r'].append(tr.type.r)
            transformer['x'].append(tr.type.x)
            transformer['s_nom'].append(tr.type.s / 1e3)
            transformer['tap_ratio'].append(1)

            transformer_count += 1

    # create dataframe for MV stations (only secondary side bus)
    for mv_st in mv_stations:
        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', mv_st.__repr__(side='mv')])
        bus['name'].append(bus1_name)
        bus['v_nom'].append(mv_st.transformers[0].voltage_op)

    # Add separate slack generator at MV station secondary side bus bar
    generator['name'].append("Generator_slack")
    generator['bus'].append(bus1_name)
    generator['control'].append('Slack')
    generator['p_nom'].append(0)
    generator['type'].append('Slack generator')

    components = {
        'Generator': pd.DataFrame(generator).set_index('name'),
        'Bus': pd.DataFrame(bus).set_index('name'),
        'Load': pd.DataFrame(load).set_index('name'),
        'Line': pd.DataFrame(line).set_index('name'),
        'Transformer': pd.DataFrame(transformer).set_index('name')}

    return components


def lv_to_pypsa():
    """
    Convert LV grid topology to PyPSA representation

    Returns
    -------

    """


def combine_mv_and_lv():
    """Combine MV and LV grid topology in PyPSA format

    Idea for implementation
    -----------------------
    Merge all DataFrames except for LV transformers which are already included
    in the MV grid PyPSA representation

    """
    pass


def add_aggregated_lv_components(network, components):
    """
    Aggregates LV load and generation at LV stations

    Use this function if you aim for MV calculation only. The according
    DataFrames of `components` are extended by load and generators representing
    these aggregated respecting the technology type.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    components : dict of :pandas:`pandas.DataFrame<dataframe>`
        PyPSA components in tabular format

    Returns
    -------
    :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        The dictionary components passed to the function is returned altered.
    """
    generators = {}

    loads = {}

    # collect aggregated generation capacity by type and subtype
    # collect aggregated load grouped by sector
    for lv_grid in network.mv_grid.lv_grids:
        generators.setdefault(lv_grid, {})
        for gen in lv_grid.graph.nodes_by_attribute('generator'):
            generators[lv_grid].setdefault(gen.type, {})
            generators[lv_grid][gen.type].setdefault(gen.subtype, {})
            generators[lv_grid][gen.type][gen.subtype].setdefault(
                'capacity', 0)
            generators[lv_grid][gen.type][gen.subtype][
                'capacity'] += gen.nominal_capacity
            generators[lv_grid][gen.type][gen.subtype].setdefault(
                'name',
                '_'.join([gen.type,
                          gen.subtype,
                          'aggregated',
                          'LV_grid',
                          str(lv_grid.id)]))
        loads.setdefault(lv_grid, {})
        for lo in lv_grid.graph.nodes_by_attribute('load'):
            for sector, val in lo.consumption.items():
                loads[lv_grid].setdefault(sector, 0)
                loads[lv_grid][sector] += val


    # define dict for DataFrame creation of aggr. generation and load
    generator = {'name': [],
                 'bus': [],
                 'control': [],
                 'p_nom': [],
                 'type': []}

    load = {'name': [], 'bus': []}

    # fill generators dictionary for DataFrame creation
    for lv_grid_obj, lv_grid in generators.items():
        for _, gen_type in lv_grid.items():
            for _, gen_subtype in gen_type.items():
                generator['name'].append(gen_subtype['name'])
                generator['bus'].append(
                    '_'.join(['Bus', lv_grid_obj.station.__repr__('lv')]))
                generator['control'].append('PQ')
                generator['p_nom'].append(gen_subtype['capacity'])
                generator['type'].append("")

    # fill loads dictionary for DataFrame creation
    for lv_grid_obj, lv_grid in loads.items():
        for sector, val in lv_grid.items():
            load['name'].append('_'.join(['Load', sector, repr(lv_grid_obj)]))
            load['bus'].append(
                '_'.join(['Bus', lv_grid_obj.station.__repr__('lv')]))

    components['Generator'] = pd.concat(
        [components['Generator'],pd.DataFrame(generator).set_index('name')])
    components['Load'] = pd.concat(
        [components['Load'], pd.DataFrame(load).set_index('name')])

    return components


def _pypsa_load_timeseries(network, mode=None):
    """Timeseries in PyPSA compatible format for load instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    mode : str, optional
        Specifically retrieve load time series for MV or LV grid level or both.
        Either choose 'mv' or 'lv'.
        Defaults to None, which returns both timeseries for MV and LV in a
        single DataFrame.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """
    mv_load_timeseries_p = []
    mv_load_timeseries_q = []
    lv_load_timeseries_p = []
    lv_load_timeseries_q = []

    # add MV grid loads
    if mode is 'mv' or mode is None:
        for load in network.mv_grid.graph.nodes_by_attribute('load'):
            mv_load_timeseries_q.append(
                load.pypsa_timeseries('q').rename(repr(load)).to_frame())
            mv_load_timeseries_p.append(
                load.pypsa_timeseries('p').rename(repr(load)).to_frame())

    # add LV grid's loads
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for load in lv_grid.graph.nodes_by_attribute('load'):
                for sector in list(load.consumption.keys()):
                    lv_load_timeseries_q.append(
                        load.pypsa_timeseries('q').rename(
                            repr(load)).to_frame())
                    lv_load_timeseries_p.append(
                        load.pypsa_timeseries('p').rename(
                            repr(load)).to_frame())

    load_df_p = pd.concat(mv_load_timeseries_p + lv_load_timeseries_p, axis=1)
    load_df_q = pd.concat(mv_load_timeseries_q + lv_load_timeseries_q, axis=1)

    # TODO: maybe names of load object have to be changed to distinguish between different grids

    return load_df_p, load_df_q


def _pypsa_generator_timeseries(network, mode=None):
    """Timeseries in PyPSA compatible format for generator instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    mode : str, optional
        Specifically retrieve generator time series for MV or LV grid level or
        both. Either choose 'mv' or 'lv'.
        Defaults to None, which returns both timeseries for MV and LV in a
        single DataFrame.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    mv_gen_timeseries_q = []
    mv_gen_timeseries_p = []
    lv_gen_timeseries_q = []
    lv_gen_timeseries_p = []

    # MV generator timeseries
    if mode is 'mv' or mode is None:
        for gen in network.mv_grid.graph.nodes_by_attribute('generator'):
            mv_gen_timeseries_q.append(
                gen.pypsa_timeseries('q').rename(repr(gen)).to_frame())
            mv_gen_timeseries_p.append(
                gen.pypsa_timeseries('p').rename(repr(gen)).to_frame())

    # LV generator timeseries
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for gen in lv_grid.graph.nodes_by_attribute('generator'):
                lv_gen_timeseries_q.append(
                    gen.pypsa_timeseries('q').rename(repr(gen)).to_frame())
                lv_gen_timeseries_p.append(
                    gen.pypsa_timeseries('p').rename(repr(gen)).to_frame())

    gen_df_p = pd.concat(mv_gen_timeseries_p + lv_gen_timeseries_p, axis=1)
    gen_df_q = pd.concat(mv_gen_timeseries_q + lv_gen_timeseries_q, axis=1)

    # TODO: maybe names of load object have to be changed to distinguish between different grids

    return gen_df_p, gen_df_q

def _pypsa_bus_timeseries(network, buses, mode=None):
    """Timeseries in PyPSA compatible format for generator instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    buses : list
        Buses names
    mode : str, optional
        Specifically retrieve generator time series for MV or LV grid level or
        both. Either choose 'mv' or 'lv'.
        Defaults to None, which returns both timeseries for MV and LV in a
        single DataFrame.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    v_set_dict = {bus: [1, 1] for bus in buses}

    v_set_df = pd.DataFrame(v_set_dict,
                            index=network.scenario.timeseries.timeindex)

    return v_set_df


def _pypsa_timeseries_aggregated_at_lv_station(network):
    """

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container

    Returns
    -------
    tuple of :pandas:`pandas.DataFrame<dataframe>`
        Altered time series DataFrames. Tuple of size for containing DataFrames
        that represent

            1. 'p_set' of aggregated Generation at each LV station
            2. 'q_set' of aggregated Generation at each LV station
            3. 'p_set' of aggregated Load at each LV station
            4. 'q_set' of aggregated Load at each LV station
    """

    generation_p = []
    generation_q = []
    load_p = []
    load_q = []

    for lv_grid in network.mv_grid.lv_grids:
        # Determine aggregated generation at LV stations
        generation = {}
        for gen in lv_grid.graph.nodes_by_attribute('generator'):
            # for type in gen.type:
            #     for subtype in gen.subtype:
            gen_name = '_'.join([gen.type,
                                 gen.subtype,
                                 'aggregated',
                                 'LV_grid',
                                 str(lv_grid.id)])

            generation.setdefault(gen.type, {})
            generation[gen.type].setdefault(gen.subtype, {})
            generation[gen.type][gen.subtype].setdefault('timeseries_p', [])
            generation[gen.type][gen.subtype].setdefault('timeseries_q', [])
            generation[gen.type][gen.subtype]['timeseries_p'].append(
                gen.pypsa_timeseries('p').rename(gen_name).to_frame())
            generation[gen.type][gen.subtype]['timeseries_q'].append(
                gen.pypsa_timeseries('q').rename(gen_name).to_frame())

        for k_type, v_type in generation.items():
            for k_type, v_subtype in v_type.items():
                col_name = v_subtype['timeseries_p'][0].columns[0]
                generation_p.append(
                    pd.concat(v_subtype['timeseries_p'],
                              axis=1).sum(axis=1).rename(col_name).to_frame())
                generation_q.append(
                    pd.concat(v_subtype['timeseries_q'], axis=1).sum(
                        axis=1).rename(col_name).to_frame())

        # Determine aggregated load at LV stations
        load = {}
        for lo in lv_grid.graph.nodes_by_attribute('load'):
            for sector, val in lo.consumption.items():
                load.setdefault(sector, {})
                load[sector].setdefault('timeseries_p', [])
                load[sector].setdefault('timeseries_q', [])

                load[sector]['timeseries_p'].append(
                    lo.pypsa_timeseries('p').rename(repr(lo)).to_frame())
                load[sector]['timeseries_q'].append(
                    lo.pypsa_timeseries('q').rename(repr(lo)).to_frame())
                # TODO: now, there are single loads from eDisGo topology. Summarize these by sector and LV grid

        for sector, val in load.items():
            load_p.append(
                pd.concat(val['timeseries_p'], axis=1).sum(axis=1).rename(
                    '_'.join(['Load', sector, repr(lv_grid)])).to_frame())
            load_q.append(
                pd.concat(val['timeseries_q'], axis=1).sum(axis=1).rename(
                    '_'.join(['Load', sector, repr(lv_grid)])).to_frame())

    return pd.concat(generation_p, axis=1), \
           pd.concat(generation_q, axis=1), \
           pd.concat(load_p, axis=1), \
           pd.concat(load_q, axis=1)


def _check_topology(mv_components):
    buses = mv_components['Bus'].index.tolist()
    line_buses = mv_components['Line']['bus0'].tolist() + \
                 mv_components['Line']['bus1'].tolist()
    load_buses = mv_components['Load']['bus'].tolist()
    generator_buses = mv_components['Generator']['bus'].tolist()
    transformer_buses = mv_components['Transformer']['bus0'].tolist() + \
                        mv_components['Transformer']['bus1'].tolist()

    buses_to_check = line_buses + load_buses + generator_buses + \
                     transformer_buses

    missing_buses = []

    missing_buses.extend([_ for _ in buses_to_check if _ not in buses])

    if missing_buses:
        raise ValueError("Buses {buses} are not defined.".format(
            buses=missing_buses))


def _check_integrity_of_pypsa(pypsa_network):
    """"""

    # check for sub-networks
    pypsa_network.determine_network_topology()
    subgraphs = list(connected_component_subgraphs(pypsa_network.graph()))

    if len(subgraphs) > 1 or len(pypsa_network.sub_networks) > 1:
        raise ValueError("The graph has isolated nodes or edges")

    # check consistency of topology and time series data
    generators_ts_p_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['p_set'].columns.tolist())]
    generators_ts_q_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['q_set'].columns.tolist())]
    loads_ts_p_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t['p_set'].columns.tolist())]
    loads_ts_q_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t['q_set'].columns.tolist())]
    bus_v_set_missing = pypsa_network.buses.loc[
        ~pypsa_network.buses.index.isin(
            pypsa_network.buses_t['v_mag_pu_set'].columns.tolist())]

    # Comparison of generators excludes slack generators (have no time series)
    if (not generators_ts_p_missing.empty and not all(
                generators_ts_p_missing['control'] == 'Slack')):
        raise ValueError("Following generators have no `p_set` time series "
                         "{generators}".format(
            generators=generators_ts_p_missing))

    if (not generators_ts_q_missing.empty and not all(
                generators_ts_q_missing['control'] == 'Slack')):
        raise ValueError("Following generators have no `q_set` time series "
                         "{generators}".format(
            generators=generators_ts_q_missing))
    
    if not loads_ts_p_missing.empty:
        raise ValueError("Following loads have no `p_set` time series "
                         "{loads}".format(
            loads=loads_ts_p_missing))

    if not loads_ts_q_missing.empty:
        raise ValueError("Following loads have no `q_set` time series "
                         "{loads}".format(
            loads=loads_ts_q_missing))

    if not bus_v_set_missing.empty:
        raise ValueError("Following loads have no `v_mag_pu_set` time series "
                         "{buses}".format(
            buses=bus_v_set_missing))


def process_pfa_results(network, pypsa):
    """
    Assing values from PyPSA to
    :meth:`results <edisgo.grid.network.Network.results>`

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    pypsa :
        `Network container <https://www.pypsa.org/doc/components.html#network>`_
        of PyPSA

    Returns
    -------

    See Also
    --------
    :class:`~.grid.network.Results`
        Understand how results of power flow analysis are structured in eDisGo.

    """

    # line results
    q0 = abs(pypsa.lines_t['q0'])
    q1 = abs(pypsa.lines_t['q1'])
    p0 = abs(pypsa.lines_t['p0'])
    p1 = abs(pypsa.lines_t['p1'])
    network.results.pfa_p = p0.where(p0 > p1, p1)
    network.results.pfa_q = q0.where(q0 > q1, q1)

    # process results at nodes
    generators_names = [repr(g) for g in
                        network.mv_grid.graph.nodes_by_attribute('generator')]
    generators_mapping = {v: k for k, v in
                          pypsa.generators.loc[generators_names][
                              'bus'].to_dict().items()}
    branch_t_names = [repr(bt) for bt in
                      network.mv_grid.graph.nodes_by_attribute('branch_tee')]
    branch_t_mapping = {'_'.join(['Bus', v]): v for v in branch_t_names}
    mv_station_names = [repr(m) for m in
                        network.mv_grid.graph.nodes_by_attribute('mv_station')]
    mv_station_mapping_sec = {'_'.join(['Bus', v]): v for v in mv_station_names}
    lv_station_names = [repr(l) for l in
                        network.mv_grid.graph.nodes_by_attribute('lv_station')]
    lv_station_mapping_pri = {
        '_'.join(['Bus', l.__repr__('mv')]): repr(l)
        for l in network.mv_grid.graph.nodes_by_attribute('lv_station')}
    lv_station_mapping_sec = {
        '_'.join(['Bus', l.__repr__('lv')]): repr(l)
        for l in network.mv_grid.graph.nodes_by_attribute('lv_station')}
    loads_names = [repr(lo) for lo in
                   network.mv_grid.graph.nodes_by_attribute('load')]
    loads_mapping = {v: k for k, v in
                     pypsa.loads.loc[loads_names][
                         'bus'].to_dict().items()}

    names_mapping = {
        **generators_mapping,
        **branch_t_mapping,
        **mv_station_mapping_sec,
        **lv_station_mapping_pri,
        **lv_station_mapping_sec,
        **loads_mapping
    }

    # write voltage levels obtained from power flow to results object
    pfa_v_mag_pu = pypsa.buses_t['v_mag_pu'].rename(columns=names_mapping)
    network.results.pfa_v_mag_pu = pd.concat(
        {'mv': pfa_v_mag_pu[list(generators_mapping.values()) +
                            list(branch_t_mapping.values()) +
                            list(mv_station_mapping_sec.values()) +
                            list(lv_station_mapping_pri.values()) +
                            list(loads_mapping.values())],
         'lv': pfa_v_mag_pu[list(lv_station_mapping_sec.values())]}, axis=1)