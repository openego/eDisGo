"""
This module provides tools to convert graph based representation of the network
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA network
container.
"""

import numpy as np
import pandas as pd
from math import sqrt
from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe
from networkx import connected_components
import collections


def to_pypsa(grid_object, mode, timesteps):
    """
    Translate graph based network representation to PyPSA Network

    For details from a user perspective see API documentation of
    :meth:`~.network.network.EDisGo.analyze` of the API class
    :class:`~.network.network.EDisGo`.

    Translating eDisGo's network topology to PyPSA representation is structured
    into translating the topology and adding time series for components of the
    network. In both cases translation of MV network only (`mode='mv'`), LV network only
    (`mode='lv'`), MV and LV (`mode=None`) share some code. The
    code is organized as follows:

    * Medium-voltage only (`mode='mv'`): All medium-voltage network components are
      exported by :func:`mv_to_pypsa` including the LV station. LV network load
      and generation is considered using :func:`add_aggregated_lv_components`.
      Time series are collected by `_pypsa_load_timeseries` (as example
      for loads, generators and buses) specifying `mode='mv'`). Timeseries
      for aggregated load/generation at substations are determined individually.
    * Low-voltage only (`mode='lv'`): LV network topology including the MV-LV
      transformer is exported. The slack is defind at primary side of the MV-LV
      transformer.
    * Both level MV+LV (`mode=None`): The entire network topology is translated to
      PyPSA in order to perform a complete power flow analysis in both levels
      together. First, both network levels are translated seperately using
      :func:`mv_to_pypsa` and :func:`lv_to_pypsa`. Those are merge by
      :func:`combine_mv_and_lv`. Time series are obtained at once for both network
      levels.

    This PyPSA interface is aware of translation errors and performs so checks
    on integrity of data converted to PyPSA network representation

    * Sub-graphs/ Sub-networks: It is ensured the network has no islanded parts
    * Completeness of time series: It is ensured each component has a time
      series
    * Buses available: Each component (load, generator, line, transformer) is
      connected to a bus. The PyPSA representation is check for completeness of
      buses.
    * Duplicate labels in components DataFrames and components' time series
      DataFrames

    Parameters
    ----------
    network : :class:`~.network.network.Network`
        eDisGo network container
    mode : str
        Determines network levels that are translated to
        `PyPSA network representation
        <https://www.pypsa.org/doc/components.html#network>`_. Specify

        * None to export MV and LV network levels. None is the default.
        * ('mv' to export MV network level only. This includes cumulative load and
          generation from underlying LV network aggregated at respective LV
          station. This option is implemented, though the rest of edisgo does
          not handle it yet.)
        * ('lv' to export LV network level only. This option is not yet
           implemented)
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or \
        :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps to export to pypsa representation
        and use in power flow analysis.

    Returns
    -------
        :pypsa:`pypsa.Network<network>`
            The `PyPSA network
            <https://www.pypsa.org/doc/components.html#network>`_ container.

    """

    # check if timesteps is array-like, otherwise convert to list (necessary
    # to obtain a dataframe when using .loc in time series functions)
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.set_snapshots(timesteps)

    # get topology and time series data
    if mode is None:
        edisgo_obj = grid_object
        buses_df = edisgo_obj.network.buses_df.loc[:, ['v_nom']]
        buses = edisgo_obj.network.buses_df.index

        # loads generators buses storages lines transformers
        #ToDo change getting generators once slack is separate dataframe
        components = {
            'Load': edisgo_obj.network.loads_df.loc[:, ['bus', 'peak_load']].rename(
                columns={'peak_load':'p_set'}
            ),
            'Generator': edisgo_obj.network._generators_df.loc[:, ['bus', 'control', 'p_nom']],
            'StorageUnit': edisgo_obj.network.storages_df.loc[:, ['bus', 'control']],
            'Line': edisgo_obj.network.lines_df.loc[:, ['bus0', 'bus1', 'x', 'r', 's_nom']],
            'Transformer': edisgo_obj.network.transformers_df.loc[
                           :, ['bus0', 'bus1', 'x_pu', 'r_pu', 'type', 's_nom']].rename(
                columns={'r_pu': 'r', 'x_pu': 'x'})
        }


    elif 'mv' in mode:
        grid = grid_object
        edisgo_obj = grid.edisgo_obj
        lv_components_to_aggregate = {'Load': 'loads_df',
                                      'Generator': 'generators_df',
                                      'StorageUnit': 'storages_df'}
        lv_components = {key: {} for key in lv_components_to_aggregate}
        buses_df = grid.buses_df.loc[:, ['v_nom']]
        buses = grid.buses_df.index
    # mv network with lv loads and generators connected to mv side of station
        if mode is 'mv':
            # get mv_components
            mv_components = {
                'Load': grid.loads_df.loc[:, ['bus', 'peak_load']].rename(
                columns={'peak_load': 'p_set'}
            ),
                'Generator': grid.generators_df.loc[:, ['bus', 'control', 'p_nom']].append(
                    grid.edisgo_obj.network._generators_df.loc['Generator_slack',
                                                    ['bus', 'control', 'p_nom']]), # Todo: change when slack is dataframe
                'StorageUnit': grid.storages_df.loc[:, ['bus', 'control']],
                'Line': grid.lines_df.loc[:, ['bus0', 'bus1', 'x', 'r', 's_nom']],
                'Transformer': grid.transformers_df[grid.transformers_df.bus1.isin(
                    grid.buses_df.index
                )].loc[
                               :, ['bus0', 'bus1', 'x_pu', 'r_pu', 'type',
                                   's_nom']].rename(
                    columns={'r_pu': 'r', 'x_pu': 'x'})
            }

            # get lv_components
            for lv_grid in grid.lv_grids:
                # get primary side of station to append loads and generators to
                station_bus = grid.buses_df.loc[
                    lv_grid.transformers_df.bus0.unique()]
                # handle one gate component
                for comp, df in lv_components_to_aggregate.items():
                    comps = getattr(lv_grid, df).copy()
                    comps.bus = station_bus.index.values[0]
                    append_lv_components(comp, comps, lv_components, lv_grid)

                # Todo: accumulate loads?

        # mv network with accumulated loads and generators at lv side of station
        elif mode is 'mvlv':

            mv_components = {
                'Load': grid.loads_df.loc[:, ['bus', 'peak_load']].rename(
                columns={'peak_load':'p_set'}),
                'Generator': grid.generators_df.loc[:, ['bus', 'control', 'p_nom']].append(
                    grid.edisgo_obj.network._generators_df.loc['Generator_slack',
                                                    ['bus', 'control', 'p_nom']]), # Todo: change when slack is dataframe
                'StorageUnit': grid.storages_df.loc[:, ['bus', 'control']],
                'Line': grid.lines_df.loc[:, ['bus0', 'bus1', 'x', 'r', 's_nom']],
                'Transformer': edisgo_obj.network.transformers_df.loc[
                               :, ['bus0', 'bus1', 'x_pu', 'r_pu', 'type',
                                   's_nom']].rename(
                    columns={'r_pu': 'r', 'x_pu': 'x'})
            }


            # get lv_components
            lv_components_to_aggregate = {'Load': 'loads_df',
                                          'Generator': 'generators_df',
                                          'StorageUnit': 'storages_df'}
            lv_components = {key: {} for key in lv_components_to_aggregate}
            for lv_grid in grid.lv_grids:
                # get primary side of station to append loads and generators to
                station_bus = lv_grid.station
                buses_df = buses_df.append(station_bus.loc[:, ['v_nom']])
                buses = buses.append(station_bus.index)
                # handle one gate component
                for comp, df in lv_components_to_aggregate.items():
                    comps = getattr(lv_grid, df).copy()
                    comps.bus = station_bus.index.values[0]
                    append_lv_components(comp, comps, lv_components, lv_grid)
                # Todo: accumulate loads?

        else:
            raise ValueError("Provide proper mode for mv network export.")

        for key in lv_components:
            lv_components[key] = pd.concat(lv_components[key].values())
        # merge components
        components = collections.defaultdict(pd.DataFrame)
        for comps in (mv_components, lv_components):
            for key, value in comps.items():
                components[key] = components[key].append(value)

        # import network topology to PyPSA network
        # buses are created first to avoid warnings
        pypsa_network.import_components_from_dataframe(
            buses_df, 'Bus')

    elif mode is 'lv':
        grid = grid_object
        edisgo_obj = grid.edisgo_obj
        buses_df = grid.buses_df.loc[:, ['v_nom']]
        buses = grid.buses_df.index

        slack = pd.DataFrame({'name': ['Generator_slack'],
                              'bus': [grid.station.index.values[0]],
                              'control': ['Slack']}).set_index('name')
        components = {
            'Load': grid.loads_df.loc[:, ['bus', 'peak_load']].rename(
                columns={'peak_load': 'p_set'}),
            'Generator': grid.generators_df.loc[:, ['bus', 'control', 'p_nom']].append(
                slack),
            'StorageUnit': grid.storages_df.loc[:, ['bus', 'control']],
            'Line': grid.lines_df.loc[:, ['bus0', 'bus1', 'x', 'r', 's_nom']]
        }
    else:
        raise ValueError("Provide proper mode or leave it empty to export "
                         "entire network topology.")

    # import network topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(
        buses_df, 'Bus')

    for k, comps in components.items():
        pypsa_network.import_components_from_dataframe(comps, k)

    if len(buses) > 0:
        import_series_from_dataframe(
            pypsa_network,
            _buses_voltage_set_point(edisgo_obj, buses, timesteps),
            'Bus', 'v_mag_pu_set')

    # import time series to PyPSA network
    if len(components['Generator'].index) > 0:
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.generators_active_power.loc[
                timesteps, components['Generator'].index],
            'Generator', 'p_set')
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.generators_reactive_power.loc[
                timesteps, components['Generator'].index],
            'Generator', 'q_set')

    if len(components['Load'].index) > 0:
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.loads_active_power.loc[
                timesteps, components['Load'].index],
            'Load', 'p_set')
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.loads_reactive_power.loc[
                timesteps, components['Load'].index],
            'Load', 'q_set')

    if len(components['StorageUnit'].index) > 0:
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.storages_active_power.loc[
                timesteps, components['StorageUnit'].index],
            'StorageUnit', 'p_set')
        import_series_from_dataframe(
            pypsa_network,
            edisgo_obj.timeseries.storages_reactive_power.loc[
                timesteps, components['StorageUnit'].index],
            'StorageUnit', 'q_set')

    _check_integrity_of_pypsa(pypsa_network)

    return pypsa_network


def append_lv_components(comp, comps, lv_components, lv_grid):
    if comp is 'Load':
        lv_components[comp][str(lv_grid.id)] = \
            comps.loc[:, ['bus', 'peak_load']].rename(
                columns={'peak_load': 'p_set'})
    elif comp is 'Generator':
        lv_components[comp][str(lv_grid.id)] = \
            comps.loc[:, ['bus', 'control', 'p_nom']]
    elif comp is 'StorageUnit':
        lv_components[comp][str(lv_grid.id)] = comps.loc[:,
                                               ['bus', 'control']]
    else:
        raise ValueError('Component Type not defined.')


def _buses_voltage_set_point(edisgo_obj, buses, timesteps):
    """
    Time series in PyPSA compatible format for bus instances

    Set all buses except for the slack bus to voltage of 1 p.u. (it is assumed
    this setting is entirely ignored during solving the power flow problem).
    The slack bus voltage is set based on a given HV/MV transformer offset and
    a control deviation, both defined in the config files. The control
    deviation is added to the offset in the reverse power flow case and
    subtracted from the offset in the heavy load flow case.

    Parameters
    ----------
    network : Network
        The eDisGo network topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
    buses : list
        Buses names

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    # get slack bus label
    #ToDo change once slack is property in network
    slack_bus = edisgo_obj.network._generators_df.at['Generator_slack', 'bus'] # Todo: change to at['Slack', 'control']

    # set all buses to nominal voltage
    v_nom = pd.DataFrame(1, columns=buses, index=timesteps)

    # set slack bus to operational voltage (includes offset and control
    # deviation)
    control_deviation = edisgo_obj.config[
        'grid_expansion_allowed_voltage_deviations'][
        'hv_mv_trafo_control_deviation']
    if control_deviation != 0:
        control_deviation_ts = \
            edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
                lambda _: control_deviation if _ == 'feedin_case'
                else -control_deviation).loc[timesteps]
    else:
        control_deviation_ts = pd.Series(0, index=timesteps)

    slack_voltage_pu = \
        control_deviation_ts + 1 + edisgo_obj.config[
            'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']

    v_nom.loc[timesteps, slack_bus] = slack_voltage_pu

    return v_nom


def add_aggregated_lv_components(network, components):
    """
    Aggregates LV load and generation at LV stations

    Use this function if you aim for MV calculation only. The according
    DataFrames of `components` are extended by load and generators representing
    these aggregated respecting the technology type.

    Parameters
    ----------
    network : Network
        The eDisGo network topology model overall container
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
        for gen in lv_grid.generators:
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
        [components['Generator'], pd.DataFrame(generator).set_index('name')])
    components['Load'] = pd.concat(
        [components['Load'], pd.DataFrame(load).set_index('name')])

    return components


def _pypsa_bus_timeseries(network, buses, timesteps):
    """
    Time series in PyPSA compatible format for bus instances

    Set all buses except for the slack bus to voltage of 1 pu (it is assumed
    this setting is entirely ignored during solving the power flow problem).
    This slack bus is set to an operational voltage which is typically greater
    than nominal voltage plus a control deviation.
    The control deviation is always added positively to the operational voltage.
    For example, the operational voltage (offset) is set to 1.025 pu plus the
    control deviation of 0.015 pu. This adds up to a set voltage of the slack
    bus of 1.04 pu.

    .. warning::

        Voltage settings for the slack bus defined by this function assume the
        feedin case (reverse power flow case) as the worst-case for the power
        system. Thus, the set point for the slack is always greater 1.


    Parameters
    ----------
    network : Network
        The eDisGo network topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
    buses : list
        Buses names

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Time series table in PyPSA format
    """

    # get slack bus label
    slack_bus = '_'.join(
        ['Bus', network.mv_grid.station.__repr__(side='mv')])

    # set all buses (except slack bus) to nominal voltage
    v_set_dict = {bus: 1 for bus in buses if bus != slack_bus}

    # Set slack bus to operational voltage (includes offset and control
    # deviation
    control_deviation = network.config[
        'grid_expansion_allowed_voltage_deviations'][
        'hv_mv_trafo_control_deviation']
    if control_deviation != 0:
        control_deviation_ts = \
            network.timeseries.timesteps_load_feedin_case.case.apply(
                lambda _: control_deviation if _ == 'feedin_case'
                else -control_deviation)
    else:
        control_deviation_ts = 0

    slack_voltage_pu = control_deviation_ts + 1 + \
                       network.config[
                           'grid_expansion_allowed_voltage_deviations'][
                           'hv_mv_trafo_offset']

    v_set_dict.update({slack_bus: slack_voltage_pu})

    # Convert to PyPSA compatible dataframe
    v_set_df = pd.DataFrame(v_set_dict, index=timesteps)

    return v_set_df


def _pypsa_generator_timeseries_aggregated_at_lv_station(network, timesteps):
    """
    Aggregates generator time series per generator subtype and LV network.

    Parameters
    ----------
    network : Network
        The eDisGo network topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.

    Returns
    -------
    tuple of :pandas:`pandas.DataFrame<dataframe>`
        Tuple of size two containing DataFrames that represent

            1. 'p_set' of aggregated Generation per subtype at each LV station
            2. 'q_set' of aggregated Generation per subtype at each LV station

    """

    generation_p = []
    generation_q = []

    for lv_grid in network.mv_grid.lv_grids:
        # Determine aggregated generation at LV stations
        generation = {}
        for gen in lv_grid.generators:
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
                gen.pypsa_timeseries('p').rename(gen_name).to_frame().loc[
                    timesteps])
            generation[gen.type][gen.subtype]['timeseries_q'].append(
                gen.pypsa_timeseries('q').rename(gen_name).to_frame().loc[
                    timesteps])

        for k_type, v_type in generation.items():
            for k_type, v_subtype in v_type.items():
                col_name = v_subtype['timeseries_p'][0].columns[0]
                generation_p.append(
                    pd.concat(v_subtype['timeseries_p'],
                              axis=1).sum(axis=1).rename(col_name).to_frame())
                generation_q.append(
                    pd.concat(v_subtype['timeseries_q'], axis=1).sum(
                        axis=1).rename(col_name).to_frame())

    return generation_p, generation_q


def _pypsa_load_timeseries_aggregated_at_lv_station(network, timesteps):
    """
    Aggregates load time series per sector and LV network.

    Parameters
    ----------
    network : Network
        The eDisGo network topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.

    Returns
    -------
    tuple of :pandas:`pandas.DataFrame<dataframe>`
        Tuple of size two containing DataFrames that represent

            1. 'p_set' of aggregated Load per sector at each LV station
            2. 'q_set' of aggregated Load per sector at each LV station

    """
    # ToDo: Load.pypsa_timeseries is not differentiated by sector so this
    # function will not work (either change here and in
    # add_aggregated_lv_components or in Load class)

    load_p = []
    load_q = []

    for lv_grid in network.mv_grid.lv_grids:
        # Determine aggregated load at LV stations
        load = {}
        for lo in lv_grid.graph.nodes_by_attribute('load'):
            for sector, val in lo.consumption.items():
                load.setdefault(sector, {})
                load[sector].setdefault('timeseries_p', [])
                load[sector].setdefault('timeseries_q', [])

                load[sector]['timeseries_p'].append(
                    lo.pypsa_timeseries('p').rename(repr(lo)).to_frame().loc[
                        timesteps])
                load[sector]['timeseries_q'].append(
                    lo.pypsa_timeseries('q').rename(repr(lo)).to_frame().loc[
                        timesteps])

        for sector, val in load.items():
            load_p.append(
                pd.concat(val['timeseries_p'], axis=1).sum(axis=1).rename(
                    '_'.join(['Load', sector, repr(lv_grid)])).to_frame())
            load_q.append(
                pd.concat(val['timeseries_q'], axis=1).sum(axis=1).rename(
                    '_'.join(['Load', sector, repr(lv_grid)])).to_frame())

    return load_p, load_q


def _check_topology(components):
    buses = components['Bus'].index.tolist()
    line_buses = components['Line']['bus0'].tolist() + \
                 components['Line']['bus1'].tolist()
    load_buses = components['Load']['bus'].tolist()
    generator_buses = components['Generator']['bus'].tolist()
    transformer_buses = components['Transformer']['bus0'].tolist() + \
                        components['Transformer']['bus1'].tolist()

    buses_to_check = line_buses + load_buses + generator_buses + \
                     transformer_buses

    missing_buses = []

    missing_buses.extend([_ for _ in buses_to_check if _ not in buses])

    if missing_buses:
        raise ValueError("Buses {buses} are not defined.".format(
            buses=missing_buses))

    # check if there are duplicate components and print them
    for k, comps in components.items():
        if len(list(comps.index.values)) != len(set(comps.index.values)):
            raise ValueError("There are duplicates in the {comp} list: {dupl}"
                             .format(comp=k,
                                     dupl=[item for item, count in
                                           collections.Counter(
                                               comps.index.values).items()
                                           if count > 1])
                             )


def _check_integrity_of_pypsa(pypsa_network):
    """
    #ToDo docstring
    :param pypsa_network:
    :return:
    """

    # check for sub-networks
    subgraphs = list(pypsa_network.graph().subgraph(c) for c in
                     connected_components(pypsa_network.graph()))
    pypsa_network.determine_network_topology()

    if len(subgraphs) > 1 or len(pypsa_network.sub_networks) > 1:
        raise ValueError("The pypsa graph has isolated nodes or edges.")

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

    # check for duplicate labels (of components)
    duplicated_labels = []
    if any(pypsa_network.buses.index.duplicated()):
        duplicated_labels.append(pypsa_network.buses.index[
                                     pypsa_network.buses.index.duplicated()])
    if any(pypsa_network.generators.index.duplicated()):
        duplicated_labels.append(pypsa_network.generators.index[
                                     pypsa_network.generators.index.duplicated()])
    if any(pypsa_network.loads.index.duplicated()):
        duplicated_labels.append(pypsa_network.loads.index[
                                     pypsa_network.loads.index.duplicated()])
    if any(pypsa_network.transformers.index.duplicated()):
        duplicated_labels.append(pypsa_network.transformers.index[
                                     pypsa_network.transformers.index.duplicated()])
    if any(pypsa_network.lines.index.duplicated()):
        duplicated_labels.append(pypsa_network.lines.index[
                                     pypsa_network.lines.index.duplicated()])
    if duplicated_labels:
        raise ValueError("{labels} have duplicate entry in "
                         "one of the components dataframes".format(
            labels=duplicated_labels))

    # duplicate p_sets and q_set
    duplicate_p_sets = []
    duplicate_q_sets = []
    if any(pypsa_network.loads_t['p_set'].columns.duplicated()):
        duplicate_p_sets.append(pypsa_network.loads_t['p_set'].columns[
                                    pypsa_network.loads_t[
                                        'p_set'].columns.duplicated()])
    if any(pypsa_network.loads_t['q_set'].columns.duplicated()):
        duplicate_q_sets.append(pypsa_network.loads_t['q_set'].columns[
                                    pypsa_network.loads_t[
                                        'q_set'].columns.duplicated()])

    if any(pypsa_network.generators_t['p_set'].columns.duplicated()):
        duplicate_p_sets.append(pypsa_network.generators_t['p_set'].columns[
                                    pypsa_network.generators_t[
                                        'p_set'].columns.duplicated()])
    if any(pypsa_network.generators_t['q_set'].columns.duplicated()):
        duplicate_q_sets.append(pypsa_network.generators_t['q_set'].columns[
                                    pypsa_network.generators_t[
                                        'q_set'].columns.duplicated()])

    if duplicate_p_sets:
        raise ValueError("{labels} have duplicate entry in "
                         "generators_t['p_set']"
                         " or loads_t['p_set']".format(
            labels=duplicate_p_sets))
    if duplicate_q_sets:
        raise ValueError("{labels} have duplicate entry in "
                         "generators_t['q_set']"
                         " or loads_t['q_set']".format(
            labels=duplicate_q_sets))

    # find duplicate v_mag_set entries
    duplicate_v_mag_set = []
    if any(pypsa_network.buses_t['v_mag_pu_set'].columns.duplicated()):
        duplicate_v_mag_set.append(
            pypsa_network.buses_t['v_mag_pu_set'].columns[
                pypsa_network.buses_t[
                    'v_mag_pu_set'].columns.duplicated()])

    if duplicate_v_mag_set:
        raise ValueError("{labels} have duplicate entry in buses_t".format(
            labels=duplicate_v_mag_set))


def process_pfa_results(edisgo, pypsa, timesteps):
    """
    Assing values from PyPSA to
    :meth:`results <edisgo.network.network.Network.results>`

    Parameters
    ----------
    edisgo : Network
        The eDisGo network topology model overall container
    pypsa : :pypsa:`pypsa.Network<network>`
        The PyPSA `Network container
        <https://www.pypsa.org/doc/components.html#network>`_
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Time steps for which latest power flow analysis was conducted for and
        for which to retrieve pypsa results.

    Notes
    -----
    P and Q (and respectively later S) are returned from the line ending/
    transformer side with highest apparent power S, exemplary written as

    .. math::
        S_{max} = max(\sqrt{P0^2 + Q0^2}, \sqrt{P1^2 + Q1^2})
        P = P0P1(S_{max})
        Q = Q0Q1(S_{max})

    See Also
    --------
    :class:`~.network.network.Results`
        Understand how results of power flow analysis are structured in eDisGo.

    """
    # get the absolute losses in the system
    # subtracting total generation (including slack) from total load
    grid_losses = {'p': (pypsa.generators_t['p'].sum(axis=1) -
                               pypsa.loads_t['p'].sum(axis=1)),
                   'q': (pypsa.generators_t['q'].sum(axis=1) -
                               pypsa.loads_t['q'].sum(axis=1))}

    edisgo.results.grid_losses = pd.DataFrame(grid_losses).loc[timesteps, :]

    # get slack results
    grid_exchanges = {'p': (pypsa.generators_t['p']['Generator_slack']),
                      'q': (pypsa.generators_t['q']['Generator_slack'])}

    edisgo.results.hv_mv_exchanges = pd.DataFrame(grid_exchanges).loc[
                                      timesteps, :]

    # get p and q of lines, LV transformers and MV Station (slack generator)
    # in absolute values
    q0 = pd.concat(
        [np.abs(pypsa.lines_t['q0']),
         np.abs(pypsa.transformers_t['q0']),
         np.abs(pypsa.generators_t['q']['Generator_slack'].rename(
             repr(edisgo.network.mv_grid.station)))], axis=1).loc[timesteps, :]
    q1 = pd.concat(
        [np.abs(pypsa.lines_t['q1']),
         np.abs(pypsa.transformers_t['q1']),
         np.abs(pypsa.generators_t['q']['Generator_slack'].rename(
             repr(edisgo.network.mv_grid.station)))], axis=1).loc[timesteps, :]
    p0 = pd.concat(
        [np.abs(pypsa.lines_t['p0']),
         np.abs(pypsa.transformers_t['p0']),
         np.abs(pypsa.generators_t['p']['Generator_slack'].rename(
             repr(edisgo.network.mv_grid.station)))], axis=1).loc[timesteps, :]
    p1 = pd.concat(
        [np.abs(pypsa.lines_t['p1']),
         np.abs(pypsa.transformers_t['p1']),
         np.abs(pypsa.generators_t['p']['Generator_slack'].rename(
             repr(edisgo.network.mv_grid.station)))], axis=1).loc[timesteps, :]

    # determine apparent power and line endings/transformers' side
    s0 = np.hypot(p0, q0)
    s1 = np.hypot(p1, q1)

    # choose p and q from line ending with max(s0,s1)
    edisgo.results.pfa_p = p0.where(s0 > s1, p1)
    edisgo.results.pfa_q = q0.where(s0 > s1, q1)

    # Todo: delete?
    lines_bus0 = pypsa.lines['bus0'].to_dict()
    bus0_v_mag_pu = pypsa.buses_t['v_mag_pu'].T.loc[
                    list(lines_bus0.values()), :].copy()
    bus0_v_mag_pu.index = list(lines_bus0.keys())

    lines_bus1 = pypsa.lines['bus1'].to_dict()
    bus1_v_mag_pu = pypsa.buses_t['v_mag_pu'].T.loc[
                    list(lines_bus1.values()), :].copy()
    bus1_v_mag_pu.index = list(lines_bus1.keys())

    # Get line current
    edisgo.results._i_res = np.hypot(
        pypsa.lines_t['p0'], pypsa.lines_t['q0']).truediv(
        pypsa.lines['v_nom'] * bus0_v_mag_pu.T,
        axis='columns') / sqrt(3) * 1e3

    #Todo: overthink mapping, should results not be assigned to original nodes?

    # process results at nodes
    generators_names = edisgo.network.mv_grid.generators_df.index.values
    generators_mapping = {v: k for k, v in
                          pypsa.generators.loc[generators_names][
                              'bus'].to_dict().items()}
    storages_names = edisgo.network.mv_grid.storages_df.index.values
    storages_mapping = {v: k for k, v in
                        pypsa.storage_units.loc[storages_names][
                            'bus'].to_dict().items()}
    loads_names = edisgo.network.mv_grid.loads_df.index.values
    loads_mapping = {v: k for k, v in
                        pypsa.loads.loc[loads_names][
                            'bus'].to_dict().items()}
    buses_names = edisgo.network.mv_grid.buses_df.index.values
    buses_mapping = {k: k for k in buses_names}

    lv_generators_names = []
    lv_storages_names = []
    lv_loads_names = []
    lv_buses_names = []
    for lv_grid in edisgo.network.mv_grid.lv_grids:
        lv_generators_names.extend(list(lv_grid.generators_df.index.values))
        lv_storages_names.extend(list(lv_grid.storages_df.index.values))
        lv_loads_names.extend(list(lv_grid.loads_df.index.values))
        lv_buses_names.extend(list(lv_grid.buses_df.index.values))

    lv_generators_mapping = {v: k for k, v in
                             pypsa.generators.loc[lv_generators_names][
                                 'bus'].to_dict().items()}
    lv_storages_mapping = {v: k for k, v in
                           pypsa.storage_units.loc[lv_storages_names][
                               'bus'].to_dict().items()}
    lv_loads_mapping = {v: k for k, v in pypsa.loads.loc[lv_loads_names][
        'bus'].to_dict().items()}
    lv_buses_mapping = {k: k for k in buses_names}

    names_mapping = {
        **generators_mapping,
        **storages_mapping,
        **loads_mapping,
        **buses_mapping,
        **lv_generators_mapping,
        **lv_storages_mapping,
        **lv_loads_mapping,
        **lv_buses_mapping
    }

    # write voltage levels obtained from power flow to results object
    pfa_v_mag_pu_mv = (pypsa.buses_t['v_mag_pu'][
        list(generators_mapping) +
        list(storages_mapping) +
        list(loads_mapping) +
        list(buses_mapping)]).rename(columns=names_mapping)
    pfa_v_mag_pu_lv = (pypsa.buses_t['v_mag_pu'][
        list(lv_generators_mapping) +
        list(lv_storages_mapping) +
        list(lv_loads_mapping)+
        list(lv_buses_mapping)]).rename(columns=names_mapping)
    edisgo.results.pfa_v_mag_pu = pd.concat(
        {'mv': pfa_v_mag_pu_mv.loc[timesteps, :],
         'lv': pfa_v_mag_pu_lv.loc[timesteps, :]}, axis=1)


