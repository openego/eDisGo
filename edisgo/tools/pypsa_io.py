"""
This modules provides tools to convert graph based representation of the grid
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA grid
container.
"""

import numpy as np
import pandas as pd
import itertools
from math import pi, sqrt
from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe
from networkx import connected_component_subgraphs
import collections

from edisgo.grid.components import Transformer, Line, LVStation, MVStation
from edisgo.grid.grids import LVGrid


def to_pypsa(network, mode, timesteps):
    """
    Translate graph based grid representation to PyPSA Network

    For details from a user perspective see API documentation of
    :meth:`~.grid.network.EDisGo.analyze` of the API class
    :class:`~.grid.network.EDisGo`.

    Translating eDisGo's grid topology to PyPSA representation is structured
    into translating the topology and adding time series for components of the
    grid. In both cases translation of MV grid only (`mode='mv'`), LV grid only
    (`mode='lv'`), MV and LV (`mode=None`) share some code. The
    code is organized as follows:

    * Medium-voltage only (`mode='mv'`): All medium-voltage grid components are
      exported by :func:`mv_to_pypsa` including the LV station. LV grid load
      and generation is considered using :func:`add_aggregated_lv_components`.
      Time series are collected by `_pypsa_load_timeseries` (as example
      for loads, generators and buses) specifying `mode='mv'`). Timeseries
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
    * Duplicate labels in components DataFrames and components' time series
      DataFrames

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        eDisGo grid container
    mode : str
        Determines grid levels that are translated to
        `PyPSA grid representation
        <https://www.pypsa.org/doc/components.html#network>`_. Specify

        * None to export MV and LV grid levels. None is the default.
        * ('mv' to export MV grid level only. This includes cumulative load and
          generation from underlying LV grid aggregated at respective LV
          station. This option is implemented, though the rest of edisgo does
          not handle it yet.)
        * ('lv' to export LV grid level only. This option is not yet
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

    # get topology and time series data
    if mode is None:
        mv_components = mv_to_pypsa(network)
        lv_components = lv_to_pypsa(network)
        components = combine_mv_and_lv(mv_components, lv_components)

        if list(components['Load'].index.values):
            timeseries_load_p, timeseries_load_q = _pypsa_load_timeseries(
                network, mode=mode, timesteps=timesteps)

        if len(list(components['Generator'].index.values)) > 1:
            timeseries_gen_p, timeseries_gen_q = _pypsa_generator_timeseries(
                network, mode=mode, timesteps=timesteps)

        if list(components['Bus'].index.values):
            timeseries_bus_v_set = _pypsa_bus_timeseries(
                network, components['Bus'].index.tolist(), timesteps=timesteps)

        if len(list(components['StorageUnit'].index.values)) > 0:
            timeseries_storage_p, timeseries_storage_q = \
                _pypsa_storage_timeseries(
                    network, mode=mode, timesteps=timesteps)

    elif mode is 'mv':
        # the pypsa export works but NotImplementedError is raised since the
        # rest of edisgo (handling of results from pfa, grid expansion, etc.)
        # does not yet work
        raise NotImplementedError
        mv_components = mv_to_pypsa(network)
        components = add_aggregated_lv_components(network, mv_components)

        if list(components['Load'].index.values):
            timeseries_load_p, timeseries_load_q = _pypsa_load_timeseries(
                network, mode=mode, timesteps=timesteps)

        if len(list(components['Generator'].index.values)) > 1:
            timeseries_gen_p, timeseries_gen_q = _pypsa_generator_timeseries(
                network, mode=mode, timesteps=timesteps)

        if list(components['Bus'].index.values):
            timeseries_bus_v_set = _pypsa_bus_timeseries(
                network, components['Bus'].index.tolist(), timesteps=timesteps)

        if len(list(components['StorageUnit'].index.values)) > 0:
            timeseries_storage_p, timeseries_storage_q = \
                _pypsa_storage_timeseries(
                    network, mode=mode, timesteps=timesteps)

    elif mode is 'lv':
        raise NotImplementedError
        #lv_to_pypsa(network)
    else:
        raise ValueError("Provide proper mode or leave it empty to export "
                         "entire grid topology.")

    # check topology
    _check_topology(components)

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.edisgo_mode = mode
    pypsa_network.set_snapshots(timesteps)

    # import grid topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(components['Bus'], 'Bus')

    for k, comps in components.items():
        if k is not 'Bus' and not comps.empty:
            pypsa_network.import_components_from_dataframe(comps, k)

    # import time series to PyPSA network
    if len(list(components['Generator'].index.values)) > 1:
        import_series_from_dataframe(pypsa_network, timeseries_gen_p,
                                     'Generator', 'p_set')
        import_series_from_dataframe(pypsa_network, timeseries_gen_q,
                                     'Generator', 'q_set')

    if list(components['Load'].index.values):
        import_series_from_dataframe(pypsa_network, timeseries_load_p,
                                     'Load', 'p_set')
        import_series_from_dataframe(pypsa_network, timeseries_load_q,
                                     'Load', 'q_set')

    if list(components['Bus'].index.values):
        import_series_from_dataframe(pypsa_network, timeseries_bus_v_set,
                                     'Bus', 'v_mag_pu_set')

    if len(list(components['StorageUnit'].index.values)) > 0:
        import_series_from_dataframe(pypsa_network, timeseries_storage_p,
                                     'StorageUnit', 'p_set')
        import_series_from_dataframe(pypsa_network, timeseries_storage_q,
                                     'StorageUnit', 'q_set')

    _check_integrity_of_pypsa(pypsa_network)

    return pypsa_network


def mv_to_pypsa(network):
    """Translate MV grid topology representation to PyPSA format

    MV grid topology translated here includes

    * MV station (no transformer, see :meth:`~.grid.network.EDisGo.analyze`)
    * Loads, Generators, Lines, Storages, Branch Tees of MV grid level as well
      as LV stations. LV stations do not have load and generation of LV level.

    Parameters
    ----------
    network : Network
        eDisGo grid container

    Returns
    -------
    dict of :pandas:`pandas.DataFrame<dataframe>`
        A DataFrame for each type of PyPSA components constituting the grid
        topology. Keys included

        * 'Generator'
        * 'Load'
        * 'Line'
        * 'BranchTee'
        * 'Transformer'
        * 'StorageUnit'

    .. warning::

        PyPSA takes resistance R and reactance X in p.u. The conversion from
        values in ohm to pu notation is performed by following equations

        .. math::

            r_{p.u.} = R_{\Omega} / Z_{B}

            x_{p.u.} = X_{\Omega} / Z_{B}

            with

            Z_{B} = V_B / S_B

        I'm quite sure, but its not 100 % clear if the base voltage V_B is
        chosen correctly. We take the primary side voltage of transformer as
        the transformers base voltage. See
        `#54 <https://github.com/openego/eDisGo/issues/54>`_ for discussion.
    """

    generators = network.mv_grid.generators
    loads = network.mv_grid.graph.nodes_by_attribute('load')
    branch_tees = network.mv_grid.graph.nodes_by_attribute('branch_tee')
    lines = network.mv_grid.graph.lines()
    lv_stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    mv_stations = network.mv_grid.graph.nodes_by_attribute('mv_station')
    disconnecting_points = network.mv_grid.graph.nodes_by_attribute(
        'mv_disconnecting_point')
    storages = network.mv_grid.graph.nodes_by_attribute(
        'storage')

    omega = 2 * pi * 50

    # define required dataframe columns for components
    generator = {'name': [],
                 'bus': [],
                 'control': [],
                 'p_nom': [],
                 'type': []}

    bus = {'name': [], 'v_nom': [], 'x': [], 'y': []}

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

    storage = {
        'name': [],
        'bus': [],
        'p_nom': [],
        'state_of_charge_initial': [],
        'efficiency_store': [],
        'efficiency_dispatch': [],
        'standing_loss': []}

    # create dataframe representing generators and associated buses
    for gen in generators:
        bus_name = '_'.join(['Bus', repr(gen)])
        generator['name'].append(repr(gen))
        generator['bus'].append(bus_name)
        generator['control'].append('PQ')
        generator['p_nom'].append(gen.nominal_capacity / 1e3)
        generator['type'].append('_'.join([gen.type, gen.subtype]))

        bus['name'].append(bus_name)
        bus['v_nom'].append(gen.grid.voltage_nom)
        bus['x'].append(gen.geom.x)
        bus['y'].append(gen.geom.y)

    # create dataframe representing branch tees
    for bt in branch_tees:
        bus['name'].append('_'.join(['Bus', repr(bt)]))
        bus['v_nom'].append(bt.grid.voltage_nom)
        bus['x'].append(bt.geom.x)
        bus['y'].append(bt.geom.y)

    # create dataframes representing loads and associated buses
    for lo in loads:
        bus_name = '_'.join(['Bus', repr(lo)])
        load['name'].append(repr(lo))
        load['bus'].append(bus_name)

        bus['name'].append(bus_name)
        bus['v_nom'].append(lo.grid.voltage_nom)
        bus['x'].append(lo.geom.x)
        bus['y'].append(lo.geom.y)

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
            l['line'].type['L'] * omega / 1e3 * l['line'].length)
        line['r'].append(l['line'].type['R'] * l['line'].length)
        line['s_nom'].append(
            sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
        line['length'].append(l['line'].length)

    # create dataframe for LV stations incl. primary/secondary side bus
    for lv_st in lv_stations:
        transformer_count = 1
        # add primary side bus (bus0)
        bus0_name = '_'.join(['Bus', lv_st.__repr__(side='mv')])
        bus['name'].append(bus0_name)
        bus['v_nom'].append(lv_st.mv_grid.voltage_nom)
        bus['x'].append(lv_st.geom.x)
        bus['y'].append(lv_st.geom.y)

        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', lv_st.__repr__(side='lv')])
        bus['name'].append(bus1_name)
        bus['v_nom'].append(lv_st.transformers[0].voltage_op)
        bus['x'].append(None)
        bus['y'].append(None)

        # we choose voltage of transformers' primary side
        v_base = lv_st.mv_grid.voltage_nom

        for tr in lv_st.transformers:
            z_base = v_base ** 2 / tr.type.S_nom
            transformer['name'].append(
                '_'.join([repr(lv_st), 'transformer', str(transformer_count)]))
            transformer['bus0'].append(bus0_name)
            transformer['bus1'].append(bus1_name)
            transformer['type'].append("")
            transformer['model'].append('pi')
            transformer['r'].append(tr.type.R / z_base)
            transformer['x'].append(tr.type.X / z_base)
            transformer['s_nom'].append(tr.type.S_nom / 1e3)
            transformer['tap_ratio'].append(1)

            transformer_count += 1

    # create dataframe for MV stations (only secondary side bus)
    for mv_st in mv_stations:
        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', mv_st.__repr__(side='mv')])
        bus['name'].append(bus1_name)
        bus['v_nom'].append(mv_st.transformers[0].voltage_op)
        bus['x'].append(mv_st.geom.x)
        bus['y'].append(mv_st.geom.y)

    # create dataframe representing disconnecting points
    for dp in disconnecting_points:
        bus['name'].append('_'.join(['Bus', repr(dp)]))
        bus['v_nom'].append(dp.grid.voltage_nom)
        bus['x'].append(dp.geom.x)
        bus['y'].append(dp.geom.y)

    # create dataframe representing storages
    for sto in storages:
        bus_name = '_'.join(['Bus', repr(sto)])

        storage['name'].append(repr(sto))
        storage['bus'].append(bus_name)
        storage['p_nom'].append(sto.nominal_power / 1e3)
        storage['state_of_charge_initial'].append(sto.soc_initial)
        storage['efficiency_store'].append(sto.efficiency_in)
        storage['efficiency_dispatch'].append(sto.efficiency_out)
        storage['standing_loss'].append(sto.standing_loss)

        bus['name'].append(bus_name)
        bus['v_nom'].append(sto.grid.voltage_nom)
        bus['x'].append(sto.geom.x)
        bus['y'].append(sto.geom.y)

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
        'Transformer': pd.DataFrame(transformer).set_index('name'),
        'StorageUnit': pd.DataFrame(storage).set_index('name')}

    return components


def lv_to_pypsa(network):
    """
    Convert LV grid topology to PyPSA representation

    Includes grid topology of all LV grids of :attr:`~.grid.grid.Grid.lv_grids`

    Parameters
    ----------
    network : Network
        eDisGo grid container

    Returns
    -------
    dict of :pandas:`pandas.DataFrame<dataframe>`
        A DataFrame for each type of PyPSA components constituting the grid
        topology. Keys included

        * 'Generator'
        * 'Load'
        * 'Line'
        * 'BranchTee'
        * 'StorageUnit'
    """

    generators = []
    loads = []
    branch_tees = []
    lines = []
    lv_stations = []
    storages = []

    for lv_grid in network.mv_grid.lv_grids:
        generators.extend(lv_grid.generators)
        loads.extend(lv_grid.graph.nodes_by_attribute('load'))
        branch_tees.extend(lv_grid.graph.nodes_by_attribute('branch_tee'))
        lines.extend(lv_grid.graph.lines())
        lv_stations.extend(lv_grid.graph.nodes_by_attribute('lv_station'))
        storages.extend(lv_grid.graph.nodes_by_attribute('storage'))

    omega = 2 * pi * 50

    generator = {'name': [],
                 'bus': [],
                 'control': [],
                 'p_nom': [],
                 'type': []}

    bus = {'name': [], 'v_nom': [], 'x': [], 'y': []}

    load = {'name': [], 'bus': []}

    line = {'name': [],
            'bus0': [],
            'bus1': [],
            'type': [],
            'x': [],
            'r': [],
            's_nom': [],
            'length': []}

    storage = {
        'name': [],
        'bus': [],
        'p_nom': [],
        'state_of_charge_initial': [],
        'efficiency_store': [],
        'efficiency_dispatch': [],
        'standing_loss': []}

    # create dictionary representing generators and associated buses
    for gen in generators:
        bus_name = '_'.join(['Bus', repr(gen)])
        generator['name'].append(repr(gen))
        generator['bus'].append(bus_name)
        generator['control'].append('PQ')
        generator['p_nom'].append(gen.nominal_capacity / 1e3)
        generator['type'].append('_'.join([gen.type, gen.subtype]))

        bus['name'].append(bus_name)
        bus['v_nom'].append(gen.grid.voltage_nom)
        bus['x'].append(None)
        bus['y'].append(None)

    # create dictionary representing branch tees
    for bt in branch_tees:
        bus['name'].append('_'.join(['Bus', repr(bt)]))
        bus['v_nom'].append(bt.grid.voltage_nom)
        bus['x'].append(None)
        bus['y'].append(None)

    # create dataframes representing loads and associated buses
    for lo in loads:
        bus_name = '_'.join(['Bus', repr(lo)])
        load['name'].append(repr(lo))
        load['bus'].append(bus_name)

        bus['name'].append(bus_name)
        bus['v_nom'].append(lo.grid.voltage_nom)
        bus['x'].append(None)
        bus['y'].append(None)

    # create dataframe for lines
    for l in lines:
        line['name'].append(repr(l['line']))

        if l['adj_nodes'][0] in lv_stations:
            line['bus0'].append(
                '_'.join(['Bus', l['adj_nodes'][0].__repr__(side='lv')]))
        else:
            line['bus0'].append('_'.join(['Bus', repr(l['adj_nodes'][0])]))

        if l['adj_nodes'][1] in lv_stations:
            line['bus1'].append(
                '_'.join(['Bus', l['adj_nodes'][1].__repr__(side='lv')]))
        else:
            line['bus1'].append('_'.join(['Bus', repr(l['adj_nodes'][1])]))

        line['type'].append("")
        line['x'].append(
            l['line'].type['L'] * omega / 1e3 * l['line'].length)
        line['r'].append(l['line'].type['R'] * l['line'].length)
        line['s_nom'].append(
            sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
        line['length'].append(l['line'].length)

    # create dataframe representing storages
    for sto in storages:
        bus_name = '_'.join(['Bus', repr(sto)])

        storage['name'].append(repr(sto))
        storage['bus'].append(bus_name)
        storage['p_nom'].append(sto.nominal_power)
        storage['state_of_charge_initial'].append(sto.soc_initial)
        storage['efficiency_store'].append(sto.efficiency_in)
        storage['efficiency_dispatch'].append(sto.efficiency_out)
        storage['standing_loss'].append(sto.standing_loss)

        bus['name'].append(bus_name)
        bus['v_nom'].append(sto.grid.voltage_nom)
        bus['x'].append(None)
        bus['y'].append(None)

    lv_components = {
        'Generator': pd.DataFrame(generator).set_index('name'),
        'Bus': pd.DataFrame(bus).set_index('name'),
        'Load': pd.DataFrame(load).set_index('name'),
        'Line': pd.DataFrame(line).set_index('name'),
        'StorageUnit': pd.DataFrame(storage).set_index('name')}

    return lv_components


def combine_mv_and_lv(mv, lv):
    """Combine MV and LV grid topology in PyPSA format
    """

    combined = {
        c: pd.concat([mv[c], lv[c]], axis=0) for c in list(lv.keys())
    }

    combined['Transformer'] = mv['Transformer']

    return combined


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


def _pypsa_load_timeseries(network, timesteps, mode=None):
    """
    Time series in PyPSA compatible format for load instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
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
            mv_load_timeseries_q.append(load.pypsa_timeseries('q').rename(
                repr(load)).to_frame().loc[timesteps])
            mv_load_timeseries_p.append(load.pypsa_timeseries('p').rename(
                repr(load)).to_frame().loc[timesteps])
        if mode is 'mv':
            lv_load_timeseries_p, lv_load_timeseries_q = \
                _pypsa_load_timeseries_aggregated_at_lv_station(
                    network, timesteps)

    # add LV grid's loads
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for load in lv_grid.graph.nodes_by_attribute('load'):
                lv_load_timeseries_q.append(load.pypsa_timeseries('q').rename(
                    repr(load)).to_frame().loc[timesteps])
                lv_load_timeseries_p.append(load.pypsa_timeseries('p').rename(
                    repr(load)).to_frame().loc[timesteps])

    load_df_p = pd.concat(mv_load_timeseries_p + lv_load_timeseries_p, axis=1)
    load_df_q = pd.concat(mv_load_timeseries_q + lv_load_timeseries_q, axis=1)

    return load_df_p, load_df_q


def _pypsa_generator_timeseries(network, timesteps, mode=None):
    """Timeseries in PyPSA compatible format for generator instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
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
        for gen in network.mv_grid.generators:
            mv_gen_timeseries_q.append(gen.pypsa_timeseries('q').rename(
                repr(gen)).to_frame().loc[timesteps])
            mv_gen_timeseries_p.append(gen.pypsa_timeseries('p').rename(
                repr(gen)).to_frame().loc[timesteps])
        if mode is 'mv':
            lv_gen_timeseries_p, lv_gen_timeseries_q = \
                _pypsa_generator_timeseries_aggregated_at_lv_station(
                    network, timesteps)

    # LV generator timeseries
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for gen in lv_grid.generators:
                lv_gen_timeseries_q.append(gen.pypsa_timeseries('q').rename(
                    repr(gen)).to_frame().loc[timesteps])
                lv_gen_timeseries_p.append(gen.pypsa_timeseries('p').rename(
                    repr(gen)).to_frame().loc[timesteps])

    gen_df_p = pd.concat(mv_gen_timeseries_p + lv_gen_timeseries_p, axis=1)
    gen_df_q = pd.concat(mv_gen_timeseries_q + lv_gen_timeseries_q, axis=1)

    return gen_df_p, gen_df_q


def _pypsa_storage_timeseries(network, timesteps, mode=None):
    """
    Timeseries in PyPSA compatible format for storage instances

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    timesteps : array_like
        Timesteps is an array-like object with entries of type
        :pandas:`pandas.Timestamp<timestamp>` specifying which time steps
        to export to pypsa representation and use in power flow analysis.
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

    mv_storage_timeseries_q = []
    mv_storage_timeseries_p = []
    lv_storage_timeseries_q = []
    lv_storage_timeseries_p = []

    # MV storage time series
    if mode is 'mv' or mode is None:
        for storage in network.mv_grid.graph.nodes_by_attribute('storage'):
            mv_storage_timeseries_q.append(
                storage.pypsa_timeseries('q').rename(
                    repr(storage)).to_frame().loc[timesteps])
            mv_storage_timeseries_p.append(
                storage.pypsa_timeseries('p').rename(
                    repr(storage)).to_frame().loc[timesteps])

    # LV storage time series
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for storage in lv_grid.graph.nodes_by_attribute('storage'):
                lv_storage_timeseries_q.append(
                    storage.pypsa_timeseries('q').rename(
                        repr(storage)).to_frame().loc[timesteps])
                lv_storage_timeseries_p.append(
                    storage.pypsa_timeseries('p').rename(
                        repr(storage)).to_frame().loc[timesteps])

    storage_df_p = pd.concat(
        mv_storage_timeseries_p + lv_storage_timeseries_p, axis=1)
    storage_df_q = pd.concat(
        mv_storage_timeseries_q + lv_storage_timeseries_q, axis=1)

    return storage_df_p, storage_df_q


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
        The eDisGo grid topology model overall container
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
    Aggregates generator time series per generator subtype and LV grid.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
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
    Aggregates load time series per sector and LV grid.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
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
                                           collections.Counter(comps.index.values).items()
                                           if count > 1])
                             )


def _check_integrity_of_pypsa(pypsa_network):
    """"""

    # check for sub-networks
    subgraphs = list(connected_component_subgraphs(pypsa_network.graph()))
    pypsa_network.determine_network_topology()

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
        duplicate_v_mag_set.append(pypsa_network.buses_t['v_mag_pu_set'].columns[
                                    pypsa_network.buses_t[
                                        'v_mag_pu_set'].columns.duplicated()])
        
    if duplicate_v_mag_set:
        raise ValueError("{labels} have duplicate entry in buses_t".format(
            labels=duplicate_v_mag_set))


def process_pfa_results(network, pypsa, timesteps):
    """
    Assing values from PyPSA to
    :meth:`results <edisgo.grid.network.Network.results>`

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
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
    :class:`~.grid.network.Results`
        Understand how results of power flow analysis are structured in eDisGo.

    """
    # get the absolute losses in the system
    # subtracting total generation (including slack) from total load
    grid_losses = {'p': 1e3 * (pypsa.generators_t['p'].sum(axis=1) -
                               pypsa.loads_t['p'].sum(axis=1)),
                   'q': 1e3 * (pypsa.generators_t['q'].sum(axis=1) -
                               pypsa.loads_t['q'].sum(axis=1))}

    network.results.grid_losses = pd.DataFrame(grid_losses).loc[timesteps, :]

    # get slack results
    grid_exchanges = {'p': 1e3 * (pypsa.generators_t['p']['Generator_slack']),
                      'q': 1e3 * (pypsa.generators_t['q']['Generator_slack'])}

    network.results.hv_mv_exchanges = pd.DataFrame(grid_exchanges).loc[timesteps, :]

    # get p and q of lines, LV transformers and MV Station (slack generator)
    # in absolute values
    q0 = pd.concat(
        [np.abs(pypsa.lines_t['q0']),
         np.abs(pypsa.transformers_t['q0']),
         np.abs(pypsa.generators_t['q']['Generator_slack'].rename(
             repr(network.mv_grid.station)))], axis=1).loc[timesteps, :]
    q1 = pd.concat(
        [np.abs(pypsa.lines_t['q1']),
         np.abs(pypsa.transformers_t['q1']),
         np.abs(pypsa.generators_t['q']['Generator_slack'].rename(
             repr(network.mv_grid.station)))], axis=1).loc[timesteps, :]
    p0 = pd.concat(
        [np.abs(pypsa.lines_t['p0']),
         np.abs(pypsa.transformers_t['p0']),
         np.abs(pypsa.generators_t['p']['Generator_slack'].rename(
            repr(network.mv_grid.station)))], axis=1).loc[timesteps, :]
    p1 = pd.concat(
        [np.abs(pypsa.lines_t['p1']),
         np.abs(pypsa.transformers_t['p1']),
         np.abs(pypsa.generators_t['p']['Generator_slack'].rename(
             repr(network.mv_grid.station)))], axis=1).loc[timesteps, :]

    # determine apparent power and line endings/transformers' side
    s0 = np.hypot(p0, q0)
    s1 = np.hypot(p1, q1)

    # choose p and q from line ending with max(s0,s1)
    network.results.pfa_p = p0.where(s0 > s1, p1) * 1e3
    network.results.pfa_q = q0.where(s0 > s1, q1) * 1e3

    lines_bus0 = pypsa.lines['bus0'].to_dict()
    bus0_v_mag_pu = pypsa.buses_t['v_mag_pu'].T.loc[
                    list(lines_bus0.values()), :].copy()
    bus0_v_mag_pu.index = list(lines_bus0.keys())

    lines_bus1 = pypsa.lines['bus1'].to_dict()
    bus1_v_mag_pu = pypsa.buses_t['v_mag_pu'].T.loc[
                    list(lines_bus1.values()), :].copy()
    bus1_v_mag_pu.index = list(lines_bus1.keys())

    line_voltage_avg = 0.5 * (bus0_v_mag_pu.loc[:, timesteps] +
                              bus1_v_mag_pu.loc[:, timesteps])

    # Get voltage levels at line (avg. of buses at both sides)
    network.results._i_res = s0[pypsa.lines_t['q0'].columns].truediv(
        pypsa.lines['v_nom'] * line_voltage_avg.T, axis='columns') * 1e3
    # process results at nodes
    generators_names = [repr(g) for g in network.mv_grid.generators]
    generators_mapping = {v: k for k, v in
                          pypsa.generators.loc[generators_names][
                              'bus'].to_dict().items()}
    storages_names = [repr(g) for g in
                      network.mv_grid.graph.nodes_by_attribute('storage')]
    storages_mapping = {v: k for k, v in
                        pypsa.storage_units.loc[storages_names][
                            'bus'].to_dict().items()}
    branch_t_names = [repr(bt) for bt in
                      network.mv_grid.graph.nodes_by_attribute('branch_tee')]
    branch_t_mapping = {'_'.join(['Bus', v]): v for v in branch_t_names}
    mv_station_names = [repr(m) for m in
                        network.mv_grid.graph.nodes_by_attribute('mv_station')]
    mv_station_mapping_sec = {'_'.join(['Bus', v]): v for v in
                              mv_station_names}
    mv_switch_disconnector_names = [repr(sd) for sd in
                                    network.mv_grid.graph.nodes_by_attribute(
                                        'mv_disconnecting_point')]
    mv_switch_disconnector_mapping = {'_'.join(['Bus', v]): v for v in
                                      mv_switch_disconnector_names}

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

    lv_generators_names = []
    lv_storages_names = []
    lv_branch_t_names = []
    lv_loads_names = []
    for lv_grid in network.mv_grid.lv_grids:
        lv_generators_names.extend([repr(g) for g in
                                    lv_grid.graph.nodes_by_attribute(
                                        'generator')])
        lv_storages_names.extend([repr(g) for g in
                                  lv_grid.graph.nodes_by_attribute(
                                      'storage')])
        lv_branch_t_names.extend([repr(bt) for bt in
                             lv_grid.graph.nodes_by_attribute('branch_tee')])
        lv_loads_names.extend([repr(lo) for lo in
                          lv_grid.graph.nodes_by_attribute('load')])

    lv_generators_mapping = {v: k for k, v in
                             pypsa.generators.loc[lv_generators_names][
                                 'bus'].to_dict().items()}
    lv_storages_mapping = {v: k for k, v in
                           pypsa.storage_units.loc[lv_storages_names][
                               'bus'].to_dict().items()}
    lv_branch_t_mapping = {'_'.join(['Bus', v]): v for v in lv_branch_t_names}
    lv_loads_mapping = {v: k for k, v in pypsa.loads.loc[lv_loads_names][
        'bus'].to_dict().items()}

    names_mapping = {
        **generators_mapping,
        **storages_mapping,
        **branch_t_mapping,
        **mv_station_mapping_sec,
        **lv_station_mapping_pri,
        **lv_station_mapping_sec,
        **mv_switch_disconnector_mapping,
        **loads_mapping,
        **lv_generators_mapping,
        **lv_storages_mapping,
        **lv_loads_mapping,
        **lv_branch_t_mapping
    }

    # write voltage levels obtained from power flow to results object
    pfa_v_mag_pu_mv = (pypsa.buses_t['v_mag_pu'][
        list(generators_mapping) +
        list(storages_mapping) +
        list(branch_t_mapping) +
        list(mv_station_mapping_sec) +
        list(mv_switch_disconnector_mapping) +
        list(lv_station_mapping_pri) +
        list(loads_mapping)]).rename(columns=names_mapping)
    pfa_v_mag_pu_lv = (pypsa.buses_t['v_mag_pu'][
        list(lv_station_mapping_sec) +
        list(lv_generators_mapping) +
        list(lv_storages_mapping) +
        list(lv_branch_t_mapping) +
        list(lv_loads_mapping)]).rename(columns=names_mapping)
    network.results.pfa_v_mag_pu = pd.concat(
        {'mv': pfa_v_mag_pu_mv.loc[timesteps, :],
         'lv': pfa_v_mag_pu_lv.loc[timesteps, :]}, axis=1)


def update_pypsa_generator_import(network):
    """
    Translate graph based grid representation to PyPSA Network

    For details from a user perspective see API documentation of
    :meth:`~.grid.network.EDisGo.analyze` of the API class
    :class:`~.grid.network.EDisGo`.

    Translating eDisGo's grid topology to PyPSA representation is structured
    into translating the topology and adding time series for components of the
    grid. In both cases translation of MV grid only (`mode='mv'`), LV grid only
    (`mode='lv'`), MV and LV (`mode=None`) share some code. The
    code is organized as follows:

    * Medium-voltage only (`mode='mv'`): All medium-voltage grid components are
      exported by :func:`mv_to_pypsa` including the LV station. LV grid load
      and generation is considered using :func:`add_aggregated_lv_components`.
      Time series are collected by `_pypsa_load_timeseries` (as example
      for loads, generators and buses) specifying `mode='mv'`). Timeseries
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
    * Duplicate labels in components DataFrames and components' time series
      DataFrames

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        eDisGo grid container
    mode : str
        Determines grid levels that are translated to
        `PyPSA grid representation
        <https://www.pypsa.org/doc/components.html#network>`_. Specify

        * None to export MV and LV grid levels. None is the default.
        * ('mv' to export MV grid level only. This includes cumulative load and
          generation from underlying LV grid aggregated at respective LV
          station. This option is implemented, though the rest of edisgo does
          not handle it yet.)
        * ('lv' to export LV grid level only. This option is not yet
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

    # get topology and time series data
    if network.pypsa.edisgo_mode is None:
        mv_components = mv_to_pypsa(network)
        lv_components = lv_to_pypsa(network)
        components = combine_mv_and_lv(mv_components, lv_components)
    elif network.pypsa.edisgo_mode is 'mv':
        raise NotImplementedError
    elif network.pypsa.edisgo_mode is 'lv':
        raise NotImplementedError
    else:
        raise ValueError("Provide proper mode or leave it empty to export "
                         "entire grid topology.")

    # check topology
    _check_topology(components)

    # create power flow problem
    pypsa_network = PyPSANetwork()
    pypsa_network.edisgo_mode = network.pypsa.edisgo_mode
    pypsa_network.set_snapshots(network.pypsa.snapshots)

    # import grid topology to PyPSA network
    # buses are created first to avoid warnings
    pypsa_network.import_components_from_dataframe(components['Bus'], 'Bus')

    for k, comps in components.items():
        if k is not 'Bus' and not comps.empty:
            pypsa_network.import_components_from_dataframe(comps, k)

    # import time series to PyPSA network
    pypsa_network.generators_t.p_set = network.pypsa.generators_t.p_set
    pypsa_network.generators_t.q_set = network.pypsa.generators_t.q_set
    pypsa_network.loads_t.p_set = network.pypsa.loads_t.p_set
    pypsa_network.loads_t.q_set = network.pypsa.loads_t.q_set
    pypsa_network.storage_units_t.p_set = network.pypsa.storage_units_t.p_set
    pypsa_network.storage_units_t.q_set = network.pypsa.storage_units_t.q_set
    pypsa_network.buses_t.v_mag_pu_set = network.pypsa.buses_t.v_mag_pu_set

    network.pypsa = pypsa_network

    if len(list(components['Generator'].index.values)) > 1:
        update_pypsa_generator_timeseries(network)
    if list(components['Bus'].index.values):
        update_pypsa_bus_timeseries(network)
    if len(list(components['StorageUnit'].index.values)) > 0:
        update_pypsa_storage_timeseries(network)

    _check_integrity_of_pypsa(pypsa_network)


def update_pypsa_grid_reinforcement(network, equipment_changes):
    """
    Update equipment data of lines and transformers after grid reinforcement.

    During grid reinforcement (cf.
    :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid`) grid topology and
    equipment of lines and transformers are changed.
    In order to save time and not do a full translation of eDisGo's grid
    topology to the PyPSA format, this function provides an updater for data
    that may change during grid reinforcement.

    The PyPSA grid topology :meth:`edisgo.grid.network.Network.pypsa` is update
    by changed equipment stored in
    :attr:`edisgo.grid.network.Network.equipment_changes`.

    Parameters
    ----------
    network : Network
        eDisGo grid container
    equipment_changes : `pandas.DataFrame<dataframe>`
        Dataframe with latest equipment changes (of latest iteration step)
        from grid reinforcement. See `equipment_changes` property of
        :class:`~.grid.network.Results` for more information on the Dataframe.

    """

    # Step 1: Update transformers
    transformers = equipment_changes[
        equipment_changes['equipment'].apply(isinstance, args=(Transformer,))]
    # HV/MV transformers are excluded because it's not part of the PFA
    removed_transformers = [repr(_) for _ in
                            transformers[transformers['change'] == 'removed'][
                                'equipment'].tolist() if _.voltage_op < 10]
    added_transformers = transformers[transformers['change'] == 'added']

    transformer = {'name': [],
                   'bus0': [],
                   'bus1': [],
                   'type': [],
                   'model': [],
                   'x': [],
                   'r': [],
                   's_nom': [],
                   'tap_ratio': []}

    for idx, row in added_transformers.iterrows():

        if isinstance(idx, LVStation):
            # we choose voltage of transformers' primary side
            v_base = idx.mv_grid.voltage_nom
            z_base = v_base ** 2 / row['equipment'].type.S_nom

            transformer['bus0'].append('_'.join(['Bus', idx.__repr__(side='mv')]))
            transformer['bus1'].append('_'.join(['Bus', idx.__repr__(side='lv')]))
            transformer['name'].append(repr(row['equipment']))
            transformer['type'].append("")
            transformer['model'].append('pi')
            transformer['r'].append(row['equipment'].type.R / z_base)
            transformer['x'].append(row['equipment'].type.X / z_base)
            transformer['s_nom'].append(row['equipment'].type.S_nom / 1e3)
            transformer['tap_ratio'].append(1)

    network.pypsa.transformers.drop(removed_transformers, inplace=True)

    if transformer['name']:
        network.pypsa.import_components_from_dataframe(
            pd.DataFrame(transformer).set_index('name'), 'Transformer')

    # Step 2: Update lines
    lines = equipment_changes.loc[equipment_changes.index[
        equipment_changes.reset_index()['index'].apply(
            isinstance, args=(Line,))]]
    changed_lines = lines[lines['change'] == 'changed']

    lv_stations = network.mv_grid.graph.nodes_by_attribute('lv_station')

    omega = 2 * pi * 50

    for idx, row in changed_lines.iterrows():
        # Update line parameters
        network.pypsa.lines.loc[repr(idx), 'r'] = (
            idx.type['R'] / idx.quantity * idx.length)
        network.pypsa.lines.loc[repr(idx), 'x'] = (
            idx.type['L'] / 1e3 * omega / idx.quantity * idx.length)
        # ToDo remove s_nom?
        network.pypsa.lines.loc[repr(idx), 's_nom'] = (
            sqrt(3) * idx.type['I_max_th'] * idx.type[
                'U_n'] * idx.quantity / 1e3)
        network.pypsa.lines.loc[repr(idx), 'length'] = idx.length

        # Update buses line is connected to
        adj_nodes = idx.grid.graph.nodes_from_line(idx)

        if adj_nodes[0] in lv_stations:
            bus0 = '_'.join(['Bus', adj_nodes[0].__repr__(side='mv')])
        elif adj_nodes[0] is network.mv_grid.station:
            bus0 = '_'.join(['Bus', adj_nodes[0].__repr__(side='lv')])
        else:
            bus0 = '_'.join(['Bus', repr(adj_nodes[0])])

        if adj_nodes[1] in lv_stations:
            bus1 = '_'.join(['Bus', adj_nodes[1].__repr__(side='mv')])
        elif adj_nodes[1] is network.mv_grid.station:
            bus1 = '_'.join(['Bus', adj_nodes[1].__repr__(side='lv')])
        else:
            bus1 = '_'.join(['Bus', repr(adj_nodes[1])])

        network.pypsa.lines.loc[repr(idx), 'bus0'] = bus0
        network.pypsa.lines.loc[repr(idx), 'bus1'] = bus1


def update_pypsa_storage(pypsa, storages, storages_lines):
    """
    Adds storages and their lines to pypsa representation of the edisgo graph.

    This function effects the following attributes of the pypsa network:
    components ('StorageUnit'), storage_units, storage_units_t (p_set, q_set),
    buses, lines

    Parameters
    -----------
    pypsa : :pypsa:`pypsa.Network<network>`
    storages : :obj:`list`
        List with storages of type :class:`~.grid.components.Storage` to add
        to pypsa network.
    storages_lines : :obj:`list`
        List with lines of type :class:`~.grid.components.Line` that connect
        storages to the grid.

    """
    bus = {'name': [], 'v_nom': [], 'x': [], 'y': []}

    line = {'name': [],
            'bus0': [],
            'bus1': [],
            'type': [],
            'x': [],
            'r': [],
            's_nom': [],
            'length': []}

    storage = {
        'name': [],
        'bus': [],
        'p_nom': [],
        'state_of_charge_initial': [],
        'efficiency_store': [],
        'efficiency_dispatch': [],
        'standing_loss': []}

    for s in storages:
        bus_name = '_'.join(['Bus', repr(s)])

        storage['name'].append(repr(s))
        storage['bus'].append(bus_name)
        storage['p_nom'].append(s.nominal_power / 1e3)
        storage['state_of_charge_initial'].append(s.soc_initial)
        storage['efficiency_store'].append(s.efficiency_in)
        storage['efficiency_dispatch'].append(s.efficiency_out)
        storage['standing_loss'].append(s.standing_loss)

        bus['name'].append(bus_name)
        bus['v_nom'].append(s.grid.voltage_nom)
        bus['x'].append(s.geom.x)
        bus['y'].append(s.geom.y)

    omega = 2 * pi * 50
    for l in storages_lines:
        line['name'].append(repr(l))

        adj_nodes = l.grid.graph.nodes_from_line(l)
        if isinstance(l.grid, LVGrid):
            if isinstance(adj_nodes[0], LVStation):
                line['bus0'].append(
                    '_'.join(['Bus', adj_nodes[0].__repr__(side='lv')]))
            else:
                line['bus0'].append('_'.join(['Bus', repr(adj_nodes[0])]))

            if isinstance(adj_nodes[1], LVStation):
                line['bus1'].append(
                    '_'.join(['Bus', adj_nodes[1].__repr__(side='lv')]))
            else:
                line['bus1'].append('_'.join(['Bus', repr(adj_nodes[1])]))
        else:
            if isinstance(adj_nodes[0], LVStation):
                line['bus0'].append(
                    '_'.join(['Bus', adj_nodes[0].__repr__(side='mv')]))
            elif isinstance(adj_nodes[0], MVStation):
                line['bus0'].append(
                    '_'.join(['Bus', adj_nodes[0].__repr__(side='lv')]))
            else:
                line['bus0'].append('_'.join(['Bus', repr(adj_nodes[0])]))

            if isinstance(adj_nodes[1], LVStation):
                line['bus1'].append(
                    '_'.join(['Bus', adj_nodes[1].__repr__(side='mv')]))
            elif isinstance(adj_nodes[1], MVStation):
                line['bus1'].append(
                    '_'.join(['Bus', adj_nodes[1].__repr__(side='lv')]))
            else:
                line['bus1'].append('_'.join(['Bus', repr(adj_nodes[1])]))

        line['type'].append("")
        line['x'].append(l.type['L'] * omega / 1e3 * l.length)
        line['r'].append(l.type['R'] * l.length)
        line['s_nom'].append(
            sqrt(3) * l.type['I_max_th'] * l.type['U_n'] / 1e3)
        line['length'].append(l.length)

    # import new components to pypsa
    pypsa.import_components_from_dataframe(
        pd.DataFrame(bus).set_index('name'), 'Bus')
    pypsa.import_components_from_dataframe(
        pd.DataFrame(storage).set_index('name'), 'StorageUnit')
    pypsa.import_components_from_dataframe(
        pd.DataFrame(line).set_index('name'), 'Line')

    # import time series of storages and buses to pypsa
    timeseries_storage_p = pd.DataFrame()
    timeseries_storage_q = pd.DataFrame()
    for s in storages:
        timeseries_storage_p[repr(s)] = s.pypsa_timeseries('p').loc[
            pypsa.storage_units_t.p_set.index]
        timeseries_storage_q[repr(s)] = s.pypsa_timeseries('q').loc[
            pypsa.storage_units_t.q_set.index]

    import_series_from_dataframe(pypsa, timeseries_storage_p,
                                 'StorageUnit', 'p_set')
    import_series_from_dataframe(pypsa, timeseries_storage_q,
                                 'StorageUnit', 'q_set')


def update_pypsa_timeseries(network, loads_to_update=None,
                            generators_to_update=None, storages_to_update=None,
                            timesteps=None):
    """
    Updates load, generator, storage and bus time series in pypsa network.

    See functions :func:`update_pypsa_load_timeseries`,
    :func:`update_pypsa_generator_timeseries`,
    :func:`update_pypsa_storage_timeseries`, and
    :func:`update_pypsa_bus_timeseries` for more information.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    loads_to_update : :obj:`list`, optional
        List with all loads (of type :class:`~.grid.components.Load`) that need
        to be updated. If None all loads are updated depending on mode. See
        :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    generators_to_update : :obj:`list`, optional
        List with all generators (of type :class:`~.grid.components.Generator`)
        that need to be updated. If None all generators are updated depending
        on mode. See :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    storages_to_update : :obj:`list`, optional
        List with all storages (of type :class:`~.grid.components.Storage`)
        that need to be updated. If None all storages are updated depending on
        mode. See :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the load time series to export
        to pypsa representation and use in power flow analysis.
        If None all time steps currently existing in pypsa representation are
        updated. If not None current time steps are overwritten by given
        time steps. Default: None.

    """
    update_pypsa_load_timeseries(
        network, loads_to_update=loads_to_update, timesteps=timesteps)
    update_pypsa_generator_timeseries(
        network, generators_to_update=generators_to_update,
        timesteps=timesteps)
    update_pypsa_storage_timeseries(
        network, storages_to_update=storages_to_update, timesteps=timesteps)
    update_pypsa_bus_timeseries(network, timesteps=timesteps)
    # update pypsa snapshots
    if timesteps is None:
        timesteps = network.pypsa.buses_t.v_mag_pu_set.index
    network.pypsa.set_snapshots(timesteps)


def update_pypsa_load_timeseries(network, loads_to_update=None,
                                 timesteps=None):
    """
    Updates load time series in pypsa representation.

    This function overwrites p_set and q_set of loads_t attribute of
    pypsa network.
    Be aware that if you call this function with `timesteps` and thus overwrite
    current time steps it may lead to inconsistencies in the pypsa network
    since only load time series are updated but none of the other time series
    or the snapshots attribute of the pypsa network. Use the function
    :func:`update_pypsa_timeseries` to change the time steps you want to
    analyse in the power flow analysis.
    This function will also raise an error when a load that is currently not
    in the pypsa representation is added.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    loads_to_update : :obj:`list`, optional
        List with all loads (of type :class:`~.grid.components.Load`) that need
        to be updated. If None all loads are updated depending on mode. See
        :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the load time series to export
        to pypsa representation. If None all time steps currently existing in
        pypsa representation are updated. If not None current time steps are
        overwritten by given time steps. Default: None.

    """
    _update_pypsa_timeseries_by_type(
        network, type='load', components_to_update=loads_to_update,
        timesteps=timesteps)


def update_pypsa_generator_timeseries(network, generators_to_update=None,
                                      timesteps=None):
    """
    Updates generator time series in pypsa representation.

    This function overwrites p_set and q_set of generators_t attribute of
    pypsa network.
    Be aware that if you call this function with `timesteps` and thus overwrite
    current time steps it may lead to inconsistencies in the pypsa network
    since only generator time series are updated but none of the other time
    series or the snapshots attribute of the pypsa network. Use the function
    :func:`update_pypsa_timeseries` to change the time steps you want to
    analyse in the power flow analysis.
    This function will also raise an error when a generator that is currently
    not in the pypsa representation is added.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    generators_to_update : :obj:`list`, optional
        List with all generators (of type :class:`~.grid.components.Generator`)
        that need to be updated. If None all generators are updated depending
        on mode. See :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the generator time series to
        export to pypsa representation. If None all time steps currently
        existing in pypsa representation are updated. If not None current time
        steps are overwritten by given time steps. Default: None.

    """
    _update_pypsa_timeseries_by_type(
        network, type='generator', components_to_update=generators_to_update,
        timesteps=timesteps)


def update_pypsa_storage_timeseries(network, storages_to_update=None,
                                    timesteps=None):
    """
    Updates storage time series in pypsa representation.

    This function overwrites p_set and q_set of storage_unit_t attribute of
    pypsa network.
    Be aware that if you call this function with `timesteps` and thus overwrite
    current time steps it may lead to inconsistencies in the pypsa network
    since only storage time series are updated but none of the other time
    series or the snapshots attribute of the pypsa network. Use the function
    :func:`update_pypsa_timeseries` to change the time steps you want to
    analyse in the power flow analysis.
    This function will also raise an error when a storage that is currently
    not in the pypsa representation is added.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    storages_to_update : :obj:`list`, optional
        List with all storages (of type :class:`~.grid.components.Storage`)
        that need to be updated. If None all storages are updated depending on
        mode. See :meth:`~.tools.pypsa_io.to_pypsa` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the storage time series to
        export to pypsa representation. If None all time steps currently
        existing in pypsa representation are updated. If not None current time
        steps are overwritten by given time steps. Default: None.

    """
    _update_pypsa_timeseries_by_type(
        network, type='storage', components_to_update=storages_to_update,
        timesteps=timesteps)


def update_pypsa_bus_timeseries(network, timesteps=None):
    """
    Updates buses voltage time series in pypsa representation.

    This function overwrites v_mag_pu_set of buses_t attribute of
    pypsa network.
    Be aware that if you call this function with `timesteps` and thus overwrite
    current time steps it may lead to inconsistencies in the pypsa network
    since only bus time series are updated but none of the other time
    series or the snapshots attribute of the pypsa network. Use the function
    :func:`update_pypsa_timeseries` to change the time steps you want to
    analyse in the power flow analysis.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the time series to
        export to pypsa representation. If None all time steps currently
        existing in pypsa representation are updated. If not None current
        time steps are overwritten by given time steps. Default: None.

    """
    if timesteps is None:
        timesteps = network.pypsa.buses_t.v_mag_pu_set.index
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]
    buses = network.pypsa.buses.index
    v_mag_pu_set = _pypsa_bus_timeseries(network, buses, timesteps)
    network.pypsa.buses_t.v_mag_pu_set = v_mag_pu_set


def _update_pypsa_timeseries_by_type(network, type, components_to_update=None,
                                     timesteps=None):
    """
    Updates time series of specified component in pypsa representation.

    Be aware that if you call this function with `timesteps` and thus overwrite
    current time steps it may lead to inconsistencies in the pypsa network
    since only time series of the specified component are updated but none of
    the other time series or the snapshots attribute of the pypsa network.
    Use the function :func:`update_pypsa_timeseries` to change the time steps
    you want to analyse in the power flow analysis.
    This function will raise an error when a component that is currently not in
    the pypsa representation is added.

    Parameters
    ----------
    network : Network
        The eDisGo grid topology model overall container
    type : :obj:`str`
        Type specifies the type of component (load, generator or storage)
        that is updated.
    components_to_update : :obj:`list`, optional
        List with all components (either of type
        :class:`~.grid.components.Load`, :class:`~.grid.components.Generator`
        or :class:`~.grid.components.Storage`) that need to be updated.
        Possible options are 'load', 'generator' and 'storage'.
        Components in list must all be of the same type. If None all components
        specified by `type` are updated depending on the mode. See
        :meth:`~.tools.pypsa_io.to_pypsa` for more information on mode.
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps of the time series to
        export to pypsa representation. If None all time steps currently
        existing in pypsa representation are updated. If not None current
        time steps are overwritten by given time steps. Default: None.

    """

    # pypsa dataframe to update
    if type == 'load':
        pypsa_ts = network.pypsa.loads_t
        components_in_pypsa = network.pypsa.loads.index
    elif type == 'generator':
        pypsa_ts = network.pypsa.generators_t
        components_in_pypsa = network.pypsa.generators.index
    elif type == 'storage':
        pypsa_ts = network.pypsa.storage_units_t
        components_in_pypsa = network.pypsa.storage_units.index
    else:
        raise ValueError('{} is not a valid type.'.format(type))

    # MV and LV loads
    if network.pypsa.edisgo_mode is None:
        # if no components are specified get all components of specified type
        # in whole grid
        if components_to_update is None:
            grids = [network.mv_grid] + list(network.mv_grid.lv_grids)
            if type == 'generator':
                components_to_update = list(itertools.chain(
                    *[grid.generators for grid in grids]))
            else:
                components_to_update = list(itertools.chain(
                    *[grid.graph.nodes_by_attribute(type) for grid in grids]))
        # if no time steps are specified update all time steps currently
        # contained in pypsa representation
        if timesteps is None:
            timesteps = pypsa_ts.p_set.index
        # check if timesteps is array-like, otherwise convert to list
        # (necessary to avoid getting a scalar using .loc)
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        p_set = pd.DataFrame()
        q_set = pd.DataFrame()
        for comp in components_to_update:
            if repr(comp) in components_in_pypsa:
                p_set[repr(comp)] = comp.pypsa_timeseries('p').loc[timesteps]
                q_set[repr(comp)] = comp.pypsa_timeseries('q').loc[timesteps]
            else:
                raise KeyError("Tried to update component {} but could not "
                               "find it in pypsa network.".format(comp))
        # overwrite pypsa time series
        pypsa_ts.p_set = p_set
        pypsa_ts.q_set = q_set

    # MV and aggregated LV loads
    elif network.pypsa.edisgo_mode is 'mv':
        raise NotImplementedError

    # LV only
    elif network.pypsa.edisgo_mode is 'lv':
        raise NotImplementedError
