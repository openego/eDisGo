"""
This modules provides tools to convert graph based representation of the grid
topology to PyPSA data model. Call :func:`to_pypsa` to retrieve the PyPSA grid
container.
"""

from edisgo.grid.components import Transformer, Line, LVStation

import numpy as np
import pandas as pd
import itertools
from math import pi, sqrt
from pypsa import Network as PyPSANetwork
from pypsa.io import import_series_from_dataframe
from networkx import connected_component_subgraphs
import collections


def to_pypsa(network, mode, timesteps):
    """
    Translate graph based grid representation to PyPSA Network

    For details from a user perspective see API documentation of
    :meth:`~.grid.network.EDisGo.analyze` of the API class
    :class:`~.grid.network.EDisGo`.

    Translating eDisGo's grid topology to PyPSA representation is structured
    into tranlating the topology and adding time series for components of the
    grid. In both cases translation of MV grid only (`mode='mv'`), LV grid only
    (`mode='lv'`), MV and LV (`mode=None`) share some code. The
    code is organized as follows

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
    network : Network
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
    timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
        Timesteps specifies which time steps to export to pypsa representation
        and use in power flow analysis.

    Returns
    -------
        PyPSA Network

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
            timeseries_load_p_set = _pypsa_load_timeseries(
                network, mode=mode, timesteps=timesteps)

        if len(list(components['Generator'].index.values)) > 1:
            timeseries_gen_p_min, timeseries_gen_p_max = \
                _pypsa_generator_timeseries(
                    network, mode=mode, timesteps=timesteps)
            timeseries_storage_p_min, timeseries_storage_p_max = \
                _pypsa_storage_timeseries(
                    network, mode=mode, timesteps=timesteps)

        if list(components['Bus'].index.values):
            timeseries_bus_v_set = _pypsa_bus_timeseries(
                network, components['Bus'].index.tolist(), timesteps=timesteps)
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
        import_series_from_dataframe(pypsa_network, timeseries_gen_p_min,
                                     'Generator', 'p_min_pu')
        import_series_from_dataframe(pypsa_network, timeseries_gen_p_max,
                                     'Generator', 'p_max_pu')
        import_series_from_dataframe(pypsa_network, timeseries_storage_p_min,
                                     'Generator', 'p_min_pu')
        import_series_from_dataframe(pypsa_network, timeseries_storage_p_max,
                                     'Generator', 'p_max_pu')

    if list(components['Load'].index.values):
        import_series_from_dataframe(pypsa_network, timeseries_load_p_set,
                                     'Load', 'p_set')

    if list(components['Bus'].index.values):
        import_series_from_dataframe(pypsa_network, timeseries_bus_v_set,
                                     'Bus', 'v_mag_pu_set')

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

    generators = network.mv_grid.graph.nodes_by_attribute('generator') + \
                 network.mv_grid.graph.nodes_by_attribute('generator_aggr')
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
                 'type': [],
                 'p_nom_extendable': [],
                 'p_nom_min': [],
                 'p_nom_max': [],
                 'capital_cost': []
                 }

    bus = {'name': [], 'v_nom': [], 'x': [], 'y': []}

    load = {'name': [], 'bus': []}

    line = {'name': [],
            'bus0': [],
            'bus1': [],
            'type': [],
            'x': [],
            'r': [],
            's_nom': [],
            's_nom_min': [],
            's_max_pu': [],
            's_nom_extendable': [],
            'capital_cost': [],
            'length': []}

    transformer = {'name': [],
                   'bus0': [],
                   'bus1': [],
                   'type': [],
                   'model': [],
                   'x': [],
                   'r': [],
                   's_nom': [],
                   's_nom_extendable': [],
                   'capital_cost': [],
                   'tap_ratio': []}

    storage = {
        'name': [],
        'bus': [],
        'p_nom': [],
        'p_nom_extendable': [],
        'p_nom_min': [],
        'p_nom_max': [],
        'capital_cost': [],
        'max_hours': []}

    # create dataframe representing generators and associated buses
    for gen in generators:
        bus_name = '_'.join(['Bus', repr(gen)])
        generator['name'].append(repr(gen))
        generator['bus'].append(bus_name)
        generator['control'].append('PQ')
        generator['p_nom'].append(gen.nominal_capacity / 1e3)
        generator['type'].append('_'.join([gen.type, gen.subtype]))
        generator['p_nom_extendable'].append(False)
        generator['p_nom_min'].append(0)  # 0.3
        generator['p_nom_max'].append(0)
        generator['capital_cost'].append(0)

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
        s_nom = sqrt(3) * l['line'].type['I_max_th'] * \
                l['line'].type['U_n'] / 1e3
        line['s_nom'].append(s_nom)
        line['s_nom_min'].append(s_nom)
        line['s_max_pu'].append(0.6)
        line['s_nom_extendable'].append(True)
        line['capital_cost'].append(100)
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
            transformer['s_nom_extendable'].append(True)
            transformer['capital_cost'].append(100)
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

        generator['name'].append(repr(sto))
        generator['bus'].append(bus_name)
        generator['control'].append('PQ')
        generator['p_nom'].append(sto.nominal_power / 1e3)
        generator['type'].append('Storage')
        generator['p_nom_extendable'].append(True)
        generator['p_nom_min'].append(0) # 0.3
        generator['p_nom_max'].append(4.5)
        generator['capital_cost'].append(10)

        bus['name'].append(bus_name)
        bus['v_nom'].append(sto.grid.voltage_nom)
        bus['x'].append(sto.geom.x)
        bus['y'].append(sto.geom.y)

    # Add separate slack generator at MV station secondary side bus bar
    s_station = sum([_.type.S_nom for _ in mv_stations[0].transformers])
    generator['name'].append("Generator_slack")
    generator['bus'].append(bus1_name)
    generator['control'].append('PQ')
    generator['p_nom'].append(2*s_station)
    generator['type'].append('Slack generator')
    generator['p_nom_extendable'].append(False)
    generator['p_nom_min'].append(0)  # 0.3
    generator['p_nom_max'].append(0)
    generator['capital_cost'].append(0)

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
        generators.extend(lv_grid.graph.nodes_by_attribute('generator'))
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
                 'type': [],
                 'p_nom_extendable': [],
                 'p_nom_min': [],
                 'p_nom_max': [],
                 'capital_cost': []
                 }

    bus = {'name': [], 'v_nom': [], 'x': [], 'y': []}

    load = {'name': [], 'bus': []}

    line = {'name': [],
            'bus0': [],
            'bus1': [],
            'type': [],
            'x': [],
            'r': [],
            's_nom': [],
            's_nom_min': [],
            's_max_pu': [],
            's_nom_extendable': [],
            'capital_cost': [],
            'length': []}

    storage = {
        'name': [],
        'bus': [],
        'p_nom': [],
        'p_nom_extendable': [],
        'p_nom_min': [],
        'p_nom_max': [],
        'capital_cost': [],
        'max_hours': []}

    # create dictionary representing generators and associated buses
    for gen in generators:
        bus_name = '_'.join(['Bus', repr(gen)])
        generator['name'].append(repr(gen))
        generator['bus'].append(bus_name)
        generator['control'].append('PQ')
        generator['p_nom'].append(gen.nominal_capacity / 1e3)
        generator['type'].append('_'.join([gen.type, gen.subtype]))
        generator['p_nom_extendable'].append(False)
        generator['p_nom_min'].append(0)  # 0.3
        generator['p_nom_max'].append(0)
        generator['capital_cost'].append(0)

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
        s_nom = sqrt(3) * l['line'].type['I_max_th'] * \
                l['line'].type['U_n'] / 1e3
        line['s_nom'].append(s_nom)
        line['s_nom_min'].append(s_nom)
        line['s_max_pu'].append(0.6)
        line['s_nom_extendable'].append(True)
        line['capital_cost'].append(100)
        line['length'].append(l['line'].length)

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
    lv_load_timeseries_p = []

    # add MV grid loads
    if mode is 'mv' or mode is None:
        for load in network.mv_grid.graph.nodes_by_attribute('load'):
            mv_load_timeseries_p.append(load.pypsa_timeseries('p').rename(
                repr(load)).to_frame().loc[timesteps])

    # add LV grid's loads
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for load in lv_grid.graph.nodes_by_attribute('load'):
                for sector in list(load.consumption.keys()):
                    # for sector in list(list(load.consumption.keys())[0]):
                    # ToDo: remove consideration of only industrial sector
                    # now, if a load object has consumption in multiple sectors
                    # (like currently only industrial/retail) the consumption is
                    # implicitly assigned to the industrial sector when being
                    # exported to pypsa.
                    # ToDo: resolve this in the importer
                    if sector != 'retail':
                        # lv_load_timeseries_q.append(
                        #     load.pypsa_timeseries('q').rename(
                        #         repr(load)).to_frame().loc[timesteps])
                        lv_load_timeseries_p.append(
                            load.pypsa_timeseries('p').rename(
                                repr(load)).to_frame().loc[timesteps])

    load_df_p = pd.concat(mv_load_timeseries_p + lv_load_timeseries_p, axis=1)

    return load_df_p


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

    mv_gen_timeseries_p_min = []
    mv_gen_timeseries_p_max = []
    lv_gen_timeseries_p_min = []
    lv_gen_timeseries_p_max = []

    # MV generator timeseries
    if mode is 'mv' or mode is None:
        for gen in network.mv_grid.graph.nodes_by_attribute('generator') + \
                network.mv_grid.graph.nodes_by_attribute('generator_aggr'):
            mv_gen_timeseries_p_min.append(gen.pypsa_timeseries('p').rename(
                repr(gen)).to_frame().loc[timesteps] / gen.nominal_capacity)
            mv_gen_timeseries_p_max.append(gen.pypsa_timeseries('p').rename(
                repr(gen)).to_frame().loc[timesteps] / gen.nominal_capacity)

    # LV generator timeseries
    if mode is 'lv' or mode is None:
        for lv_grid in network.mv_grid.lv_grids:
            for gen in lv_grid.graph.nodes_by_attribute('generator'):
                lv_gen_timeseries_p_min.append(gen.pypsa_timeseries('p').rename(
                    repr(gen)).to_frame().loc[timesteps] / gen.nominal_capacity)
                lv_gen_timeseries_p_max.append(gen.pypsa_timeseries('p').rename(
                    repr(gen)).to_frame().loc[timesteps] / gen.nominal_capacity)

    # Slack time series
    lv_gen_timeseries_p_min.append(
        pd.Series([-1] * len(timesteps), index=timesteps).rename(
            "Generator_slack").to_frame())
    lv_gen_timeseries_p_max.append(
        pd.Series([1] * len(timesteps), index=timesteps).rename(
            "Generator_slack").to_frame())

    gen_df_p_max = pd.concat(
        mv_gen_timeseries_p_max + lv_gen_timeseries_p_max, axis=1)
    gen_df_p_min = pd.concat(
        mv_gen_timeseries_p_min + lv_gen_timeseries_p_min, axis=1)

    return gen_df_p_min, gen_df_p_max


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

    mv_storage_timeseries_p_min = []
    mv_storage_timeseries_p_max = []

    # MV storage time series
    if mode is 'mv' or mode is None:
        for storage in network.mv_grid.graph.nodes_by_attribute('storage'):
            mv_storage_timeseries_p_min.append(
                storage.timeseries.p.rename(repr(
                    storage)).to_frame().loc[timesteps])
            mv_storage_timeseries_p_max.append(
                storage.timeseries.p.rename(repr(
                    storage)).to_frame().loc[timesteps])

    storage_df_p_min = pd.concat(
        mv_storage_timeseries_p_min, axis=1)
    storage_df_p_max = pd.concat(
        mv_storage_timeseries_p_max, axis=1)

    return storage_df_p_min, storage_df_p_max


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
    """"""

    # check for sub-networks
    subgraphs = list(connected_component_subgraphs(pypsa_network.graph()))
    pypsa_network.determine_network_topology()

    if len(subgraphs) > 1 or len(pypsa_network.sub_networks) > 1:
        raise ValueError("The graph has isolated nodes or edges")

    # check consistency of topology and time series data
    generators_ts_p_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['p_min_pu'].columns.tolist())]
    generators_ts_q_missing = pypsa_network.generators.loc[
        ~pypsa_network.generators.index.isin(
            pypsa_network.generators_t['p_max_pu'].columns.tolist())]
    loads_ts_p_missing = pypsa_network.loads.loc[
        ~pypsa_network.loads.index.isin(
            pypsa_network.loads_t['p_set'].columns.tolist())]
    # loads_ts_q_missing = pypsa_network.loads.loc[
    #     ~pypsa_network.loads.index.isin(
    #         pypsa_network.loads_t['q_set'].columns.tolist())]
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

    # if not loads_ts_q_missing.empty:
    #     raise ValueError("Following loads have no `q_set` time series "
    #                      "{loads}".format(
    #         loads=loads_ts_q_missing))

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
