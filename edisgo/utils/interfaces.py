import pandas as pd
from math import pi, sqrt, floor


def mv_to_pypsa(network):
    """Translate grid topology representation to PyPSA format"""

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
        bus_name = '_'.join(['Bus', repr(load)])
        load['name'].append(repr(lo))
        load['bus'].append(bus_name)

        bus['name'].append(bus_name)
        bus['v_nom'].append(lo.grid.voltage_nom)

    # create dataframe for lines
    for l in lines:
        line['name'].append(repr(l['line']))

        if l['adj_nodes'][0] in lv_stations:
            line['bus0'].append(
                '_'.join(['Bus', 'primary', repr(l['adj_nodes'][0])]))
        else:
            line['bus0'].append('_'.join(['Bus', repr(l['adj_nodes'][0])]))

        if l['adj_nodes'][0] in lv_stations:
            line['bus1'].append(
                '_'.join(['Bus', 'primary', repr(l['adj_nodes'][1])]))
        else:
            line['bus1'].append('_'.join(['Bus', repr(l['adj_nodes'][1])]))

        line['type'].append("")
        line['x'].append(l['line'].type['L'] * omega)
        line['r'].append(l['line'].type['R'])
        line['s_nom'].append(
            sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
        line['length'].append(l['line'].length)

    # create dataframe for LV stations incl. primary/secondary side bus
    for lv_st in lv_stations:
        transformer_count = 1
        # add primary side bus (bus0)
        bus0_name = '_'.join(['Bus', 'primary', repr(lv_st)])
        bus['name'].append(bus0_name)
        bus['v_nom'].append(lv_st.grid.voltage_nom)

        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', 'secondary', repr(lv_st)])
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

    # create dataframe for MV stations incl. primary/secondary side bus
    for mv_st in mv_stations:
        mv_transformer_count = 1
        # add primary side bus (bus0)
        bus0_name = '_'.join(['Bus', 'primary', repr(mv_st)])
        bus['name'].append(bus0_name)
        # TODO: replace hard-coded primary side nominal voltage
        bus['v_nom'].append(110)

        # add secondary side bus (bus1)
        bus1_name = '_'.join(['Bus', 'secondary', repr(mv_st)])
        bus['name'].append(bus1_name)
        bus['v_nom'].append(mv_st.transformers[0].voltage_op)

        for mv_tr in mv_st.transformers:
            if mv_transformer_count <= floor(len(mv_st.transformers) / 2):
                transformer['name'].append(
                    '_'.join([repr(mv_st),
                              'transformer',
                              str(mv_transformer_count)]))
                transformer['bus0'].append(bus0_name)
                transformer['bus1'].append(bus1_name)
                transformer['type'].append("")
                transformer['model'].append('pi')
                # TODO: once Dingo data come with correct MV transformer params replace lines below
                transformer['r'].append(0.1)
                transformer['x'].append(0.1)
                # transformer['r'].append(mv_tr.type.r)
                # transformer['x'].append(mv_tr.type.x)
                transformer['s_nom'].append(mv_tr.type.s)
                # TODO: MV transformers currently come with s_nom in MVA. Take commented line below once this is changend in Dingo
                # transformer['s_nom'].append(mv_tr.type.s / 1e3)
                # TODO: discuss with @jochenbuehler if we need a tap changer here
                transformer['tap_ratio'].append(1)

                mv_transformer_count += 1

    components = {
        'Generator': pd.DataFrame(generator).set_index('name'),
        'Bus': pd.DataFrame(bus).set_index('name'),
        'Load': pd.DataFrame(load).set_index('name'),
        'Line': pd.DataFrame(line).set_index('name'),
        'Transformer': pd.DataFrame(transformer).set_index('name')}

    return components


def attach_aggregated_lv_components():
    """
    This function aggregates LV load and generation at LV stations

    Use this function if you aim for MV calculation only

    Returns
    -------

    """
    pass


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
