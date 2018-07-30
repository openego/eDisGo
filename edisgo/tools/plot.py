from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pypsa import Network as PyPSANetwork


def line_loading(network, timestep, filename=None, arrows=True):
    """
    Plot line loading as color on lines

    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    timestep : :pandas:`pandas.Timestamp<timestamp>`
        Time step to plot analysis results for.
    filename : :obj:`str`
        Filename to save plot under. If not provided, figure is shown directly.
        Default: None.
    arrows: :obj:`Boolean`
        If True draws arrows on lines in the direction of the power flow.
        Default: True.

    """
    # define color map
    cmap = plt.cm.jet

    # calculate relative line loading
    case = network.timeseries.timesteps_load_feedin_case.loc[timestep, 'case']
    load_factor = network.config['grid_expansion_load_factors'][
        'mv_{}_line'.format(case)]

    lines = [_['line'] for _ in network.mv_grid.graph.lines()]
    i_line_allowed = pd.DataFrame(
        {repr(l): [l.type['I_max_th'] * l.quantity * load_factor]
         for l in lines},
        index=[timestep])
    i_line_pfa = network.results.i_res.loc[[timestep],
                                           [repr(l) for l in lines]]
    loading = (i_line_pfa / i_line_allowed).iloc[0]

    # create pypsa network only containing MV buses and lines
    pypsa_plot = PyPSANetwork()
    pypsa_plot.buses = network.pypsa.buses.loc[network.pypsa.buses.v_nom == 20]
    pypsa_plot.lines = network.pypsa.lines.loc[[repr(_) for _ in lines]]

    # bus colors and sizes
    bus_sizes = {}
    bus_colors = {}
    colors_dict = {'BranchTee': 'b',
                   'GeneratorFluctuating': 'g',
                   'Generator': 'k',
                   'LVStation': 'c',
                   'MVStation': 'r',
                   'Storage': 'y',
                   'DisconnectingPoint': '0.75',
                   'else': 'orange'}
    sizes_dict = {'BranchTee': 10,
                  'GeneratorFluctuating': 100,
                  'Generator': 100,
                  'LVStation': 50,
                  'MVStation': 120,
                  'Storage': 100,
                  'DisconnectingPoint': 50,
                  'else': 200}

    def get_color_and_size(name):
        if 'BranchTee' in name:
            return colors_dict['BranchTee'], sizes_dict['BranchTee']
        elif 'LVStation' in name:
            return colors_dict['LVStation'], sizes_dict['LVStation']
        elif 'GeneratorFluctuating' in name:
            return (colors_dict['GeneratorFluctuating'],
                    sizes_dict['GeneratorFluctuating'])
        elif 'Generator' in name:
            return colors_dict['Generator'], sizes_dict['Generator']
        elif 'DisconnectingPoint' in name:
            return (colors_dict['DisconnectingPoint'],
                    sizes_dict['DisconnectingPoint'])
        elif 'MVStation' in name:
            return colors_dict['MVStation'], sizes_dict['MVStation']
        elif 'Storage' in name:
            return colors_dict['Storage'], sizes_dict['Storage']
        else:
            return colors_dict['else'], sizes_dict['else']

    for bus in pypsa_plot.buses.index:
        bus_colors[bus], bus_sizes[bus] = get_color_and_size(bus)

    # plot
    ll = pypsa_plot.plot(line_colors=loading, line_cmap=cmap,
                         title="Line loading", line_widths=0.55,
                         branch_components=['Line'], basemap=True,
                         bus_sizes=bus_sizes, bus_colors=bus_colors)

    # color bar
    v = np.linspace(min(loading), max(loading), 101)
    boundaries = [min(loading), max(loading)]
    cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10])
    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    cb.set_label('Line loading in %')

    # draw arrows on lines
    if arrows:
        ax = plt.axes()
        path = ll[1].get_segments()
        colors = cmap(ll[1].get_array() / 100)
        for i in range(len(path)):
            if network.pypsa.lines_t.p0.loc[timestep, repr(lines[i])] > 0:
                arrowprops = dict(arrowstyle="->", color=colors[i])
            else:
                arrowprops = dict(arrowstyle="<-", color=colors[i])
            ax.annotate(
                "",
                xy=abs(
                    (path[i][0] - path[i][1]) * 0.51 - path[i][0]),
                xytext=abs(
                    (path[i][0] - path[i][1]) * 0.49 - path[i][0]),
                arrowprops=arrowprops,
                size=10)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()