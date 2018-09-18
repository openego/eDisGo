import os
import pandas as pd
import numpy as np
import logging
from math import sqrt
from matplotlib import pyplot as plt
from pypsa import Network as PyPSANetwork

from edisgo.tools import tools


def create_curtailment_characteristic(curtailment, pypsa_network, timestep,
                                      directory, **kwargs):
    """
    Function to create some voltage histograms.

    Parameters
    ----------
    curtailment : :pandas:`pandas.DataFrame<dataframe>`
        Assigned curtailment in kW of all generators to be included in the
        plot. The column names are the generators representatives, index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Curtailment can be obtained from to each generator per curtailment target.
        The assigned curtailment in kW from of the generators typically
        obtained from :py:mod:`edisgo.network.Results` object
        in the attribute
        :attr:`edisgo.network.Results.assigned_curtailment`.
    pypsa_network : :pypsa:`pypsa.Network<network>`
    generator_feedins: :pandas:`pandas.DataFrame<dataframe>`
        The feedins in kW of every single generator typically
        obtained from :py:mod:`edisgo.grid.tools.generator_feedins`
        The columns names are the individual generators as
        `edisgo.grid.components.GeneratorFluctuating` and
        `edisgo.grid.components.Generator` objects
        and the index is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    bus_voltages_before_curtailment: :pandas:`pandas.DataFrame<dataframe>`
        The voltages in per unit at the buses before curtailment
        as in the :py:mod:`edisgo.network.pypsa` object
        from the attribute 'buses_t['v_mag_pu'].
        The columns names are the individual buses as
        :obj:`str` objects containing the bus IDs
        (including Generators as 'Bus_Generator...')
        and the index is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    gens_fluct_info: :pandas:`pandas.DataFrame<dataframe>`
        The information about all the fluctuating generators
        i.e. gen_repr, type, voltage_level, weather_cell_id and nominal_capacity
        as can be obtained from :py:mod:`edisgo.grid.tools.get_gen_info`
        with the 'fluctuating' switch set to 'True'.
        The columns names are information categories
        namely 'gen_repr', 'type', 'voltage_level',
        'nominal_capacity', 'weather_cell_id' and
        the index contains the
        `edisgo.grid.components.GeneratorFluctuating` objects.
    directory: :obj:`str`
        path to save the plots
    filetype: :obj:`str`
        filetype to save the file with, the allowed types
        are the same as those allowed from matplotlib
        Default: png
    timeindex: :pandas:`pandas.DatetimeIndex<datetimeindex>`
        Datetime index which the histogram should be constructed from.
        Default: all indexes in the results
    color: :obj:`str`
        color of plot in matplotlib standard color
    transparency: :obj:`float`
        transparency of the plot, a number from 0 to 1,
        where 0 is see through and 1 is opaque.
    xlabel: :obj:`str`
        label for x axis. Both by default and in failing cases this
        would be set to 'Normalized Frequency [per unit]'.
    ylabel: :obj:`str`
        label for y axis. Both by default and in failing cases this
        would be set to 'Voltage [per unit]'.
    xlim: :obj:`tuple`
        tuple of limits of x axis (left_limit,right_limit)
    ylim: :obj:`tuple`
        tuple of limits of y axis (left_limit,right_limit)
    figsize: :obj:`str` or :obj:`tuple`
        size of the figure in inches or a string with the following options:
         * 'a4portrait'
         * 'a4landscape'
         * 'a5portrait'
         * 'a5landscape'

         By default and in failing cases this would be set to 'a5landscape'.
    binwidth: :obj:`float`
        width of bins in per unit voltage,
        By default and in failing cases this would be set to 0.01.

    """

    # get voltages
    gens_buses = list(map(lambda _: 'Bus_{}'.format(_), curtailment.columns))
    voltages = pypsa_network.buses_t.v_mag_pu.loc[timestep, gens_buses]
    voltages = pd.Series(voltages.values, index=curtailment.columns)

    # get feed-ins
    feedins = pypsa_network.generators_t.p.loc[
                  timestep, curtailment.columns] * 1e3

    # relative curtailment
    rel_curtailment = curtailment.loc[timestep, :] / feedins

    plot_df = pd.DataFrame({'voltage_pu': voltages,
                            'curtailment_pu': rel_curtailment})

    # configure plot
    x_limits = kwargs.get('xlim', None)
    y_limits = kwargs.get('ylim', None)
    color = kwargs.get('color', 'blue')
    transparency = kwargs.get('transparency', 0)
    filetype = kwargs.get('filetype', 'png')
    fig_size = kwargs.get('figsize', 'a5landscape')
    standard_sizes = {'a4portrait': (8.27, 11.69),
                      'a4landscape': (11.69, 8.27),
                      'a5portrait': (5.8, 8.3),
                      'a5landscape': (8.3, 5.8)}
    try:
        fig_size = standard_sizes[fig_size]
    except:
        message = "Unknown size {}. Using default a5landscape".format(fig_size)
        logging.warning(message)
        fig_size = standard_sizes['a5landscape']

    alpha = 1 - transparency
    if alpha > 1:
        alpha = 1
    elif alpha < 0:
        alpha = 0

    x_label = kwargs.get('xlabel', "Voltage in p.u.")
    y_label = kwargs.get('ylabel', "Curtailment normalized by feedin in kW/kW")

    # plot
    plt.figure(figsize=fig_size)
    plot_title = "Curtailment Characteristic at {}".format(timestep)
    plot_df.plot(kind='scatter', x='voltage_pu', y='curtailment_pu',
                 xlim=x_limits, ylim=y_limits,
                 color=color, alpha=alpha, edgecolor=None, grid=True)
    plt.minorticks_on()
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if kwargs.get('voltage_threshold', None):
        plt.axvline(1.0, color='black', linestyle='--')

    # save
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory,
                             'curtailment_characteristic_{}.{}'.format(
                                 timestep.strftime('%Y%m%d%H%M'),
                                 filetype)))
    plt.close('all')


def create_voltage_plots(voltage_data, directory, **kwargs):
    """
    Function to create some voltage histograms.

    Parameters
    ----------
    voltage_data: either :pandas:`pandas.DataFrame<dataframe>` or :py:mod:`~/edisgo/grid/network.Results` Object
        The voltage data to be plotted. If this is a
        :pandas:`pandas.DataFrame<dataframe>`, the columns are to be
        String labels of the node names with IDs, else if its an
        :py:mod:`~/edisgo/grid/network.Results` the function will automatically
        get the dataframe in pfa_v_mag_pu.
    directory: :obj:`str`
        path to save the plots
    filetype: :obj:`str`
        filetype to save the file with, the allowed types
        are the same as those allowed from matplotlib
        Default: png
    timeindex: :pandas:`pandas.DatetimeIndex<datetimeindex>`
        Datetime index which the histogram should be constructed from.
        Default: all indexes in the results
    plot_separate_timesteps: :obj:`boolean`
        If true then a separate histogram is generated for each timestep
        Default: False
    color: :obj:`str`
        color of plot in matplotlib standard color
    transparency: :obj:`float`
        transparency of the plot, a number from 0 to 1,
        where 0 is see through and 1 is opaque.
    xlabel: :obj:`str`
        label for x axis. Both by default and in failing cases this
        would be set to 'Normalized Frequency [per unit]'.
    ylabel: :obj:`str`
        label for y axis. Both by default and in failing cases this
        would be set to 'Voltage [per unit]'.
    xlim: :obj:`tuple`
        tuple of limits of x axis (left_limit,right_limit)
    ylim: :obj:`tuple`
        tuple of limits of y axis (left_limit,right_limit)
    figsize: :obj:`str` or :obj:`tuple`
        size of the figure in inches or a string with the following options:
         * 'a4portrait'
         * 'a4landscape'
         * 'a5portrait'
         * 'a5landscape'

         By default and in failing cases this would be set to 'a5landscape'.
    binwidth: :obj:`float`
        width of bins in per unit voltage,
        By default and in failing cases this would be set to 0.01.

    """
    voltage = voltage_data.copy()
    x_label = kwargs.get('xlabel', "Voltage [per unit]")
    y_label = kwargs.get('ylabel', "Normalized Frequency [per unit]")
    x_limits = kwargs.get('xlim', (0.9, 1.1))
    y_limits = kwargs.get('ylim', (0, 60))
    color = kwargs.get('color', None)
    transparency = kwargs.get('transparency', 0)
    binwidth = kwargs.get('binwidth', 0.01)
    lowerlimit = x_limits[0] - binwidth / 2
    upperlimit = x_limits[1] + binwidth / 2
    filetype = kwargs.get('filetype', 'png')
    fig_size = kwargs.get('figsize', 'a5landscape')
    standard_sizes = {'a4portrait': (8.27, 11.69),
                      'a4landscape': (11.69, 8.27),
                      'a5portrait': (5.8, 8.3),
                      'a5landscape': (8.3, 5.8)}
    try:
        fig_size = standard_sizes[fig_size]
    except:
        message = "Unknown size {}. using default a5landscape".format(fig_size)
        logging.warning(message)
        fig_size = standard_sizes['a5landscape']

    alpha = 1 - transparency
    if alpha > 1:
        alpha = 1
    elif alpha < 0:
        alpha = 0

    timeindex = kwargs.get('timeindex', voltage.index)

    plot_separate_timesteps = kwargs.get('plot_separate_timesteps', False)

    os.makedirs(directory, exist_ok=True)

    if plot_separate_timesteps:
        for timestamp in timeindex:
            plot_title = "Voltage Histogram at {}".format(str(timestamp))

            bins = np.arange(lowerlimit, upperlimit, binwidth)
            plt.figure(figsize=fig_size)
            voltage.loc[str(timestamp), :].plot(kind='hist', normed=True,
                                                color=color,
                                                alpha=alpha,
                                                bins=bins,
                                                xlim=x_limits,
                                                ylim=y_limits,
                                                grid=True)
            plt.minorticks_on()
            plt.axvline(1.0, color='black', linestyle='--')
            plt.axvline(voltage.loc[str(timestamp), :].mean(),
                        color='green', linestyle='--')
            plt.title(plot_title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(directory,
                                     'voltage_histogram_{}.{}'.format(
                                         timestamp.strftime('%Y%m%d%H%M'),
                                         filetype)))
            plt.close('all')
    else:
        plot_title = "Voltage Histogram \nfrom {} to {}".format(str(timeindex[0]), str(timeindex[-1]))

        bins = np.arange(lowerlimit, upperlimit, binwidth)
        plt.figure(figsize=fig_size)
        voltage.plot(kind='hist', normed=True,
                     color=color,
                     alpha=alpha,
                     bins=bins,
                     xlim=x_limits,
                     ylim=y_limits,
                     grid=True,
                     legend=False)
        plt.minorticks_on()
        plt.axvline(1.0, color='black', linestyle='--')
        # plt.axvline(voltage.mean(),
        #             color='green', linestyle='--')
        plt.legend()
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(os.path.join(directory,
                                 'voltage_histogram_all.{}'.format(filetype)))
        plt.close('all')


def line_loading(pypsa_network, configs, line_load, timestep,
                 filename=None, arrows=True):
    """
    Plot line loading as color on lines.

    Displays line loading relative to nominal capacity.

    Parameters
    ----------
    pypsa_network : :pypsa:`pypsa.Network<network>`
    timestep : :pandas:`pandas.Timestamp<timestamp>`
        Time step to plot analysis results for.
    filename : :obj:`str`
        Filename to save plot under. If not provided, figure is shown directly.
        Default: None.
    arrows: :obj:`Boolean`
        If True draws arrows on lines in the direction of the power flow.
        Default: True.

    """
    # create pypsa network only containing MV buses and lines
    pypsa_plot = PyPSANetwork()
    pypsa_plot.buses = pypsa_network.buses.loc[pypsa_network.buses.v_nom >= 10]
    pypsa_plot.lines = pypsa_network.lines.loc[pypsa_network.lines.v_nom >= 10]

    # calculate relative line loading
    # get load factor
    residual_load = tools.get_residual_load_from_pypsa_network(pypsa_network)
    case = residual_load.apply(
            lambda _: 'feedin_case' if _ < 0 else 'load_case').loc[timestep]
    load_factor = float(configs['grid_expansion_load_factors'][
        'mv_{}_line'.format(case)])
    # get allowed line load
    i_line_allowed = pypsa_plot.lines.s_nom.divide(
        pypsa_plot.lines.v_nom) / sqrt(3) * 1e3 * load_factor
    # get line load from pf
    i_line_pfa = line_load.loc[timestep,
                               pypsa_plot.lines.index]
    loading = i_line_pfa.divide(i_line_allowed)

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
    cmap = plt.cm.jet
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
            if pypsa_network.lines_t.p0.loc[timestep,
                                            loading.index[i]] > 0:
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


def storage_size(mv_grid, pypsa_network, filename=None, lopf=True):
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

    # create pypsa network only containing MV buses and lines
    lines = [_['line'] for _ in mv_grid.graph.lines()]
    pypsa_plot = PyPSANetwork()
    pypsa_plot.buses = pypsa_network.buses.loc[
        (pypsa_network.buses.v_nom >= 10)]
    pypsa_plot.lines = pypsa_network.lines.loc[[repr(_) for _ in lines]]

    # bus colors and sizes
    bus_sizes = {}
    bus_colors = {}
    colors_dict = {'BranchTee': 'b',
                   'GeneratorFluctuating': 'g',
                   'Generator': 'k',
                   'LVStation': 'c',
                   'MVStation': 'orange',
                   'Storage': 'r',
                   'DisconnectingPoint': '0.75',
                   'else': 'y'}
    sizes_dict = {'BranchTee': 10,
                  'GeneratorFluctuating': 10,
                  'Generator': 10,
                  'LVStation': 50,
                  'MVStation': 120,
                  'Storage': 1000,
                  'DisconnectingPoint': 50,
                  'else': 10}

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
        elif 'storage' in name:
            tmp = name.split('_')
            storage_repr = '_'.join(tmp[1:])
            if lopf:
                size = pypsa_network.generators.loc[
                           storage_repr, 'p_nom_opt'] * sizes_dict['Storage']
                return colors_dict['Storage'], size
            else:
                size = pypsa_network.storage_units.loc[
                           storage_repr, 'p_nom'] * sizes_dict['Storage'] + 200
                return colors_dict['Storage'], size
        else:
            return colors_dict['else'], sizes_dict['else']

    for bus in pypsa_plot.buses.index:
        bus_colors[bus], bus_sizes[bus] = get_color_and_size(bus)

    # plot
    ll = pypsa_plot.plot(title="Line loading", line_widths=0.55,
                         branch_components=['Line'], basemap=True,
                         bus_sizes=bus_sizes, bus_colors=bus_colors)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, papertype='a4')
        plt.close()
