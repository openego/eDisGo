import os
import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from pypsa import Network as PyPSANetwork


def create_curtailment_characteristic(assigned_curtailment,
                                      generator_feedins,
                                      bus_voltages_before_curtailment,
                                      gens_fluct_info,
                                      directory, **kwargs):
    """
    Function to create some voltage histograms.
    Parameters
    ----------
    assigned_curtailment: :pandas:`pandas.DataFrame<dataframe>`
        The assigned curtailment in kW of the generators typically
        obtained from :py:mod:`edisgo.network.Results` object
        in the attribute
        :attr:`edisgo.network.Results.assigned_curtailment`.
        The columns names are the individual generators as
        `edisgo.grid.components.GeneratorFluctuating` objects
        and the index is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
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
         - 'a4portrait'
         - 'a4landscape'
         - 'a5portrait'
         - 'a5landscape'
         By default and in failing cases this would be set to 'a5landscape'.
    binwidth: :obj:`float`
        width of bins in per unit voltage,
        By default and in failing cases this would be set to 0.01.
    """
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

    os.makedirs(directory, exist_ok=True)

    alpha = 1 - transparency
    if alpha > 1:
        alpha = 1
    elif alpha < 0:
        alpha = 0

    normalization = kwargs.get('normalization_method', 'by_feedin')
    if normalization == 'by_feedin':
        by_feedin = True
        by_nominal_cap = False
    elif normalization == 'by_nominal_cap':
        by_feedin = False
        by_nominal_cap = True
    else:
        raise ValueError('Invalid input to normalization method')

    # process the gen info to get the bus names
    gens_fluct_info = gens_fluct_info.reset_index().set_index('gen_repr')
    # get only those generator that are present in assigned curtailment
    gens_in_assinged_curtail = list(assigned_curtailment.columns)
    if type(gens_in_assinged_curtail[0]) != str:
        gens_in_assinged_curtail = list(map(repr, gens_in_assinged_curtail))
    gens_fluct_info = gens_fluct_info.loc[gens_in_assinged_curtail, :]
    # get the buses from the repr
    fluct_buses = list('Bus_' + gens_fluct_info.index)

    timeindex = kwargs.get('timeindex', bus_voltages_before_curtailment.index)

    v = {}
    for n, i in enumerate(bus_voltages_before_curtailment.loc[timeindex, :].index):
        v[n] = bus_voltages_before_curtailment.loc[str(i), fluct_buses]

    c = {}
    for n, i in enumerate(assigned_curtailment.loc[timeindex, :].index):
        c[n] = assigned_curtailment.loc[i, gens_fluct_info.generator]
        c[n].index = list(map(str, c[n].index.values))
        if by_feedin:
            c[n] /= generator_feedins.iloc[n]
        elif by_nominal_cap:
            c[n] /= gens_fluct_info.nominal_capacity
        else:
            raise RuntimeError("incorrect normalization method provided")
        c[n].index = list(map(lambda x: 'Bus_' + str(x), c[n].index.values))

    if by_feedin:
        x_label = kwargs.get('xlabel', "Voltage [per unit]")
        y_label = kwargs.get('ylabel', "Curtailment [per unit] normalized by feedin")
    elif by_nominal_cap:
        x_label = kwargs.get('xlabel', "Voltage [per unit]")
        y_label = kwargs.get('ylabel', "Curtailment [per unit] normalized by installed capacity")

    for n, i in enumerate([(c[x], v[x]) for x in range(len(timeindex))]):
        plt.figure(figsize=fig_size)
        plot_title = "Curtailment Characteristic at {}".format(timeindex[n])
        pd.DataFrame({'voltage_pu': i[1],
                      'curtailment_pu': i[0]}).plot(kind='scatter',
                                                    x='voltage_pu',
                                                    y='curtailment_pu',
                                                    xlim=x_limits,
                                                    ylim=y_limits,
                                                    color=color,
                                                    alpha=alpha,
                                                    edgecolor=None,
                                                    grid=True)
        plt.minorticks_on()
        plt.axvline(1.0, color='black', linestyle='--')
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(os.path.join(directory,
                                 'curtailment_voltage_characterisitc_{}.{}'.format(
                                     timeindex[n].strftime('%Y%m%d%H%M'),
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
         - 'a4portrait'
         - 'a4landscape'
         - 'a5portrait'
         - 'a5landscape'
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
    pypsa_plot.buses = network.pypsa.buses.loc[network.pypsa.buses.v_nom >= 10]
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
