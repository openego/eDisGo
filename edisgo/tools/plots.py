import os

import pandas as pd
import numpy as np

import logging

from matplotlib import pyplot as plt
from edisgo.grid.network import Results


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
        and the index is a :pandas:`pandas.DateTimeIndex`.
    generator_feedins: :pandas:`pandas.DataFrame<dataframe>`
        The feedins in kW of every single generator typically
        obtained from :py:mod:`edisgo.grid.tools.generator_feedins`
        The columns names are the individual generators as
        `edisgo.grid.components.GeneratorFluctuating` and
        `edisgo.grid.components.Generator` objects
        and the index is a :pandas:`pandas.DateTimeIndex`.
    bus_voltages_before_curtailment: :pandas:`pandas.DataFrame<dataframe>`
        The voltages in per unit at the buses before curtailment
        as in the :py:mod:`edisgo.network.pypsa` object
        from the attribute 'buses_t['v_mag_pu'].
        The columns names are the individual buses as
        :obj:`str` objects containing the bus IDs
        (including Generators as 'Bus_Generator...')
        and the index is a :pandas:`pandas.DateTimeIndex`.
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
    timeindex: :pandas:`pandas.DateTimeIndex`
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
        message = "Unknown size {}. using default a5landscape".format(fig_size)
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
    timeindex: :pandas:`pandas.DateTimeIndex`
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
    if type(voltage_data) == Results:
        voltage = voltage_data.pfa_v_mag_pu.copy()
    elif type(voltage_data) == pd.DataFrame:
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
