import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from edisgo.grid.network import Results
from edisgo.grid.tools import get_gen_info, generator_feedins, generator_reactive_powers



# def create_curtailment_characteristic(edisgo_object,
#                                       assigned_curtailment=None,
#                                       generator_info=None,
#                                       generator_feedin=None,
#                                       generator_voltages_before_curtailment,
#                                       path, **kwargs):
#     """
#     Function to create some voltage histograms.
#     Parameters
#     ----------
#     result_object: :py:mod:`~/edisgo/grid/network.Results Object
#         The eDisGo results object which should be plotted
#     directory: :obj:`str`
#         path to save the plots
#     timeindex: :pandas:`pandas.DateTimeIndex`
#         Datetime index which the histogram should be constructed from.
#         Default: all indexes in the results
#     plot_separate_timesteps: :obj:`boolean`
#         If true then a separate histogram is generated for each timestep
#         Default: True
#     color: :obj:`str`
#         color of plot in matplotlib standard color
#     transparency: :obj:`float`
#         transparency of the plot, a number from 0 to 1,
#         where 0 is see through and 1 is opaque.
#     xlabel: :obj:`str`
#         label for x axis. Both by default and in failing cases this
#         would be set to 'Normalized Frequency [per unit]'.
#     ylabel: :obj:`str`
#         label for y axis. Both by default and in failing cases this
#         would be set to 'Voltage [per unit]'.
#     xlim: :obj:`tuple`
#         tuple of limits of x axis (left_limit,right_limit)
#     ylim: :obj:`tuple`
#         tuple of limits of y axis (left_limit,right_limit)
#     figsize: :obj:`str` or :obj:`tuple`
#         size of the figure in inches or a string with the following options:
#          - 'a4portrait'
#          - 'a4landscape'
#          - 'a5portrait'
#          - 'a5landscape'
#          By default and in failing cases this would be set to 'a5landscape'.
#     binwidth: :obj:`float`
#         width of bins in per unit voltage,
#         By default and in failing cases this would be set to 0.01.
#     """
#
#     x_label = kwargs.get('xlabel', "Voltage [per unit]")
#     y_label = kwargs.get('ylabel', "Curtailment [per unit]")
#     x_limits = kwargs.get('xlim', (0.9, 1.1))
#     y_limits = kwargs.get('ylim', (0, 60))
#     color = kwargs.get('color', None)
#     transparency = kwargs.get('transparency', 0)
#     fig_size = kwargs.get('figsize', 'a5landscape')
#     standard_sizes = {'a4portrait': (8.27, 11.69),
#                       'a4landscape': (11.69, 8.27),
#                       'a5portrait': (5.8, 8.3),
#                       'a5landscape': (8.3, 5.8)}
#     try:
#         fig_size = standard_sizes[fig_size]
#     except:
#         message = "Unknown size {}. using default a5landscape".format(fig_size)
#         logging.warning(message)
#         fig_size = standard_sizes['a5landscape']
#
#     alpha = 1 - transparency
#     if alpha > 1:
#         alpha = 1
#     elif alpha < 0:
#         alpha = 0
#
#
#
#     generator_info = get_gen_info(edisgo_object.network)
#     gens_fluct = generator_info[(generator_info.type == 'solar') | (generator_info.type == 'wind')]
#     gens_fluct = gens_fluct.reset_index().set_index('gen_repr')
#     fluct_buses = list('Bus_' + gens_fluct.index)
#
#     # get the assigned curtailment
#     assigned_curtailment = edisgo_object.network.results.assigned_curtailment
#
#     # this needs to be a separate dataframe with voltages
#     generator_voltages_before_curtailment = edisgo_object.network.results.pfa_v_mag_pu
#
#     timeindex = kwargs.get('timeindex', generator_voltages_before_curtailment.index)
#
#     v = {}
#     for n, i in enumerate(generator_voltages_before_curtailment.loc[timeindex,:].index):
#         v[n] = generator_voltages_before_curtailment.loc[str(i), fluct_buses]
#
#     c = {}
#     for n, i in enumerate(assigned_curtailment.loc[timeindex,:].index):
#         c[n] = assigned_curtailment.loc[str(i), gens_fluct.generator]
#         c[n].index = list(map(str, c[n].index.values))
#         c[n] /= gens_fluct.nominal_capacity
#         c[n].index = list(map(lambda x: 'Bus_' + str(x), c[n].index.values))
#
#     for timestamp in timeindex:
#         plot_title = "Curtailment characteristic at {}".format(str(timestamp))
#
#         plt.figure(figsize=fig_size)
#         voltage.loc[str(timestamp), :].plot(kind='hist', normed=True,
#                                             color=color,
#                                             alpha=alpha,
#                                             bins=bins,
#                                             xlim=x_limits,
#                                             ylim=y_limits,
#                                             grid=True)
#         plt.minorticks_on()
#         plt.axvline(1.0, color='black', linestyle='--')
#         plt.axvline(voltage.loc[str(timestamp), :].mean(),
#                     color='green', linestyle='--')
#         plt.title(plot_title)
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.savefig(os.path.join(directory,
#                                  'curtailment_voltage_characterisitc_{}.svgz'.format(
#                                      timestamp.strftime('%Y%m%d%H%M'))))
#
#     plt.figure(figsize=(8, 14))
#     for n, i in enumerate([(c[x], v[x]) for x in range(4)]):
#         pd.DataFrame({'voltage_pu': i[1],
#                       'curtailment_pu': i[0]}).plot(kind='scatter',
#                                                     x='voltage_pu',
#                                                     y='curtailment_pu',
#                                                     alpha=0.3,
#                                                     edgecolor=None,
#                                                     grid=True)
#         plt.savefig(.format(n))
#
#
#     return NotImplementedError()







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
                                     'voltage_histogram_{}.svgz'.format(
                                         timestamp.strftime('%Y%m%d%H%M'))))
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
                                 'voltage_histogram_all.svgz'))