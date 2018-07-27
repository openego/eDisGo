import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from edisgo.grid.network import Results


def create_voltage_plots(result_object, directory, **kwargs):
    """
    Function to create some standard plots
    Parameters
    ----------
    result_object: :py:mod:`~/edisgo/grid/network.Results Object
        The eDisGo results object which should be plotted
    directory: :obj:`str`
        path to save the plots
    timeindex: :pandas:`pandas.DateTimeIndex`
        Datetime index which the histogram should be constructed from.
        Default: all indexes in the results
    plot_separate_timesteps: :obj:`boolean`
        If true then a separate histogram is generated for each timestep
        Default: True
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
    voltage = result_object.pfa_v_mag_pu.copy()
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

    plot_separate_timesteps = kwargs.get('plot_separate_timesteps', True)

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
        plot_title = "Voltage Histogram \nfrom {} to {}".format(str(timeindex.[0], timeindex[=1]))

        bins = np.arange(lowerlimit, upperlimit, binwidth)
        plt.figure(figsize=fig_size)
        voltage.plot(kind='hist', normed=True,
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