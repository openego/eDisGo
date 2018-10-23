import os
import pandas as pd
import numpy as np
import logging
from math import sqrt
from matplotlib import pyplot as plt
from pypsa import Network as PyPSANetwork
from egoio.tools.db import connection
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import shape
from pyproj import Proj, transform

from edisgo.tools import tools

if not 'READTHEDOCS' in os.environ:
    from egoio.db_tables.grid import EgoDpMvGriddistrict
    from egoio.db_tables.model_draft import EgoGridMvGriddistrict
    geopandas = True
    try:
        import geopandas as gpd
    except:
        geopandas = False
    contextily = True
    try:
        import contextily as ctx
    except:
        contextily = False


def histogram(data, **kwargs):
    """
    Function to create histogram, e.g. for voltages or currents.

    Parameters
    ----------
    data : :pandas:`pandas.DataFrame<dataframe>`
        Data to be plotted, e.g. voltage or current (`v_res` or `i_res` from
        :class:`edisgo.grid.network.Results`). Index of the dataframe must be
        a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    timeindex : :pandas:`pandas.Timestamp<timestamp>` or None, optional
        Specifies time step histogram is plotted for. If timeindex is None all
        time steps provided in dataframe are used. Default: None.
    directory : :obj:`str` or None, optional
        Path to directory the plot is saved to. Is created if it does not
        exist. Default: None.
    filename : :obj:`str` or None, optional
        Filename the plot is saved as. File format is specified by ending. If
        filename is None, the plot is shown. Default: None.
    color : :obj:`str` or None, optional
        Color used in plot. If None it defaults to blue. Default: None.
    alpha : :obj:`float`, optional
        Transparency of the plot. Must be a number between 0 and 1,
        where 0 is see through and 1 is opaque. Default: 1.
    title : :obj:`str` or None, optional
        Plot title. Default: None.
    x_label : :obj:`str`, optional
        Label for x-axis. Default: "".
    y_label : :obj:`str`, optional
        Label for y-axis. Default: "".
    normed : :obj:`bool`, optional
        Defines if histogram is normed. Default: False.
    x_limits : :obj:`tuple` or None, optional
        Tuple with x-axis limits. First entry is the minimum and second entry
        the maximum value. Default: None.
    y_limits : :obj:`tuple` or None, optional
        Tuple with y-axis limits. First entry is the minimum and second entry
        the maximum value. Default: None.
    fig_size : :obj:`str` or :obj:`tuple`, optional
        Size of the figure in inches or a string with the following options:
         * 'a4portrait'
         * 'a4landscape'
         * 'a5portrait'
         * 'a5landscape'

         Default: 'a5landscape'.
    binwidth : :obj:`float`
        Width of bins. Default: None.

    """
    timeindex = kwargs.get('timeindex', None)
    directory = kwargs.get('directory', None)
    filename = kwargs.get('filename', None)
    title = kwargs.get('title', "")
    x_label = kwargs.get('x_label', "")
    y_label = kwargs.get('y_label', "")

    color = kwargs.get('color', None)
    alpha = kwargs.get('alpha', 1)
    normed = kwargs.get('normed', False)

    x_limits = kwargs.get('x_limits', None)
    y_limits = kwargs.get('y_limits', None)
    binwidth = kwargs.get('binwidth', None)

    fig_size = kwargs.get('fig_size', 'a5landscape')
    standard_sizes = {'a4portrait': (8.27, 11.69),
                      'a4landscape': (11.69, 8.27),
                      'a5portrait': (5.8, 8.3),
                      'a5landscape': (8.3, 5.8)}
    try:
        fig_size = standard_sizes[fig_size]
    except:
        fig_size = standard_sizes['a5landscape']

    if timeindex is not None:
        plot_data = data.loc[timeindex, :]
    else:
        plot_data = data.T.stack()

    if binwidth is not None:
        if x_limits is not None:
            lowerlimit = x_limits[0] - binwidth / 2
            upperlimit = x_limits[1] + binwidth / 2
        else:
            lowerlimit = plot_data.min() - binwidth / 2
            upperlimit = plot_data.max() + binwidth / 2
        bins = np.arange(lowerlimit, upperlimit, binwidth)
    else:
        bins = 10

    plt.figure(figsize=fig_size)
    ax = plot_data.hist(
        normed=normed, color=color, alpha=alpha, bins=bins, grid=True)
    plt.minorticks_on()

    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    if title is not None:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if filename is None:
        plt.show()
    else:
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, filename)
        plt.savefig(filename)
        plt.close()


def add_basemap(ax, zoom=12):
    """
    Adds map to a plot.

    """
    url = ctx.sources.ST_TONER_LITE
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                            zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))


def get_grid_district_polygon(config, subst_id=None, projection=4326):
    """
    Get MV grid district polygon from oedb for plotting.

    """

    # make DB session
    conn = connection(section=config['db_connection']['section'])
    Session = sessionmaker(bind=conn)
    session = Session()

    # get polygon from versioned schema
    if config['data_source']['oedb_data_source'] == 'versioned':

        version = config['versioned']['version']
        query = session.query(EgoDpMvGriddistrict.subst_id,
                              EgoDpMvGriddistrict.geom)
        Regions = [(subst_id, shape.to_shape(geom)) for subst_id, geom in
                   query.filter(EgoDpMvGriddistrict.version == version,
                                EgoDpMvGriddistrict.subst_id == subst_id).all()
                   ]

    # get polygon from model_draft
    else:
        query = session.query(EgoGridMvGriddistrict.subst_id,
                              EgoGridMvGriddistrict.geom)
        Regions = [(subst_id, shape.to_shape(geom)) for subst_id, geom in
                   query.filter(EgoGridMvGriddistrict.subst_id.in_(
                       subst_id)).all()]

    crs = {'init': 'epsg:3035'}
    region = gpd.GeoDataFrame(
        Regions, columns=['subst_id', 'geometry'], crs=crs)
    region = region.to_crs(epsg=projection)
    return region


def mv_grid_topology(pypsa_network, configs, timestep=None,
                     line_color=None, node_color=None,
                     line_load=None, grid_expansion_costs=None,
                     filename=None, arrows=False,
                     grid_district_geom=True, background_map=True,
                     voltage=None, limits_cb_lines=None, limits_cb_nodes=None,
                     xlim=None, ylim=None, lines_cmap='inferno_r',
                     title=''):
    """
    Plot line loading as color on lines.

    Displays line loading relative to nominal capacity.

    Parameters
    ----------
    pypsa_network : :pypsa:`pypsa.Network<network>`
    configs : :obj:`dict`
        Dictionary with used configurations from config files. See
        :class:`~.grid.network.Config` for more information.
    timestep : :pandas:`pandas.Timestamp<timestamp>`
        Time step to plot analysis results for. If `timestep` is None maximum
        line load and if given, maximum voltage deviation, is used. In that
        case arrows cannot be drawn. Default: None.
    line_color : :obj:`str`
        Defines whereby to choose line colors (and implicitly size). Possible
        options are:

        * 'loading'
          Line color is set according to loading of the line. Loading of MV
          lines must be provided by parameter `line_load`.
        * 'expansion_costs'
          Line color is set according to investment costs of the line. This
          option also effects node colors and sizes by plotting investment in
          stations and setting `node_color` to 'storage_integration' in order
          to plot storage size of integrated storages. Grid expansion costs
          must be provided by parameter `grid_expansion_costs`.
        * None (default)
          Lines are plotted in black. Is also the fallback option in case of
          wrong input.

    node_color : :obj:`str`
        Defines whereby to choose node colors (and implicitly size). Possible
        options are:

        * 'technology'
          Node color as well as size is set according to type of node
          (generator, MV station, etc.).
        * 'voltage'
          Node color is set according to voltage deviation from 1 p.u..
          Voltages of nodes in MV grid must be provided by parameter `voltage`.
        * 'storage_integration'
          Only storages are plotted. Size of node corresponds to size of
          storage.
        * None (default)
          Nodes are not plotted. Is also the fallback option in case of wrong
          input.

    line_load : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with current results from power flow analysis in A. Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<datetimeindex>`,
        columns are the line representatives.
    grid_expansion_costs : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with grid expansion costs in kEUR. See `grid_expansion_costs`
        in :class:`~.grid.network.Results` for more information.
    filename : :obj:`str`
        Filename to save plot under. If not provided, figure is shown directly.
        Default: None.
    arrows : :obj:`Boolean`
        If True draws arrows on lines in the direction of the power flow. Does
        only work when `line_color` option 'loading' is used and a time step
        is given.
        Default: False.
    limits_cb_lines : :obj:`tuple`
        Tuple with limits for colorbar of line color. First entry is the
        minimum and second entry the maximum value. Default: None.
    limits_cb_nodes : :obj:`tuple`
        Tuple with limits for colorbar of nodes. First entry is the
        minimum and second entry the maximum value. Default: None.
    xlim : :obj:`tuple`
        Limits of x-axis. Default: None.
    ylim : :obj:`tuple`
        Limits of y-axis. Default: None.
    lines_cmap : :obj:`str`
        Colormap to use for lines in case `line_color` is 'loading' or
        'expansion_costs'. Default: 'inferno_r'.
    title : :obj:`str`
        Title of the plot. Default: ''.

    """

    def get_color_and_size(name, colors_dict, sizes_dict):
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

    def nodes_by_technology(buses):
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
        for bus in buses:
            bus_colors[bus], bus_sizes[bus] = get_color_and_size(
                bus, colors_dict, sizes_dict)
        return bus_sizes, bus_colors

    def nodes_by_voltage(buses, voltage):
        bus_colors = {}
        bus_sizes = {}
        for bus in buses:
            if 'primary' in bus:
                bus_tmp = bus[12:]
            else:
                bus_tmp = bus[4:]
            if timestep is not None:
                bus_colors[bus] = abs(1 - voltage.loc[timestep,
                                                      ('mv', bus_tmp)])
            else:
                bus_colors[bus] = max(abs(1 - voltage.loc[:, ('mv', bus_tmp)]))
            bus_sizes[bus] = 50
        return bus_sizes, bus_colors

    def nodes_storage_integration(buses):
        bus_sizes = {}
        for bus in buses:
            if not 'storage' in bus:
                bus_sizes[bus] = 0
            else:
                tmp = bus.split('_')
                storage_repr = '_'.join(tmp[1:])
                bus_sizes[bus] = pypsa_network.storage_units.loc[
                           storage_repr, 'p_nom'] * 1000 / 3
        return bus_sizes

    def nodes_by_costs(buses, grid_expansion_costs):
        # sum costs for each station
        costs_lv_stations = grid_expansion_costs[
            grid_expansion_costs.index.str.contains("LVStation")]
        costs_lv_stations['station'] = \
            costs_lv_stations.reset_index()['index'].apply(
                lambda _: '_'.join(_.split('_')[0:2])).values
        costs_lv_stations = costs_lv_stations.groupby('station').sum()
        costs_mv_station = grid_expansion_costs[
            grid_expansion_costs.index.str.contains("MVStation")]
        costs_mv_station['station'] = \
            costs_mv_station.reset_index()['index'].apply(
                lambda _: '_'.join(_.split('_')[0:2])).values
        costs_mv_station = costs_mv_station.groupby('station').sum()

        bus_sizes = {}
        bus_colors = {}
        for bus in buses:
            if 'LVStation' in bus:
                try:
                    tmp = bus.split('_')
                    lv_st = '_'.join(tmp[2:])
                    bus_colors[bus] = costs_lv_stations.loc[
                        lv_st, 'total_costs']
                    bus_sizes[bus] = 100
                except:
                    bus_colors[bus] = 0
                    bus_sizes[bus] = 0
            elif 'MVStation' in bus:
                try:
                    tmp = bus.split('_')
                    mv_st = '_'.join(tmp[2:])
                    bus_colors[bus] = costs_mv_station.loc[
                        mv_st, 'total_costs']
                    bus_sizes[bus] = 100
                except:
                    bus_colors[bus] = 0
                    bus_sizes[bus] = 0
            else:
                bus_colors[bus] = 0
                bus_sizes[bus] = 0

        return bus_sizes, bus_colors

    # create pypsa network only containing MV buses and lines
    pypsa_plot = PyPSANetwork()
    pypsa_plot.buses = pypsa_network.buses.loc[pypsa_network.buses.v_nom >= 10]
    # filter buses of aggregated loads and generators
    pypsa_plot.buses = pypsa_plot.buses[
        ~pypsa_plot.buses.index.str.contains("agg")]
    pypsa_plot.lines = pypsa_network.lines[
        pypsa_network.lines.bus0.isin(pypsa_plot.buses.index)][
        pypsa_network.lines.bus1.isin(pypsa_plot.buses.index)]

    # line colors
    if line_color == 'loading':
        # calculate relative line loading
        # get load factor
        residual_load = tools.get_residual_load_from_pypsa_network(
            pypsa_network)
        case = residual_load.apply(
                lambda _: 'feedin_case' if _ < 0 else 'load_case')
        if timestep is not None:
            timeindex = [timestep]
        else:
            timeindex = line_load.index
        load_factor = pd.DataFrame(
            data={'s_nom': [float(configs[
                                      'grid_expansion_load_factors'][
                                      'mv_{}_line'.format(case.loc[_])])
                            for _ in timeindex]},
            index=timeindex)
        # get allowed line load
        s_allowed = load_factor.dot(
            pypsa_plot.lines.s_nom.to_frame().T * 1e3)
        # get line load from pf
        line_colors = line_load.loc[:, pypsa_plot.lines.index].divide(
            s_allowed).max()
    elif line_color == 'expansion_costs':
        node_color = 'expansion_costs'
        line_costs = pypsa_plot.lines.join(
            grid_expansion_costs, rsuffix='costs', how='left')
        line_colors = line_costs.total_costs.fillna(0)
    else:
        line_colors = pd.Series('black', index=pypsa_plot.lines.index)

    # bus colors and sizes
    if node_color == 'technology':
        bus_sizes, bus_colors = nodes_by_technology(pypsa_plot.buses.index)
        bus_cmap = None
    elif node_color == 'voltage':
        bus_sizes, bus_colors = nodes_by_voltage(
            pypsa_plot.buses.index, voltage)
        bus_cmap = plt.cm.Blues
    elif node_color == 'storage_integration':
        bus_sizes = nodes_storage_integration(pypsa_plot.buses.index)
        bus_colors = 'orangered'
        bus_cmap = None
    elif node_color == 'expansion_costs':
        bus_sizes, bus_colors = nodes_by_costs(pypsa_plot.buses.index,
                                               grid_expansion_costs)
        bus_cmap = None
    elif node_color is None:
        bus_sizes = 0
        bus_colors = 'r'
        bus_cmap = None
    else:
        logging.warning('Choice for `node_color` is not valid. Default is '
                        'used instead.')
        bus_sizes = 0
        bus_colors = 'r'
        bus_cmap = None

    # convert bus coordinates to Mercator
    if contextily and background_map:
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:3857')
        x2, y2 = transform(inProj, outProj,
                           list(pypsa_plot.buses.loc[:, 'x']),
                           list(pypsa_plot.buses.loc[:, 'y']))
        pypsa_plot.buses.loc[:, 'x'] = x2
        pypsa_plot.buses.loc[:, 'y'] = y2

    # plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # plot grid district
    if grid_district_geom and geopandas:
        try:
            subst = pypsa_network.buses[
                pypsa_network.buses.index.str.contains("MVStation")].index[0]
            subst_id = subst.split('_')[-1]
            projection = 3857 if contextily and background_map else 4326
            region = get_grid_district_polygon(configs, subst_id=subst_id,
                                               projection=projection)
            region.plot(ax=ax, color='white', alpha=0.2,
                        edgecolor='red', linewidth=2)
        except Exception as e:
            logging.warning("Grid district geometry could not be plotted due "
                            "to the following error: {}".format(e))

    cmap = plt.cm.get_cmap(lines_cmap)
    ll = pypsa_plot.plot(line_colors=line_colors, line_cmap=cmap, ax=ax,
                         title=title,
                         line_widths=2, #pypsa_plot.lines.s_nom,
                         branch_components=['Line'], basemap=True,
                         bus_sizes=bus_sizes, bus_colors=bus_colors,
                         bus_cmap=bus_cmap)

    # color bar line loading
    if line_color == 'loading':
        if limits_cb_lines is None:
            limits_cb_lines = (min(line_colors), max(line_colors))
        v = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10])
        cb.set_clim(vmin=limits_cb_lines[0], vmax=limits_cb_lines[1])
        cb.set_label('Line loading in p.u.')
    # color bar grid expansion costs
    elif line_color == 'expansion_costs':
        if limits_cb_lines is None:
            limits_cb_lines = (min(min(line_colors), min(bus_colors.values())),
                              max(max(line_colors), max(bus_colors.values())))
        v = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10])
        cb.set_clim(vmin=limits_cb_lines[0], vmax=limits_cb_lines[1])
        cb.set_label('Grid expansion costs in kEUR')

    # color bar voltage
    if node_color == 'voltage':
        if limits_cb_nodes is None:
            limits_cb_nodes = (min(bus_colors.values()),
                                 max(bus_colors.values()))
        v_voltage = np.linspace(limits_cb_nodes[0], limits_cb_nodes[1], 101)
        cb_voltage = plt.colorbar(ll[0], boundaries=v_voltage,
                                  ticks=v_voltage[0:101:10])
        cb_voltage.set_clim(vmin=limits_cb_nodes[0],
                            vmax=limits_cb_nodes[1])
        cb_voltage.set_label('Voltage deviation in p.u.')

    # storages
    if node_color == 'expansion_costs':
        ax.scatter(
            pypsa_plot.buses.loc[
                pypsa_network.storage_units.loc[:, 'bus'], 'x'],
            pypsa_plot.buses.loc[
                pypsa_network.storage_units.loc[:, 'bus'], 'y'],
            c='orangered',
            s=pypsa_network.storage_units.loc[:, 'p_nom'] * 1000 / 3)
    # add legend for storage size
    if (node_color == 'storage_integration' or
        node_color == 'expansion_costs') and \
            pypsa_network.storage_units.loc[:, 'p_nom'].any() > 0:
        scatter_handle = ax.scatter(
            [], [], c='orangered', s=100, label='= 300 kW battery storage')
        ax.legend(handles=[scatter_handle], scatterpoints=1, labelspacing=1,
                  title='Storage size', borderpad=1.2, loc=2, framealpha=0.5,
                  fontsize='medium')

    # axes limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # draw arrows on lines
    if arrows and timestep and line_color=='loading':
        path = ll[1].get_segments()
        colors = cmap(ll[1].get_array() / 100)
        for i in range(len(path)):
            if pypsa_network.lines_t.p0.loc[timestep,
                                            line_colors.index[i]] > 0:
                arrowprops = dict(arrowstyle="->", color='b')#colors[i])
            else:
                arrowprops = dict(arrowstyle="<-", color='b')#colors[i])
            ax.annotate(
                "",
                xy=abs(
                    (path[i][0] - path[i][1]) * 0.51 - path[i][0]),
                xytext=abs(
                    (path[i][0] - path[i][1]) * 0.49 - path[i][0]),
                arrowprops=arrowprops,
                size=10)

    # plot map data in background
    if contextily and background_map:
        try:
            add_basemap(ax, zoom=12)
        except Exception as e:
            logging.warning("Background map could not be plotted due to the "
                            "following error: {}".format(e))

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
