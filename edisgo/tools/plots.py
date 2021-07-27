import os
import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from pypsa import Network as PyPSANetwork

from pyproj import Proj
from pyproj import Transformer
import matplotlib

from edisgo.tools import tools, session_scope

if "READTHEDOCS" not in os.environ:

    from egoio.db_tables.grid import EgoDpMvGriddistrict
    from egoio.db_tables.model_draft import EgoGridMvGriddistrict
    from geoalchemy2 import shape

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
        :class:`network.results.Results`). Index of the dataframe must be
        a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    timeindex : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
        Specifies time steps histogram is plotted for. If timeindex is None all
        time steps provided in `data` are used. Default: None.
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
    timeindex = kwargs.get("timeindex", None)
    if timeindex is None:
        timeindex = data.index
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timeindex, "__len__"):
        timeindex = [timeindex]

    directory = kwargs.get("directory", None)
    filename = kwargs.get("filename", None)
    title = kwargs.get("title", "")
    x_label = kwargs.get("x_label", "")
    y_label = kwargs.get("y_label", "")

    color = kwargs.get("color", None)
    alpha = kwargs.get("alpha", 1)
    normed = kwargs.get("normed", False)

    x_limits = kwargs.get("x_limits", None)
    y_limits = kwargs.get("y_limits", None)
    binwidth = kwargs.get("binwidth", None)

    fig_size = kwargs.get("fig_size", "a5landscape")
    standard_sizes = {
        "a4portrait": (8.27, 11.69),
        "a4landscape": (11.69, 8.27),
        "a5portrait": (5.8, 8.3),
        "a5landscape": (8.3, 5.8),
    }
    try:
        fig_size = standard_sizes[fig_size]
    except:
        fig_size = standard_sizes["a5landscape"]

    plot_data = data.loc[timeindex, :].T.stack()

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
        density=normed, color=color, alpha=alpha, bins=bins, grid=True
    )
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
    basemap, extent = ctx.bounds2img(
        xmin, ymin, xmax, ymax, zoom=zoom, source=url
    )
    ax.imshow(basemap, extent=extent, interpolation="bilinear")
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))


def get_grid_district_polygon(config, subst_id=None, projection=4326):
    """
    Get MV network district polygon from oedb for plotting.

    """
    with session_scope() as session:
        # get polygon from versioned schema
        if config["data_source"]["oedb_data_source"] == "versioned":

            version = config["versioned"]["version"]
            query = session.query(
                EgoDpMvGriddistrict.subst_id, EgoDpMvGriddistrict.geom
            )
            Regions = [
                (subst_id, shape.to_shape(geom))
                for subst_id, geom in query.filter(
                    EgoDpMvGriddistrict.version == version,
                    EgoDpMvGriddistrict.subst_id == subst_id,
                ).all()
            ]

        # get polygon from model_draft
        else:
            query = session.query(
                EgoGridMvGriddistrict.subst_id, EgoGridMvGriddistrict.geom
            )
            Regions = [
                (subst_id, shape.to_shape(geom))
                for subst_id, geom in query.filter(
                    EgoGridMvGriddistrict.subst_id.in_(subst_id)
                ).all()
            ]

    crs = {"init": "epsg:3035"}
    region = gpd.GeoDataFrame(
        Regions, columns=["subst_id", "geometry"], crs=crs
    )
    region = region.to_crs(epsg=projection)

    return region


def mv_grid_topology(
    edisgo_obj,
    timestep=None,
    line_color=None,
    node_color=None,
    line_load=None,
    grid_expansion_costs=None,
    filename=None,
    arrows=False,
    grid_district_geom=True,
    background_map=True,
    voltage=None,
    limits_cb_lines=None,
    limits_cb_nodes=None,
    xlim=None,
    ylim=None,
    lines_cmap="inferno_r",
    title="",
    scaling_factor_line_width=None,
    curtailment_df=None,
    **kwargs
):
    """
    Plot line loading as color on lines.

    Displays line loading relative to nominal capacity.

    Parameters
    ----------
    edisgo_obj : :class:`~edisgo.EDisGo`
    timestep : :pandas:`pandas.Timestamp<Timestamp>`
        Time step to plot analysis results for. If `timestep` is None maximum
        line load and if given, maximum voltage deviation, is used. In that
        case arrows cannot be drawn. Default: None.
    line_color : :obj:`str` or None
        Defines whereby to choose line colors (and implicitly size). Possible
        options are:

        * 'loading'
          Line color is set according to loading of the line. Loading of MV
          lines must be provided by parameter `line_load`.
        * 'expansion_costs'
          Line color is set according to investment costs of the line. This
          option also effects node colors and sizes by plotting investment in
          stations and setting `node_color` to 'storage_integration' in order
          to plot storage size of integrated storage units. Grid expansion costs
          must be provided by parameter `grid_expansion_costs`.
        * None (default)
          Lines are plotted in black. Is also the fallback option in case of
          wrong input.

    node_color : :obj:`str` or None
        Defines whereby to choose node colors (and implicitly size). Possible
        options are:

        * 'technology'
          Node color as well as size is set according to type of node
          (generator, MV station, etc.).
        * 'voltage'
          Node color is set according to maximum voltage at each node.
          Voltages of nodes in MV network must be provided by parameter
          `voltage`.
        * 'voltage_deviation'
          Node color is set according to voltage deviation from 1 p.u..
          Voltages of nodes in MV network must be provided by parameter
          `voltage`.
        * 'storage_integration'
          Only storage units are plotted. Size of node corresponds to size of
          storage.
        * None (default)
          Nodes are not plotted. Is also the fallback option in case of wrong
          input.
        * 'curtailment'
          Plots curtailment per node. Size of node corresponds to share of
          curtailed power for the given time span. When this option is chosen
          a dataframe with curtailed power per time step and node needs to be
          provided in parameter `curtailment_df`.
        * 'charging_park'
          Plots nodes with charging stations in red.

    line_load : :pandas:`pandas.DataFrame<dataframe>` or None
        Dataframe with current results from power flow analysis in A. Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`,
        columns are the line representatives. Only needs to be provided when
        parameter `line_color` is set to 'loading'. Default: None.
    grid_expansion_costs : :pandas:`pandas.DataFrame<dataframe>` or None
        Dataframe with network expansion costs in kEUR. See `grid_expansion_costs`
        in :class:`~.network.results.Results` for more information. Only needs to
        be provided when parameter `line_color` is set to 'expansion_costs'.
        Default: None.
    filename : :obj:`str`
        Filename to save plot under. If not provided, figure is shown directly.
        Default: None.
    arrows : :obj:`Boolean`
        If True draws arrows on lines in the direction of the power flow. Does
        only work when `line_color` option 'loading' is used and a time step
        is given.
        Default: False.
    grid_district_geom : :obj:`Boolean`
        If True network district polygon is plotted in the background. This also
        requires the geopandas package to be installed. Default: True.
    background_map : :obj:`Boolean`
        If True map is drawn in the background. This also requires the
        contextily package to be installed. Default: True.
    voltage : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with voltage results from power flow analysis in p.u.. Index
        of the dataframe is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`,
        columns are the bus representatives. Only needs to be provided when
        parameter `node_color` is set to 'voltage'. Default: None.
    limits_cb_lines : :obj:`tuple`
        Tuple with limits for colorbar of line color. First entry is the
        minimum and second entry the maximum value. Only needs to be provided
        when parameter `line_color` is not None. Default: None.
    limits_cb_nodes : :obj:`tuple`
        Tuple with limits for colorbar of nodes. First entry is the
        minimum and second entry the maximum value. Only needs to be provided
        when parameter `node_color` is not None. Default: None.
    xlim : :obj:`tuple`
        Limits of x-axis. Default: None.
    ylim : :obj:`tuple`
        Limits of y-axis. Default: None.
    lines_cmap : :obj:`str`
        Colormap to use for lines in case `line_color` is 'loading' or
        'expansion_costs'. Default: 'inferno_r'.
    title : :obj:`str`
        Title of the plot. Default: ''.
    scaling_factor_line_width : :obj:`float` or None
        If provided line width is set according to the nominal apparent power
        of the lines. If line width is None a default line width of 2 is used
        for each line. Default: None.
    curtailment_df : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with curtailed power per time step and node. Columns of the
        dataframe correspond to buses and index to the time step. Only needs
        to be provided if `node_color` is set to 'curtailment'.
    legend_loc : str
        Location of legend. See matplotlib legend location options for more
        information. Default: 'upper left'.

    """

    def get_color_and_size(connected_components, colors_dict, sizes_dict):
        # Todo: handling of multiple connected elements, so far determined as
        #  'other'
        if not connected_components["transformers_hvmv"].empty:
            return colors_dict["MVStation"], sizes_dict["MVStation"]
        elif not connected_components["transformers"].empty:
            return colors_dict["LVStation"], sizes_dict["LVStation"]
        elif (
            not connected_components["generators"].empty
            and connected_components["loads"].empty
            and connected_components["charging_points"].empty
            and connected_components["storage_units"].empty
        ):
            if (
                connected_components["generators"].type.isin(["wind", "solar"])
            ).all():
                return (
                    colors_dict["GeneratorFluctuating"],
                    sizes_dict["GeneratorFluctuating"],
                )
            else:
                return colors_dict["Generator"], sizes_dict["Generator"]
        elif (
                (not connected_components["loads"].empty
                 or not connected_components["charging_points"].empty)
                and connected_components["generators"].empty
                and connected_components["storage_units"].empty
        ):
            return colors_dict["Load"], sizes_dict["Load"]
        elif not connected_components["switches"].empty:
            return (
                colors_dict["DisconnectingPoint"],
                sizes_dict["DisconnectingPoint"],
            )
        elif (
            not connected_components["storage_units"].empty
            and connected_components["loads"].empty
            and connected_components["charging_points"].empty
            and connected_components["generators"].empty
        ):
            return colors_dict["Storage"], sizes_dict["Storage"]
        elif len(connected_components["lines"]) > 1:
            return colors_dict["BranchTee"], sizes_dict["BranchTee"]
        else:
            return colors_dict["else"], sizes_dict["else"]

    def nodes_by_technology(buses, edisgo_obj):
        bus_sizes = {}
        bus_colors = {}
        colors_dict = {
            "BranchTee": "b",
            "GeneratorFluctuating": "g",
            "Generator": "k",
            "Load": "m",
            "LVStation": "c",
            "MVStation": "r",
            "Storage": "y",
            "DisconnectingPoint": "0.75",
            "else": "orange",
        }
        sizes_dict = {
            "BranchTee": 10000,
            "GeneratorFluctuating": 100000,
            "Generator": 100000,
            "Load": 100000,
            "LVStation": 50000,
            "MVStation": 120000,
            "Storage": 100000,
            "DisconnectingPoint": 75000,
            "else": 200000,
        }
        for bus in buses:
            connected_components = edisgo_obj.topology.get_connected_components_from_bus(
                bus
            )
            bus_colors[bus], bus_sizes[bus] = get_color_and_size(
                connected_components, colors_dict, sizes_dict
            )
        return bus_sizes, bus_colors

    def nodes_charging_park(buses, edisgo_obj):
        bus_sizes = {}
        bus_colors = {}
        positions = []
        colors_dict = {"ChargingPark": "r", "else": "black"}
        sizes_dict = {"ChargingPark": 100000, "else": 10000}
        for bus in edisgo_obj.topology.loads_df.index:
            if "charging_park" in bus:
                position = str(bus).rsplit("_")[-1]
                positions.append(position)
        for bus in buses:
            bus_colors[bus] = colors_dict["else"]
            bus_sizes[bus] = sizes_dict["else"]
            for position in positions:
                if position in bus:
                    bus_colors[bus] = colors_dict["ChargingPark"]
                    bus_sizes[bus] = sizes_dict["ChargingPark"]
        return bus_sizes, bus_colors

    def nodes_by_voltage(buses, voltages):
        # ToDo: Right now maximum voltage is used. Check if this should be
        #  changed
        bus_colors_dict = {}
        bus_sizes_dict = {}
        if timestep is not None:
            bus_colors_dict.update(
                {
                    bus: voltages.loc[timestep, bus]
                    for bus in buses
                }
            )
        else:
            bus_colors_dict.update(
                {
                    bus: max(voltages.loc[:, bus])
                    for bus in buses
                }
            )

        bus_sizes_dict.update({bus: 100000^2 for bus in buses})
        return bus_sizes_dict, bus_colors_dict

    def nodes_by_voltage_deviation(buses, voltages):
        bus_colors_dict = {}
        bus_sizes_dict = {}
        if timestep is not None:
            bus_colors_dict.update(
                {
                    bus: 100
                    * abs(1 - voltages.loc[timestep, bus])
                    for bus in buses
                }
            )
        else:
            bus_colors_dict.update(
                {
                    bus: 100 * max(abs(1 - voltages.loc[:, bus]))
                    for bus in buses
                }
            )

        bus_sizes_dict.update({bus: 100000^2 for bus in buses})
        return bus_sizes_dict, bus_colors_dict

    def nodes_storage_integration(buses, edisgo_obj):
        bus_sizes = {}
        buses_with_storages = buses[
            buses.isin(edisgo_obj.topology.storage_units_df.bus.values)
        ]
        buses_without_storages = buses[~buses.isin(buses_with_storages)]
        bus_sizes.update({bus: 0 for bus in buses_without_storages})
        # size nodes such that 300 kW storage equals size 100
        bus_sizes.update(
            {
                bus: edisgo_obj.topology.get_connected_components_from_bus(
                    bus
                )["storage_units"].p_nom.values.sum()
                * 1000
                / 3
                for bus in buses_with_storages
            }
        )
        return bus_sizes

    def nodes_curtailment(buses, curtailment_df):
        bus_sizes = {}
        buses_with_curtailment = buses[buses.isin(curtailment_df.columns)]
        buses_without_curtailment = buses[~buses.isin(buses_with_curtailment)]
        bus_sizes.update({bus: 0 for bus in buses_without_curtailment})
        curtailment_total = curtailment_df.sum().sum()
        # size nodes such that 100% curtailment share equals size 1000
        bus_sizes.update(
            {
                bus: curtailment_df.loc[:, bus].sum()
                / curtailment_total
                * 2000
                for bus in buses_with_curtailment
            }
        )
        return bus_sizes

    def nodes_by_costs(buses, grid_expansion_costs, edisgo_obj):
        # sum costs for each station
        costs_lv_stations = grid_expansion_costs[
            grid_expansion_costs.index.isin(
                edisgo_obj.topology.transformers_df.index
            )
        ]
        costs_lv_stations["station"] = edisgo_obj.topology.transformers_df.loc[
            costs_lv_stations.index, "bus0"
        ].values
        costs_lv_stations = costs_lv_stations.groupby("station").sum()
        costs_mv_station = grid_expansion_costs[
            grid_expansion_costs.index.isin(
                edisgo_obj.topology.transformers_hvmv_df.index
            )
        ]
        costs_mv_station[
            "station"
        ] = edisgo_obj.topology.transformers_hvmv_df.loc[
            costs_mv_station.index, "bus1"
        ]
        costs_mv_station = costs_mv_station.groupby("station").sum()

        bus_sizes = {}
        bus_colors = {}
        for bus in buses:
            # LVStation handeling
            if bus in edisgo_obj.topology.transformers_df.bus0.values:
                try:
                    bus_colors[bus] = costs_lv_stations.loc[bus, "total_costs"]
                    bus_sizes[bus] = 100
                except:
                    bus_colors[bus] = 0
                    bus_sizes[bus] = 0
            # MVStation handeling
            elif bus in edisgo_obj.topology.transformers_hvmv_df.bus1.values:
                try:
                    bus_colors[bus] = costs_mv_station.loc[bus, "total_costs"]
                    bus_sizes[bus] = 100
                except:
                    bus_colors[bus] = 0
                    bus_sizes[bus] = 0
            else:
                bus_colors[bus] = 0
                bus_sizes[bus] = 0

        return bus_sizes, bus_colors

    # set font and font size
    font = {"family": "serif", "size": 15}
    matplotlib.rc("font", **font)

    # create pypsa network only containing MV buses and lines
    pypsa_plot = PyPSANetwork()
    pypsa_plot.buses = edisgo_obj.topology.buses_df.loc[
        edisgo_obj.topology.buses_df.v_nom > 1
    ].loc[:, ["x", "y"]]
    # filter buses of aggregated loads and generators
    pypsa_plot.buses = pypsa_plot.buses[
        ~pypsa_plot.buses.index.str.contains("agg")
    ]
    pypsa_plot.lines = edisgo_obj.topology.lines_df[
        edisgo_obj.topology.lines_df.bus0.isin(pypsa_plot.buses.index)
    ][edisgo_obj.topology.lines_df.bus1.isin(pypsa_plot.buses.index)].loc[
        :, ["bus0", "bus1"]
    ]

    # line colors
    if line_color == "loading":
        line_colors = tools.calculate_relative_line_load(
            edisgo_obj, pypsa_plot.lines.index, timestep
        ).max()
    elif line_color == "expansion_costs":
        node_color = "expansion_costs"
        line_costs = pypsa_plot.lines.join(
            grid_expansion_costs, rsuffix="costs", how="left"
        )
        line_colors = line_costs.total_costs.fillna(0)
    else:
        line_colors = pd.Series("black", index=pypsa_plot.lines.index)

    # bus colors and sizes
    if node_color == "technology":
        bus_sizes, bus_colors = nodes_by_technology(
            pypsa_plot.buses.index, edisgo_obj
        )
        bus_cmap = None
    elif node_color == "voltage":
        bus_sizes, bus_colors = nodes_by_voltage(
            pypsa_plot.buses.index, voltage
        )
        bus_cmap = plt.cm.Blues
    elif node_color == "voltage_deviation":
        bus_sizes, bus_colors = nodes_by_voltage_deviation(
            pypsa_plot.buses.index, voltage
        )
        bus_cmap = plt.cm.Blues
    elif node_color == "storage_integration":
        bus_sizes = nodes_storage_integration(
            pypsa_plot.buses.index, edisgo_obj
        )
        bus_colors = "orangered"
        bus_cmap = None
    elif node_color == "expansion_costs":
        bus_sizes, bus_colors = nodes_by_costs(
            pypsa_plot.buses.index, grid_expansion_costs, edisgo_obj
        )
        bus_cmap = plt.cm.get_cmap(lines_cmap)
    elif node_color == "curtailment":
        bus_sizes = nodes_curtailment(pypsa_plot.buses.index, curtailment_df)
        bus_colors = "orangered"
        bus_cmap = None
    elif node_color == "charging_park":
        bus_sizes, bus_colors = nodes_charging_park(
            pypsa_plot.buses.index, edisgo_obj
        )
        bus_cmap = None
    elif node_color is None:
        bus_sizes = 0
        bus_colors = "r"
        bus_cmap = None
    else:
        if kwargs.get("bus_colors", None):
            bus_colors = pd.Series(kwargs.get("bus_colors")).loc[
                pypsa_plot.buses]
        else:
            logging.warning(
                "Choice for `node_color` is not valid. Default bus colors are "
                "used instead."
            )
            bus_colors = "r"
        if kwargs.get("bus_sizes", None):
            bus_sizes = pd.Series(kwargs.get("bus_sizes")).loc[
                pypsa_plot.buses]
        else:
            logging.warning(
                "Choice for `node_color` is not valid. Default bus sizes are "
                "used instead."
            )
            bus_sizes = 0
        if kwargs.get("bus_cmap", None):
            bus_cmap = kwargs.get("bus_cmap", None)
        else:
            logging.warning(
                "Choice for `node_color` is not valid. Default bus colormap "
                "is used instead."
            )
            bus_cmap = None

    # convert bus coordinates to Mercator
    if contextily and background_map:
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        x2, y2 = transformer.transform(
            list(pypsa_plot.buses.loc[:, "x"]),
            list(pypsa_plot.buses.loc[:, "y"])
        )
        pypsa_plot.buses.loc[:, "x"] = x2
        pypsa_plot.buses.loc[:, "y"] = y2

    # plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # plot network district
    if grid_district_geom and geopandas:
        try:
            projection = 3857 if contextily and background_map else 4326
            crs = {
                "init": "epsg:{}".format(
                    int(edisgo_obj.topology.grid_district["srid"])
                )
            }
            region = gpd.GeoDataFrame(
                {"geometry": [edisgo_obj.topology.grid_district["geom"]]},
                crs=crs,
            )
            if projection != int(edisgo_obj.topology.grid_district["srid"]):
                region = region.to_crs(epsg=projection)
            region.plot(
                ax=ax, color="white", alpha=0.2, edgecolor="red", linewidth=2
            )
        except Exception as e:
            logging.warning(
                "Grid district geometry could not be plotted due "
                "to the following error: {}".format(e)
            )

    # if scaling factor is given s_nom is plotted as line width
    if scaling_factor_line_width is not None:
        line_width = pypsa_plot.lines.s_nom * scaling_factor_line_width
    else:
        line_width = 2
    cmap = plt.cm.get_cmap(lines_cmap)
    ll = pypsa_plot.plot(
        line_colors=line_colors,
        line_cmap=cmap,
        ax=ax,
        title=title,
        line_widths=line_width,
        branch_components=["Line"],
        geomap=False,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        bus_cmap=bus_cmap,
    )

    # color bar line loading
    if line_color == "loading":
        if limits_cb_lines is None:
            limits_cb_lines = (min(line_colors), max(line_colors))
        v = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10])
        cb.norm.vmin = limits_cb_lines[0]
        cb.norm.vmax = limits_cb_lines[1]
        cb.set_label("Line loading in p.u.")
    # color bar network expansion costs
    elif line_color == "expansion_costs":
        if limits_cb_lines is None:
            limits_cb_lines = (
                min(min(line_colors), min(bus_colors.values())),
                max(max(line_colors), max(bus_colors.values())),
            )
        v = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10])
        cb.norm.vmin = limits_cb_lines[0]
        cb.norm.vmax = limits_cb_lines[1]
        cb.set_label("Grid expansion costs in kEUR")

    # color bar voltage
    if node_color == "voltage" or node_color == "voltage_deviation":
        if limits_cb_nodes is None:
            limits_cb_nodes = (
                min(bus_colors.values()),
                max(bus_colors.values()),
            )
        v_voltage = np.linspace(limits_cb_nodes[0], limits_cb_nodes[1], 101)
        # for some reason, the cmap given to pypsa plot is overwritten and
        # needs to be set again
        ll[0].set(cmap='Blues')
        cb_voltage = plt.colorbar(
            ll[0], boundaries=v_voltage, ticks=v_voltage[0:101:10]
        )
        cb_voltage.norm.vmin = limits_cb_nodes[0]
        cb_voltage.norm.vmax = limits_cb_nodes[1]
        if node_color == "voltage":
            cb_voltage.set_label("Maximum voltage in p.u.")
        else:
            cb_voltage.set_label("Voltage deviation in %")

    # storage_units
    if node_color == "expansion_costs":
        ax.scatter(
            pypsa_plot.buses.loc[
                edisgo_obj.topology.storage_units_df.loc[:, "bus"], "x"
            ],
            pypsa_plot.buses.loc[
                edisgo_obj.topology.storage_units_df.loc[:, "bus"], "y"
            ],
            c="orangered",
            s=edisgo_obj.topology.storage_units_df.loc[:, "p_nom"] * 1000 / 3,
        )
    # add legend for storage size and line capacity
    if (
        node_color == "storage_integration" or node_color == "expansion_costs"
    ) and edisgo_obj.topology.storage_units_df.loc[:, "p_nom"].any() > 0:
        scatter_handle = plt.scatter(
            [], [], c="orangered", s=100, label="= 300 kW battery storage"
        )
    elif node_color == "curtailment":
        scatter_handle = plt.scatter(
            [],
            [],
            c="orangered",
            s=200,
            label="$\\equiv$ 10% share of curtailment",
        )
    else:
        scatter_handle = None
    if scaling_factor_line_width is not None:
        line_handle = plt.plot(
            [],
            [],
            c="black",
            linewidth=scaling_factor_line_width * 10,
            label="= 10 MVA",
        )
    else:
        line_handle = None
    legend_loc = kwargs.get("legend_loc", "upper left")
    if scatter_handle and line_handle:
        plt.legend(
            handles=[scatter_handle, line_handle[0]],
            labelspacing=1,
            title="Storage size and line capacity",
            borderpad=0.5,
            loc=legend_loc,
            framealpha=0.5,
            fontsize="medium",
        )
    elif scatter_handle:
        plt.legend(
            handles=[scatter_handle],
            labelspacing=0,
            title=None,
            borderpad=0.3,
            loc=legend_loc,
            framealpha=0.5,
            fontsize="medium",
        )
    elif line_handle:
        plt.legend(
            handles=[line_handle[0]],
            labelspacing=1,
            title="Line capacity",
            borderpad=0.5,
            loc=legend_loc,
            framealpha=0.5,
            fontsize="medium",
        )

    # axes limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # draw arrows on lines
    if arrows and timestep and line_color == "loading":
        path = ll[1].get_segments()
        colors = cmap(ll[1].get_array() / 100)
        for i in range(len(path)):
            if edisgo_obj.lines_t.p0.loc[timestep, line_colors.index[i]] > 0:
                arrowprops = dict(arrowstyle="->", color="b")  # colors[i])
            else:
                arrowprops = dict(arrowstyle="<-", color="b")  # colors[i])
            ax.annotate(
                "",
                xy=abs((path[i][0] - path[i][1]) * 0.51 - path[i][0]),
                xytext=abs((path[i][0] - path[i][1]) * 0.49 - path[i][0]),
                arrowprops=arrowprops,
                size=10,
            )

    # plot map data in background
    if contextily and background_map:
        try:
            add_basemap(ax, zoom=12)
        except Exception as e:
            logging.warning(
                "Background map could not be plotted due to the "
                "following error: {}".format(e)
            )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
