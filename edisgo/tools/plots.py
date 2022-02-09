from __future__ import annotations

import os
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as cm

from matplotlib import pyplot as plt
from pypsa import Network as PyPSANetwork

from pyproj import Proj
from pyproj import Transformer
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from edisgo.tools import tools, session_scope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edisgo import EDisGo
    from edisgo.network.grids import Grid
    from numbers import Number
    from plotly.basedatatypes import BaseFigure

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

logger = logging.getLogger(__name__)


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


def color_map_color(
        value: Number,
        vmin: Number,
        vmax: Number,
        cmap_name: str = 'coolwarm',
):
    """
    Get matching color for a value on a matplotlib color map.

    Parameters
    ----------
    value : float or int
        Value to get color for
    vmin : float or int
        Minimum value on color map
    vmax : float or int
        Maximum value on color map
    cmap_name : str
        Name of color map to use

    Returns
    -------
    str
        Color name in hex format

    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]
    color = matplotlib.colors.rgb2hex(rgb)

    return color


def draw_plotly(
        edisgo_obj: EDisGo,
        G: Graph | None = None,
        line_color: str = "relative_loading",
        node_color: str = "voltage_deviation",
        grid: bool | Grid = False,
) -> BaseFigure:
    """
    Draw a plotly html figure

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    G : :networkx:`networkx.Graph<network.Graph>`, optional
        Graph representation of the grid as networkx Ordered Graph, where lines are
        represented by edges in the graph, and buses and transformers are represented by
        nodes. If no graph is given the mv grid graph of the edisgo object is used.
    line_color : str
        Defines whereby to choose line colors (and implicitly size). Possible
        options are:

        * 'loading'
          Line color is set according to loading of the line.
        * 'relative_loading' (Default)
          Line color is set according to relative loading of the line.
        * 'reinforce'
          Line color is set according to investment costs of the line.

    node_color : str or None
        Defines whereby to choose node colors (and implicitly size). Possible
        options are:

        * 'adjacencies'
          Node color as well as size is set according to the number of direct neighbors.
        * 'voltage_deviation' (default)
          Node color is set according to voltage deviation from 1 p.u..

    grid : :class:`~.network.grids.Grid` or bool
        Grid to use as root node. If a grid is given the transforer station is used
        as root. If False the root is set to the coordinates x=0 and y=0. Else the
        coordinates from the hv-mv-station of the mv grid are used. Default: False

    Returns
    -------
    :plotly:`plotly.graph_objects.Figure`
        Plotly figure with branches and nodes.

    """
    # initialization
    transformer_4326_to_3035 = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3035",
        always_xy=True,
    )

    if G is None:
        G = edisgo_obj.topology.mv_grid.graph

    node_list = list(G.nodes())

    if hasattr(grid, "transformers_df"):
        node_root = grid.transformers_df.bus1.iat[0]
        x_root, y_root = G.nodes[node_root]['pos']

    elif not grid:
        x_root = 0
        y_root = 0

    else:
        node_root = edisgo_obj.topology.transformers_hvmv_df.bus1.iat[0]
        x_root, y_root = G.nodes[node_root]['pos']

    x_root, y_root = transformer_4326_to_3035.transform(x_root, y_root)

    # line text
    middle_node_x = []
    middle_node_y = []
    middle_node_text = []

    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        x0, y0 = transformer_4326_to_3035.transform(x0, y0)
        x1, y1 = transformer_4326_to_3035.transform(x1, y1)
        middle_node_x.append((x0 - x_root + x1 - x_root) / 2)
        middle_node_y.append((y0 - y_root + y1 - y_root) / 2)

        branch_name = edge[2]['branch_name']

        text = str(branch_name)
        try:
            loading = edisgo_obj.results.s_res.T.loc[branch_name].max()
            text += '<br>' + 'Loading = ' + str(loading)
        except KeyError:
            logger.debug(
                f"Could not find loading for branch {branch_name}", exc_info=True
            )
            text = text

        try:
            line_parameters = edisgo_obj.topology.lines_df.loc[branch_name, :]
            for index, value in line_parameters.iteritems():
                text += '<br>' + str(index) + ' = ' + str(value)
        except KeyError:
            logger.debug(
                f"Could not find line parameters for branch {branch_name}",
                exc_info=True,
            )
            text = text

        middle_node_text.append(text)

    middle_node_trace = go.Scatter(
        x=middle_node_x,
        y=middle_node_y,
        text=middle_node_text,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            opacity=0.0,
            size=10,
            color='white'
        )
    )

    data = [middle_node_trace]

    # line plot
    if line_color == 'loading':
        s_res_view = edisgo_obj.results.s_res.T.index.isin(
            [edge[2]['branch_name'] for edge in G.edges.data()])
        color_min = edisgo_obj.results.s_res.T.loc[s_res_view].T.min().max()
        color_max = edisgo_obj.results.s_res.T.loc[s_res_view].T.max().max()

    elif line_color == 'relative_loading':
        color_min = 0
        color_max = 1

    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        x0, y0 = transformer_4326_to_3035.transform(x0, y0)
        x1, y1 = transformer_4326_to_3035.transform(x1, y1)

        edge_x = [x0 - x_root, x1 - x_root, None]
        edge_y = [y0 - y_root, y1 - y_root, None]

        if line_color == 'reinforce':
            if edisgo_obj.results.grid_expansion_costs.index.isin(
                    [edge[2]['branch_name']]
            ).any():
                color = 'lightgreen'
            else:
                color = 'black'

        elif line_color == 'loading':
            loading = edisgo_obj.results.s_res.T.loc[edge[2]['branch_name']].max()
            color = color_map_color(
                loading,
                vmin=color_min,
                vmax=color_max,
            )

        elif line_color == 'relative_loading':
            loading = edisgo_obj.results.s_res.T.loc[edge[2]['branch_name']].max()
            s_nom = edisgo_obj.topology.lines_df.s_nom.loc[edge[2]['branch_name']]
            color = color_map_color(
                loading / s_nom,
                vmin=color_min,
                vmax=color_max,
            )
            if loading > s_nom:
                color = 'green'
        else:
            color = 'black'

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            hoverinfo='none',
            opacity=0.4,
            mode='lines',
            line=dict(
                width=2,
                color=color
            )
        )
        data.append(edge_trace)

    # node plot
    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        x, y = transformer_4326_to_3035.transform(x, y)
        node_x.append(x - x_root)
        node_y.append(y - y_root)

    if node_color == 'voltage_deviation':
        colors = []

        for node in G.nodes():
            v_res = edisgo_obj.results.v_res.T.loc[node]
			v_min = v_res.min()
            v_max = v_res.max()

            if abs(v_min - 1) > abs(v_max - 1):
                color = v_min - 1
            else:
                color = v_max - 1

            colors.append(color)

		colorbar = dict(
			thickness=15,
			title='Node Voltage Deviation',
			xanchor='left',
			titleside='right'
		)
        colorscale = 'RdBu'
        cmid = 0

    else:
        colors = [
            len(adjacencies[1]) for adjacencies in G.adjacency()
        ]
        colorscale = 'YlGnBu'
        cmid = None

		colorbar = dict(
			thickness=15,
			title='Node Connections',
			xanchor='left',
			titleside='right'
		)

    node_text = []
    for node in G.nodes():
        text = str(node)
        try:
            peak_load = edisgo_obj.topology.loads_df.loc[
                edisgo_obj.topology.loads_df.bus == node].peak_load.sum()
            text += '<br>' + 'peak_load = ' + str(peak_load)
            p_nom = edisgo_obj.topology.generators_df.loc[
                edisgo_obj.topology.generators_df.bus == node].p_nom.sum()
            text += '<br>' + 'p_nom_gen = ' + str(p_nom)
            v_min = edisgo_obj.results.v_res.T.loc[node].min()
            v_max = edisgo_obj.results.v_res.T.loc[node].max()
            if abs(v_min - 1) > abs(v_max - 1):
                text += '<br>' + 'v = ' + str(v_min)
            else:
                text += '<br>' + 'v = ' + str(v_max)
        except KeyError:
            logger.debug(
                f"Failed to add text for node {node}.", exc_info=True
            )
            text = text

        try:
            text = text + '<br>' + 'Neighbors = ' + str(G.degree(node))
        except KeyError:
            logger.debug(
                f"Failed to add neighbors to text for node {node}.", exc_info=True
            )
            text = text

        try:
            node_parameters = edisgo_obj.topology.buses_df.loc[node]
            for index, value in node_parameters.iteritems():
                text += '<br>' + str(index) + ' = ' + str(value)
        except KeyError:
            logger.debug(
                f"Failed to add neighbors to text for node {node}.", exc_info=True
            )
            text = text

        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            color=colors,
            size=8,
            cmid=cmid,
            line_width=2,
            colorbar=colorbar
        )
    )

    data.append(node_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            height=500,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
            yaxis=dict(showgrid=True, zeroline=True, showticklabels=True)
        )
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def chosen_graph(
        edisgo_obj: EDisGo,
        selected_grid: str,
        lv_grid_name_list: list[str],
) -> tuple[Graph, bool | Grid]:
    """
    Get the matching networkx graph from a chosen grid.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    selected_grid : str
        Grid name. Can be either 'Grid' to select the mv grid with all lv grids or
        the name of the mv grid to select only the mv grid or the name of one of the
        lv grids of the eDisGo object to select a specific lv grid.
    lv_grid_name_list : list(str)
        List of the names of the lv grids within the mv grid of the eDisGo object.

    Returns
    -------
    :networkx:`networkx.Graph<network.Graph>`
        networkx graph of the selected grid
    :class:`~.network.grids.Grid` or bool
        Grid to use as root node. See :py:func:`~edisgo.tools.plots.draw_plotly` for
        more information.

    """
    mv_grid = edisgo_obj.topology.mv_grid

    if selected_grid == 'Grid':
        G = edisgo_obj.to_graph()
        grid = True
    elif selected_grid == str(mv_grid):
        G = mv_grid.graph
        grid = mv_grid
    elif selected_grid.split('_')[0] == 'LVGrid':
        try:
            lv_grid_id = lv_grid_name_list.index(selected_grid)
        except ValueError:
            logger.exception(f"Selceted grid {selected_grid} is not a valid lv grid.")

        lv_grid = list(edisgo_obj.topology.mv_grid.lv_grids)[lv_grid_id]
        G = lv_grid.graph
        grid = lv_grid
    else:
        raise ValueError(f"False Grid. '{selected_grid}' is not a valid input.")

    return G, grid


def dash_plot(
        edisgo_objects: EDisGo | dict[str, EDisGo],
        line_plot_modes: list[str] | None = None,
        node_plot_modes: list[str] | None = None,
) -> JupyterDash:
    """
    Generates a jupyter dash app from given eDisGo object(s).

    TODO: The app can't display two seperate colorbars for line and bus values atm
    TODO: The colorbar title does not change yet when another mode is loaded

    Parameters
    ----------
    edisgo_objects : :class:`~.EDisGo` or dict[str, :class:`~.EDisGo`]
        eDisGo objects to show in plotly dash app. In the case of multiple edisgo
        objects pass a dictionary with the eDisGo objects as values and the respective
        eDisGo object names as keys.
    line_plot_modes : list(str), optional
        List of line plot modes to display in plotly dash app. See
        :py:func:`~edisgo.tools.plots.draw_plotly` for more information. If None is
        passed the modes 'reinforce', 'loading' and 'relative_loading' will be used.
        Default: None
    node_plot_modes : list(str), optional
        List of line plot modes to display in plotly dash app. See
        :py:func:`~edisgo.tools.plots.draw_plotly` for more information. If None is
        passed the modes 'adjacencies' and 'voltage_deviation' will be used.
        Default: None
    Returns
    -------
    JupyterDash
        Jupyter dash app.

    """
    if isinstance(edisgo_objects, dict):
        edisgo_name_list = list(edisgo_objects.keys())
        edisgo_obj_1 = list(edisgo_objects.values())[0]
    else:
        edisgo_name_list = ["edisgo_obj"]
        edisgo_obj_1 = edisgo_objects

    mv_grid = edisgo_obj_1.topology.mv_grid

    lv_grid_name_list = list(map(str, mv_grid.lv_grids))

    grid_name_list = ['Grid', str(mv_grid)] + lv_grid_name_list

    if line_plot_modes is None:
        line_plot_modes = ['reinforce', 'loading', 'relative_loading']
    if node_plot_modes is None:
        node_plot_modes = ['adjacencies', 'voltage_deviation']

    app = JupyterDash(__name__)

    if isinstance(edisgo_objects, dict) and len(edisgo_objects) > 1:
        app.layout = html.Div([
            html.Div([
                dcc.Dropdown(
                    id='dropdown_edisgo_object_1',
                    options=[{'label': i, 'value': i} for i in edisgo_name_list],
                    value=edisgo_name_list[0]
                ),
                dcc.Dropdown(
                    id='dropdown_edisgo_object_2',
                    options=[{'label': i, 'value': i} for i in edisgo_name_list],
                    value=edisgo_name_list[1]
                ),
                dcc.Dropdown(
                    id='dropdown_grid',
                    options=[{'label': i, 'value': i} for i in grid_name_list],
                    value=grid_name_list[1]
                ),
                dcc.Dropdown(
                    id='dropdown_line_plot_mode',
                    options=[{'label': i, 'value': i} for i in line_plot_modes],
                    value=line_plot_modes[0]
                ),
                dcc.Dropdown(
                    id='dropdown_node_plot_mode',
                    options=[{'label': i, 'value': i} for i in node_plot_modes],
                    value=node_plot_modes[0]
                )
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='fig_1')
                ], style={'flex': 'auto'}),
                html.Div([
                    dcc.Graph(id='fig_2')
                ], style={'flex': 'auto'})
            ], style={'display': 'flex', 'flex-direction': 'row'})
        ], style={'display': 'flex', 'flex-direction': 'column'})

        @app.callback(
            Output('fig_1', 'figure'),
            Output('fig_2', 'figure'),
            Input('dropdown_grid', 'value'),
            Input('dropdown_edisgo_object_1', 'value'),
            Input('dropdown_edisgo_object_2', 'value'),
            Input('dropdown_line_plot_mode', 'value'),
            Input('dropdown_node_plot_mode', 'value'))
        def update_figure(
                selected_grid,
                selected_edisgo_object_1,
                selected_edisgo_object_2,
                selected_line_plot_mode,
                selected_node_plot_mode
        ):
            edisgo_obj = edisgo_objects[selected_edisgo_object_1]
            (G, grid) = chosen_graph(edisgo_obj, selected_grid, lv_grid_name_list)
            fig_1 = draw_plotly(
                edisgo_obj,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid
            )

            edisgo_obj = edisgo_objects[selected_edisgo_object_2]
            (G, grid) = chosen_graph(edisgo_obj, selected_grid, lv_grid_name_list)
            fig_2 = draw_plotly(
                edisgo_obj,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid
            )

            return fig_1, fig_2
    else:
        app.layout = html.Div([
            html.Div([
                dcc.Dropdown(
                    id='dropdown_grid',
                    options=[{'label': i, 'value': i} for i in grid_name_list],
                    value=grid_name_list[1]
                ),
                dcc.Dropdown(
                    id='dropdown_line_plot_mode',
                    options=[{'label': i, 'value': i} for i in line_plot_modes],
                    value=line_plot_modes[0]
                ),
                dcc.Dropdown(
                    id='dropdown_node_plot_mode',
                    options=[{'label': i, 'value': i} for i in node_plot_modes],
                    value=node_plot_modes[0]
                )
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='fig')
                ], style={'flex': 'auto'})
            ], style={'display': 'flex', 'flex-direction': 'row'})
        ], style={'display': 'flex', 'flex-direction': 'column'})

        @app.callback(
            Output('fig', 'figure'),
            Input('dropdown_grid', 'value'),
            Input('dropdown_line_plot_mode', 'value'),
            Input('dropdown_node_plot_mode', 'value'))
        def update_figure(
                selected_grid,
                selected_line_plot_mode,
                selected_node_plot_mode
        ):
            (G, grid) = chosen_graph(edisgo_obj_1, selected_grid, lv_grid_name_list)
            fig = draw_plotly(
                edisgo_obj_1,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid
            )
            return fig

    return app
