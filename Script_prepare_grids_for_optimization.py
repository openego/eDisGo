import multiprocessing as mp
import os
import traceback

import networkx as nx
import pandas as pd

from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.electromobility import get_energy_bands_for_optimization
from edisgo.network.timeseries import TimeSeries
from edisgo.network.topology import Topology
from edisgo.opf.lopf import import_flexibility_bands
from edisgo.tools.complexity_reduction import (
    extract_feeders_nx,
    remove_1m_lines_from_edisgo,
)

# Script to prepare grids for optimisation. The necessary steps are:
# Timeseries: Extract extreme weeks
# Topology: Remove 1m lines, extract feeders, extract downstream nodes matrix

grid_dir = r"H:\no_HP"
grid_dir_ladina = r"H:\Grids Ladina"
ts_reduction_dir = r"C:\Users\aheider\Documents\Grids\simbev_nep_2035_results"
bands_dir = r"C:\Users\aheider\Documents\Grids"
data_dir = r"C:\Users\aheider\Documents\Grids"
strategy = "dumb"
use_mp = False
remove_1m_lines = False
extract_bands = False
extract_extreme_weeks = False
extract_extreme_weeks_no_hp = False
reduce_timeseries_to_extreme_weeks = False
reduce_timeseries_to_extreme_weeks_no_hp = True
reduce_bands_to_extreme_weeks = False
extract_feeders = False
get_downstream_node_matrix = False
cpu_count = 1  # int(mp.cpu_count()/2)


def remove_1m_lines_from_edisgo_parallel(grid_id):
    edisgo = import_edisgo_from_files(os.path.join(grid_dir, str(grid_id), strategy))
    no_bus_pre = len(edisgo.topology.buses_df)
    no_line_pre = len(edisgo.topology.lines_df)
    print(
        "Grid has {} buses and {} lines before reduction".format(
            no_bus_pre, no_line_pre
        )
    )
    edisgo = remove_1m_lines_from_edisgo(edisgo)
    no_bus_after = len(edisgo.topology.buses_df)
    no_line_after = len(edisgo.topology.lines_df)
    print(
        "Grid has {} buses and {} lines after reduction".format(
            no_bus_after, no_line_after
        )
    )
    print(
        "{} buses and {} lines removed".format(
            no_bus_pre - no_bus_after, no_bus_pre - no_line_after
        )
    )
    edisgo.topology.to_csv(os.path.join(grid_dir, str(grid_id), strategy, "topology"))


def extract_and_save_bands_parallel(grid_id):
    try:
        for use_case in ["home", "work"]:
            print("Extracting bands for {}-{}".format(grid_id, use_case))
            edisgo_obj = import_edisgo_from_files(
                data_dir + r"\{}\dumb".format(grid_id),
                import_timeseries=True,
                import_electromobility=True,
            )
            power, lower, upper = get_energy_bands_for_optimization(
                edisgo_obj, use_case
            )

            power.to_csv(data_dir + r"\{}\upper_power_{}.csv".format(grid_id, use_case))
            lower.to_csv(
                data_dir + r"\{}\lower_energy_{}.csv".format(grid_id, use_case)
            )
            upper.to_csv(
                data_dir + r"\{}\upper_energy_{}.csv".format(grid_id, use_case)
            )
            print("Successfully created bands for {}-{}".format(grid_id, use_case))
    except Exception:
        print("Something went wrong with {}-{}".format(grid_id, use_case))
        print(traceback.format_exc())


def save_extreme_weeks_timeindex(grid_id):
    # load extreme weeks
    ts = pd.read_csv(
        os.path.join(
            ts_reduction_dir,
            str(grid_id),
            strategy,
            "timeseries",
            "charging_points_active_power.csv",
        ),
        index_col=0,
        parse_dates=True,
    )
    timeindex = pd.DataFrame(index=ts.index)
    timeindex.to_csv(
        os.path.join(grid_dir, str(grid_id), "timeindex_extreme_weeks.csv")
    )


def save_extreme_weeks_timeindex_no_hp(grid_id):
    # get extreme week generation
    ts = pd.read_csv(
        os.path.join(
            grid_dir, str(grid_id), "timeseries", "generators_active_power.csv"
        ),
        index_col=0,
        parse_dates=True,
    ).sum(axis=1)
    max_gen = ts[ts == ts.max()]
    week = max_gen.index.isocalendar().week[0]
    # adapt week if timestep is within the first seven hours
    extreme_week = ts[ts.index.isocalendar().week == week].reset_index()
    if extreme_week.loc[extreme_week.snapshot == max_gen.index[0]].index[0] < 7:
        week = week - 1
        extreme_week = ts[ts.index.isocalendar().week == week].reset_index()
    week_max_gen = pd.date_range(
        extreme_week.loc[7, "snapshot"], periods=7 * 24, freq="1h"
    )
    # extreme week with highest demand from heat: 8
    week_max_heat_demand = pd.date_range(
        "2011-02-21 07:00:00", periods=7 * 24, freq="1h"
    )
    if week < 8:
        index = week_max_gen.append(week_max_heat_demand)
    elif week == 8:
        raise NotImplementedError(
            "Weeks of highest demand and generation are the same."
        )
    else:
        index = week_max_heat_demand.append(week_max_gen)
    timeindex = pd.DataFrame(index=index)
    timeindex.to_csv(
        os.path.join(grid_dir, str(grid_id), "timeindex_extreme_weeks.csv")
    )


def extract_extreme_weeks_parallel(grid_id):
    """
    Method to get extreme weeks from previous run and reduce new objects to these weeks.
    """
    # load extreme weeks
    ts = pd.read_csv(
        os.path.join(grid_dir, str(grid_id), "timeindex_extreme_weeks.csv"),
        index_col=0,
        parse_dates=True,
    )
    timeindex = ts.index
    # load original edisgo object
    edisgo = import_edisgo_from_files(
        os.path.join(grid_dir, str(grid_id), strategy),
        import_topology=False,
        import_timeseries=True,
    )
    if not (timeindex.isin(edisgo.timeseries.timeindex)).all():
        raise ValueError("Edisgo object does not contain the given extreme weeks")
    # adapt timeseries
    attributes = TimeSeries()._attributes
    edisgo.timeseries.timeindex = timeindex
    for attr in attributes:
        if not getattr(edisgo.timeseries, attr).empty:
            setattr(
                edisgo.timeseries, attr, getattr(edisgo.timeseries, attr).loc[timeindex]
            )
    # save adapted timeseries object
    edisgo.timeseries.to_csv(
        os.path.join(grid_dir, str(grid_id), strategy, "timeseries")
    )
    # Todo: adapt flexibility bands


def extract_extreme_weeks_ladina(grid_id, adapt_edisgo=False, adapt_bands=True):
    """
    Method to get extreme weeks from previous run and reduce new objects to these weeks.
    """
    # load extreme weeks
    ts = pd.read_csv(
        os.path.join(grid_dir, str(grid_id), "timeindex_extreme_weeks.csv"),
        index_col=0,
        parse_dates=True,
    )
    timeindex = ts.index
    if adapt_edisgo:
        # load original edisgo object
        edisgo = import_edisgo_from_files(
            os.path.join(grid_dir, str(grid_id)),
            import_topology=True,
            import_timeseries=True,
        )
        if not (timeindex.isin(edisgo.timeseries.timeindex)).all():
            raise ValueError("Edisgo object does not contain the given extreme weeks")
        # adapt timeseries
        attributes = TimeSeries()._attributes
        edisgo.timeseries.timeindex = timeindex
        for attr in attributes:
            if not getattr(edisgo.timeseries, attr).empty:
                setattr(
                    edisgo.timeseries,
                    attr,
                    getattr(edisgo.timeseries, attr).loc[timeindex],
                )
        # save adapted timeseries object
        edisgo.save(os.path.join(grid_dir_ladina, str(grid_id)))
    if adapt_bands:
        # adapt flexibility bands
        bands = import_flexibility_bands(
            os.path.join(data_dir, str(grid_id)), use_cases=["home", "work"]
        )
        for name, band in bands.items():
            if name == "upper_power":
                band.resample("1h").mean().loc[timeindex].to_csv(
                    os.path.join(grid_dir_ladina, str(grid_id), name + ".csv")
                )
            else:
                band.resample("1h").max().loc[timeindex].to_csv(
                    os.path.join(grid_dir_ladina, str(grid_id), name + ".csv")
                )
            # elif name == "lower_energy":
            #     band.resample("1h").min().loc[timeindex].to_csv(
            #         os.path.join(grid_dir_ladina, str(grid_id), name + ".csv"))


def extract_extreme_weeks_from_bands(grid_id):
    """
    Method to reduce energy bands to extreme weeks and save them
    """
    ts = pd.read_csv(
        os.path.join(grid_dir, str(grid_id), "timeindex_extreme_weeks.csv"),
        index_col=0,
        parse_dates=True,
    )
    timeindex = ts.index
    bands = ["upper_power", "upper_energy", "lower_energy"]
    bands_dict = {}
    for use_case in ["home", "work"]:
        for band in bands:
            bands_dict[band] = pd.read_csv(
                bands_dir + r"\{}\{}_{}.csv".format(grid_id, band, use_case),
                index_col=0,
                parse_dates=True,
            )
            bands_dict[band].loc[timeindex].to_csv(
                os.path.join(grid_dir, str(grid_id), "{}_{}.csv".format(band, use_case))
            )


def extract_feeders_parallel(grid_id):
    try:
        edisgo_dir = os.path.join(grid_dir, str(grid_id), strategy)
        save_dir = os.path.join(grid_dir, str(grid_id))
        edisgo_obj = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
        extract_feeders_nx(edisgo_obj, save_dir)
    except Exception as e:
        print("Problem in grid {}.".format(grid_id))
        print(e)


def get_downstream_node_matrix_feeders_parallel_server(grid_id_feeder_tuple):
    grid_id = grid_id_feeder_tuple[0]
    feeder_id = grid_id_feeder_tuple[1]
    edisgo_dir = os.path.join(grid_dir, str(grid_id), "feeder", str(feeder_id))
    if os.path.isfile(
        edisgo_dir + "/downstream_node_matrix_{}_{}.csv".format(grid_id, feeder_id)
    ):
        return
    try:
        edisgo_obj = import_edisgo_from_files(edisgo_dir)
        downstream_node_matrix = get_downstream_nodes_matrix_iterative(
            edisgo_obj.topology
        )
        downstream_node_matrix.to_csv(
            edisgo_dir + "/downstream_node_matrix_{}_{}.csv".format(grid_id, feeder_id)
        )
    except Exception as e:
        print("Problem in feeder {} of grid {}.".format(feeder_id, grid_id))
        print(e.args)
        print(e)
    return


def get_downstream_nodes_matrix_iterative(grid):
    """
    Method that returns matrix M with 0 and 1 entries describing the relation
    of buses within the network. If bus b is descendant of a (assuming the
    station is the root of the radial network) M[a,b] = 1, otherwise M[a,b] = 0.
    The matrix is later used to determine the power flow at the different buses
    by multiplying with the nodal power flow. S_sum = M * s, where s is the
    nodal power vector.

    Note: only works for radial networks.

    :param grid: either Topology, MVGrid or LVGrid
    :return:
    Todo: Check version with networkx successor
    """

    def recursive_downstream_node_matrix_filling(
        current_bus, current_feeder, downstream_node_matrix, grid, visited_buses
    ):
        current_feeder.append(current_bus)
        for neighbor in tree.successors(current_bus):
            if neighbor not in visited_buses and neighbor not in current_feeder:
                recursive_downstream_node_matrix_filling(
                    neighbor,
                    current_feeder,
                    downstream_node_matrix,
                    grid,
                    visited_buses,
                )
        # current_bus = current_feeder.pop()
        downstream_node_matrix.loc[current_feeder, current_bus] = 1
        visited_buses.append(current_bus)
        if len(visited_buses) % 10 == 0:
            print(
                "{} % of the buses have been checked".format(
                    len(visited_buses) / len(buses) * 100
                )
            )
        current_feeder.pop()

    buses = grid.buses_df.index.values
    if str(type(grid)) == str(Topology):
        graph = grid.to_graph()
        slack = grid.mv_grid.station.index[0]
    else:
        graph = grid.graph
        slack = grid.transformers_df.bus1.iloc[0]
    tree = nx.bfs_tree(graph, slack)

    print("Matrix for {} buses is extracted.".format(len(buses)))
    downstream_node_matrix = pd.DataFrame(columns=buses, index=buses)
    downstream_node_matrix.fillna(0, inplace=True)

    print("Starting iteration.")
    visited_buses = []
    current_feeder = []

    recursive_downstream_node_matrix_filling(
        slack, current_feeder, downstream_node_matrix, grid, visited_buses
    )

    return downstream_node_matrix


if __name__ == "__main__":
    grid_ids = [176, 177, 1056, 1690, 1811, 2534]

    if cpu_count > 1:
        pool = mp.Pool(cpu_count)
        if remove_1m_lines:
            print("Removing 1m lines")
            pool.map_async(remove_1m_lines_from_edisgo, grid_ids).get()
        if extract_bands:
            print("Extracting flexibility bands.")
            pool.map_async(extract_and_save_bands_parallel, grid_ids).get()
        if extract_extreme_weeks:
            print("Extracting extreme weeks.")
            pool.map_async(save_extreme_weeks_timeindex, grid_ids).get()
        if extract_extreme_weeks_no_hp:
            print("Extracting extreme weeks no hp.")
            pool.map_async(save_extreme_weeks_timeindex_no_hp, grid_ids).get()
        if reduce_timeseries_to_extreme_weeks:
            print("Reducing timeseries.")
            pool.map_async(extract_extreme_weeks_parallel, grid_ids).get()
        if reduce_timeseries_to_extreme_weeks_no_hp:
            print("Reducing timeseries and bands no hp.")
            pool.map_async(extract_extreme_weeks_ladina, grid_ids).get()
        if reduce_bands_to_extreme_weeks:
            print("Reducing bands.")
            pool.map_async(extract_extreme_weeks_from_bands, grid_ids).get()
        if extract_feeders:
            print("Extracting feeders.")
            pool.map_async(extract_feeders_parallel, grid_ids).get()
        if get_downstream_node_matrix:
            print("Getting downstream nodes matrices")
            grid_id_feeder_tuples = []
            for grid_id in grid_ids:
                feeder_dir = os.path.join(grid_dir, str(grid_id), "feeder")
                for feeder in os.listdir(feeder_dir):
                    grid_id_feeder_tuples.append((grid_id, feeder))
            pool.map_async(
                get_downstream_node_matrix_feeders_parallel_server,
                grid_id_feeder_tuples,
            ).get()
        pool.close()
    else:
        for grid_id in grid_ids:
            print("Preparing grid {}".format(grid_id))
            if remove_1m_lines:
                print("Removing 1m lines")
                remove_1m_lines_from_edisgo_parallel(grid_id)
            if extract_bands:
                print("Extracting flexibility bands.")
                extract_and_save_bands_parallel(grid_id)
            if extract_extreme_weeks:
                print("Extracting extreme weeks.")
                save_extreme_weeks_timeindex(grid_id)
            if extract_extreme_weeks_no_hp:
                print("Extracting extreme weeks no hp.")
                save_extreme_weeks_timeindex_no_hp(grid_id)
            if reduce_bands_to_extreme_weeks:
                print("Reducing bands.")
                extract_extreme_weeks_from_bands(grid_id)
            if reduce_timeseries_to_extreme_weeks:
                print("Reducing timeseries.")
                extract_extreme_weeks_parallel(grid_id)
            if reduce_timeseries_to_extreme_weeks_no_hp:
                print("Reducing timeseries and bands no hp.")
                extract_extreme_weeks_ladina(grid_id)
            if extract_feeders:
                print("Extracting feeders.")
                extract_feeders_parallel(grid_id)
            if get_downstream_node_matrix:
                print("Getting downstream nodes matrices")
                feeder_dir = os.path.join(grid_dir, str(grid_id), "feeder")
                for feeder in os.listdir(feeder_dir):
                    get_downstream_node_matrix_feeders_parallel_server(
                        (grid_id, feeder)
                    )
