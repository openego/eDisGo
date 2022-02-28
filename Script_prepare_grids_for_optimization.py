import multiprocessing as mp
import os
from edisgo.edisgo import import_edisgo_from_files
from edisgo.tools.complexity_reduction import remove_1m_lines_from_edisgo

# Script to prepare grids for optimisation. The necessary steps are:
# Timeseries: Extract extreme weeks
# Topology: Remove 1m lines, extract feeders, extract downstream nodes matrix

grid_dir = r"H:\Grids"
strategy = "dumb"
use_mp = False
remove_1m_lines = True
extract_feeders = True


def remove_1m_lines_from_edisgo_parallel(grid_id):
    edisgo = import_edisgo_from_files(os.path.join(grid_dir, str(grid_id), strategy))
    no_bus_pre = len(edisgo.topology.buses_df)
    no_line_pre = len(edisgo.topology.lines_df)
    print("Grid has {} buses and {} lines before reduction".format(no_bus_pre, no_line_pre))
    edisgo = remove_1m_lines_from_edisgo(edisgo)
    no_bus_after = len(edisgo.topology.buses_df)
    no_line_after = len(edisgo.topology.lines_df)
    print("Grid has {} buses and {} lines after reduction".format(no_bus_after, no_line_after))
    print("{} buses and {} lines removed".format(no_bus_pre-no_bus_after, no_bus_pre-no_line_after))
    edisgo.topology.to_csv(os.path.join(grid_dir, str(grid_id), strategy, "topology"))


if __name__ == '__main__':
    grid_ids = [176, 177, 1056, 1690, 1811, 2534]

    if use_mp:
        pool = mp.Pool(len(grid_ids))
        if remove_1m_lines:
            print("Removing 1m lines")
            pool.map(remove_1m_lines_from_edisgo, grid_ids)
        pool.close()
    else:
        for grid_id in grid_ids:
            if remove_1m_lines:
                print("Removing 1m lines")
                remove_1m_lines_from_edisgo_parallel(grid_id)