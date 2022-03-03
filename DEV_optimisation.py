# Script to add optimisation to the functionalities
from itertools import product
from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.electromobility import get_energy_bands_for_optimization
import multiprocessing as mp
import traceback

# grid_ids = [176, 177, 1056, 1690, 1811, 2534]
# use_cases = ["home", "work"]
# variations = list(product(grid_ids, use_cases))
variations = [(177, "home"), (177, "work"), (176, "home")]
data_dir = r"C:\Users\aheider\Documents\Grids"
# num_pools = min(mp.cpu_count()/2, len(variations))
num_pools = 3


def extract_and_save_bands_parallel(variation):
    # try:
    grid_id = variation[0]
    use_case = variation[1]
    print("Extracting bands for {}-{}".format(grid_id, use_case))
    edisgo_obj = import_edisgo_from_files(data_dir+r"\{}\dumb".format(grid_id), import_timeseries=True,
                                          import_electromobility=True)
    power, lower, upper = get_energy_bands_for_optimization(edisgo_obj, use_case)

    power.to_csv(data_dir+r"\{}\upper_power_{}.csv".format(grid_id, use_case))
    lower.to_csv(data_dir+r"\{}\lower_energy_{}.csv".format(grid_id, use_case))
    upper.to_csv(data_dir+r"\{}\upper_energy_{}.csv".format(grid_id, use_case))
    print("Successfully created bands for {}-{}".format(grid_id, use_case))
    # except:
    #     print("Something went wrong with {}-{}".format(grid_id, use_case))
    #     print(traceback.format_exc())


if __name__ == '__main__':
    if num_pools > 1:
        pool = mp.Pool(int(num_pools))
        pool.map_async(extract_and_save_bands_parallel, variations).get()
        pool.close()
    else:
        for variation in variations:
            extract_and_save_bands_parallel(variation)
    print("SUCCESS")