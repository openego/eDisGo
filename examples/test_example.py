from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
import pandas as pd
import os
import numpy as np
#from elia_project import results_helper_functions
from edisgo.tools import pypsa_io
from edisgo.network.timeseries import get_component_timeseries
from time import time
from edisgo.tools.tools import assign_feeder, select_cable

start = time()
ding0_grids_path = "/home/birgit/virtualenvs/elia_project/git_repos/elia_ding0_grids/176"

path = "/home/local/RL-INSTITUT/birgit.schachler/virtualenvs/edisgo_refactoring/git_repos/2021-04-19_09-18_elia_scenario_smart_charging/1574"
filename = "relative_load"
df_1 = pd.read_csv(
    os.path.join(path, "{}_1.csv".format(filename)),
    index_col=0,
    parse_dates=True
)
df_2 = pd.read_csv(
    os.path.join(path, "{}_2.csv".format(filename)),
    index_col=0,
    parse_dates=True
)
df = pd.concat([df_1, df_2], sort=True)
df.to_csv(os.path.join(path, "{}.csv".format(filename)))
# edisgo = EDisGo(
#     ding0_grid=ding0_grids_path,
#     worst_case_analysis='worst-case',
#     #generator_scenario="ego100"
# )
# a = edisgo.topology.rings()
# print("x")
# edisgo = import_edisgo_from_files(
#             ding0_grids_path,
#             import_timeseries=True,
#             import_results=False
#         )



# print("x")


# cable_data, num_parallel_cables = select_cable(edisgo, 'mv', 5.1)
# # "NA2XS2Y 3x1x150 RE/25"
# print(cable_data.name)
#
# cable_data, num_parallel_cables = select_cable(edisgo, 'mv', 40)
# # "NA2XS(FL)2Y 3x1x500 RM/35"
# print(cable_data.name)
# print(num_parallel_cables)
#
# cable_data, num_parallel_cables = select_cable(edisgo, 'lv', 0.18)
# # "NAYY 4x1x150"
# print(cable_data.name)
# print(num_parallel_cables)

# assign_feeder(edisgo, mode="mv_feeder")
# feeders = edisgo.topology.buses_df.loc[:, "mv_feeder"].dropna().unique()
# residual_load = (
#         edisgo.timeseries.generators_active_power.sum(axis=1) -
#         edisgo.timeseries.loads_active_power.sum(axis=1))
#
# for feeder in feeders:
#     # get bus with issues in feeder farthest away from station
#     # in order to start curtailment there
#     buses_in_feeder = edisgo.topology.buses_df[
#         edisgo.topology.buses_df.mv_feeder == feeder]
#     # b = buses_in_feeder.loc[
#     #     :, "path_length_to_station"].sort_values(
#     #     ascending=False).index[0]
#     #
#     # # get all generators and loads downstream
#     # buses_downstream = buses_df[
#     #     (buses_df.mv_feeder == feeder) &
#     #     (buses_df.path_length_to_station >=
#     #      buses_in_feeder.at[b, "path_length_to_station"])].index
#
#     buses_downstream = buses_in_feeder.index
#     gens_feeder = edisgo.topology.generators_df[
#         edisgo.topology.generators_df.bus.isin(
#             buses_downstream)].index
#     loads_feeder = edisgo.topology.loads_df[
#         edisgo.topology.loads_df.bus.isin(buses_downstream)].index
#     loads_feeder = loads_feeder.append(
#         edisgo.topology.charging_points_df[
#             edisgo.topology.charging_points_df.bus.isin(
#                 buses_downstream)].index)
#
#     gens_ts = edisgo.timeseries.generators_active_power.loc[
#         :, gens_feeder]
#     loads_ts = edisgo.timeseries.loads_active_power.loc[
#         :, loads_feeder]
#
#     residual_load_feeder = gens_ts.sum(axis=1) - loads_ts.sum(axis=1)
#     if any(~residual_load_feeder[
#                residual_load_feeder < 0].index.isin(
#         residual_load[residual_load < 0])):
#         print(feeder)


# timeindex = pd.date_range('2011-04-16 00:00:00', '2011-04-23 00:00:00',
#                           freq='0.25H')
# edisgo.analyze()
# print(time()-start)
# tmp = edisgo.timeseries.charging_points_active_power
# edisgo.timeseries.charging_points_active_power = pd.DataFrame(
#     data=0.0,
#     columns=tmp.columns,
#     index=tmp.index
# )

# for t in timeindex:
#     edisgo.analyze(timesteps=[t])
# network = edisgo.to_pypsa(timesteps=timeindex)
# network.lpf()
#
# now = network.snapshots[4]
#
# angle_diff = pd.Series(network.buses_t.v_ang.loc[now,network.lines.bus0].values -
#                        network.buses_t.v_ang.loc[now,network.lines.bus1].values,
#                        index=network.lines.index)
#
# (angle_diff*180/np.pi).describe()

# edisgo_elia = import_edisgo_from_files(
#             "/home/birgit/virtualenvs/Elia_Ergebnisse/177",
#             import_timeseries=True,
#             import_results=False
#         )