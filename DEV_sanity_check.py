from edisgo.opf.lopf import combine_results_for_grid
import matplotlib.pyplot as plt
import pandas as pd

grid_id = 2534
feeders = [0]
res_dir = "results/minimize_loading_hp_test"
root_dir = r'H:\Grids'
grid_dir = root_dir + r'\{}'.format(grid_id)
feeder_dir = root_dir + r'\{}\feeder\{}'.format(grid_id, feeders[0])

charging_hp_el = combine_results_for_grid(feeders, grid_id, res_dir, 'charging_hp_el')
charging_tes = combine_results_for_grid(feeders, grid_id, res_dir, 'charging_tes')
energy_tes = combine_results_for_grid(feeders, grid_id, res_dir, 'energy_tes')
# add Hps
timeindex = pd.date_range("2011-01-01", periods=8760, freq="h")
cop = pd.read_csv("examples/minimum_working/COP_2011.csv").set_index(timeindex).resample("15min").ffill()
heat_demand = (
    pd.read_csv("examples/minimum_working/hp_heat_2011.csv", index_col=0)
        .set_index(timeindex)
).resample("15min").ffill()

charging_hp_el.sum(axis=1)[:672].plot(title="Charging HP week 1")
plt.figure()
charging_hp_el.sum(axis=1)[672+24:].plot(title="Charging HP week 2")
plt.figure()
charging_tes.sum(axis=1)[:672].plot(title="Charging TES week 1")
plt.figure()
charging_tes.sum(axis=1)[672+24:].plot(title="Charging TES week 2")
plt.figure()
energy_tes.sum(axis=1)[:672].plot(title="Energy TES week 1")
plt.figure()
energy_tes.sum(axis=1)[672+24:].plot(title="Energy TES week 2")

plt.figure()
results = pd.DataFrame()
results["Charging HP"] = charging_hp_el.sum(axis=1)
results["Charging TES"] = charging_tes.sum(axis=1)
results["Heat Demand"] = heat_demand["0"].loc[charging_tes.index]*134
results.iloc[:672+24].plot()
results.iloc[672+24:].plot()

plt.show()
print("SUCCESS")