from edisgo.opf.lopf import combine_results_for_grid
import matplotlib.pyplot as plt
import pandas as pd

grid_id = 2534
feeders = [3]
res_dir = "results/minimize_loading_SEST"
root_dir = r'H:\Grids'
grid_dir = root_dir + r'\{}'.format(grid_id)
feeder_dir = root_dir + r'\{}\feeder\{}'.format(grid_id, feeders[0])

x_charge_ev_grid = combine_results_for_grid(feeders, grid_id, res_dir, 'x_charge_ev')
reference_charging = pd.read_csv(feeder_dir + "/timeseries/charging_points_active_power.csv",
                                 index_col=0, parse_dates=True)
x_charge_ev_grid = x_charge_ev_grid.loc[reference_charging.index]

x_charge_ev_grid.sum(axis=1).plot()
reference_charging.sum(axis=1).plot()
plt.figure()
x_charge_ev_grid.sum(axis=1)[:672].plot()
reference_charging.sum(axis=1)[:672].plot()
plt.figure()
x_charge_ev_grid.sum(axis=1)[672:].plot()
reference_charging.sum(axis=1)[672:].plot()

print("Charged energy:\n Reference: {} MWh, Optimised: {} MWh".format(
    reference_charging.sum().sum(), x_charge_ev_grid.sum().sum()))

plt.show()
print("SUCCESS")