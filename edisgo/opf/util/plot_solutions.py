from matplotlib import pyplot as plt
import numpy as np


def plot_line_expansion(edisgo_obj, timesteps):

    pypsa_plot = edisgo_obj.to_pypsa(mode="mv", timesteps=timesteps)
    bus_index = pypsa_plot.buses.index
    pypsa_plot.buses.x.loc[
        bus_index
    ] = edisgo_obj.topology.mv_grid.buses_df.x.loc[bus_index]
    pypsa_plot.buses.y.loc[
        bus_index
    ] = edisgo_obj.topology.mv_grid.buses_df.y.loc[bus_index]
    line_colors = edisgo_obj.opf_results.lines.squeeze()
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    v = np.linspace(min(line_colors), max(line_colors), 101)
    psa_plot = pypsa_plot.plot(
        line_colors=line_colors, line_cmap=plt.cm.cool, ax=ax, geomap=False
    )
    cb = plt.colorbar(psa_plot[1])
    cb.set_clim(vmin=v[0], vmax=v[-1])
    cb.set_label("Number of installed lines")
    plt.show()

    return fig, ax
