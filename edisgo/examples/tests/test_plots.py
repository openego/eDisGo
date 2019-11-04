from edisgo import EDisGo, EDisGoReimport
from edisgo.tools.tools import calculate_relative_line_load


ding0_grid = 'ding0_grids__58.pkl'


def create_pypsa_test_network(ding0_grid):
    """
    Creates test csv data to use for EDisGoReimport class for given ding0 network
    with power flow results for worst cases and saves it into directory
    'test_results'.

    """
    edisgo = EDisGo(ding0_grid=ding0_grid,
                    worst_case_analysis='worst-case')
    edisgo.analyze()
    edisgo.network.results.save('test_results')


def test_histogram_line_load():
    """
    Tests histogram plot.

    """

    edisgo = EDisGoReimport('test_results')

    # timestep = None
    edisgo.histogram_relative_line_load()

    # single timestep
    edisgo.histogram_relative_line_load(
        timestep=edisgo.network.results.i_res.index[0])

    # single timestep as list
    edisgo.histogram_relative_line_load(
        timestep=[edisgo.network.results.i_res.index[0]])

    # voltage level mv
    edisgo.histogram_relative_line_load(
        timestep=[edisgo.network.results.i_res.index[0]], voltage_level='mv')


def test_histogram_voltage():
    """
    Tests histogram plot.

    """

    edisgo = EDisGoReimport('test_results')

    # timestep = None
    edisgo.histogram_voltage()

    # single timestep
    edisgo.histogram_voltage(
        timestep=edisgo.network.results.v_res().index[0])

    # single timestep as list
    edisgo.histogram_voltage(
        timestep=[edisgo.network.results.v_res().index[0]])


def test_mv_line_loading_plot():
    edisgo = EDisGoReimport('test_results')
    edisgo.plot_mv_line_loading()


def test_get_line_loading():
    """
    Tests line loading calculation.

    """
    edisgo = EDisGoReimport('test_results')

    # all time steps, all lines
    line_load = calculate_relative_line_load(
        edisgo.network.pypsa, edisgo.network.config,
        edisgo.network.results.i_res, edisgo.network.pypsa.lines.v_nom)
    print(line_load.shape)

    # single time step, all lines
    line_load = calculate_relative_line_load(
        edisgo.network.pypsa, edisgo.network.config,
        edisgo.network.results.i_res, edisgo.network.pypsa.lines.v_nom,
        timesteps=edisgo.network.results.i_res.index[0])
    print(line_load.shape)

    # single time step, all lines
    line_load = calculate_relative_line_load(
        edisgo.network.pypsa, edisgo.network.config,
        edisgo.network.results.i_res, edisgo.network.pypsa.lines.v_nom,
        timesteps=[edisgo.network.results.i_res.index[0]])
    print(line_load.shape)

    # single time step, selection of lines
    line_load = calculate_relative_line_load(
        edisgo.network.pypsa, edisgo.network.config,
        edisgo.network.results.i_res, edisgo.network.pypsa.lines.v_nom,
        timesteps=[edisgo.network.results.i_res.index[0]],
        lines=edisgo.network.pypsa.lines.index[0:2])
    print(line_load.shape)


#create_pypsa_test_network(ding0_grid)
#test_histogram_line_load()
#test_get_line_loading()
#test_histogram_voltage()
test_mv_line_loading_plot()