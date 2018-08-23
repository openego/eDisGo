from math import sqrt
from edisgo.grid.components import Storage, Line, LVStation
from edisgo.grid.grids import MVGrid
from edisgo.grid.tools import select_cable


def storage_at_hvmv_substation(mv_grid, parameters, mode=None):
    """
    Place storage at HV/MV substation bus bar.

    Parameters
    ----------
    mv_grid : :class:`~.grid.grids.MVGrid`
        MV grid instance
    parameters : :obj:`dict`
        Dictionary with storage parameters. Must at least contain
        'nominal_power'. See :class:`~.grid.network.StorageControl` for more
        information.
    mode : :obj:`str`, optional
        Operational mode. See :class:`~.grid.network.StorageControl` for
        possible options and more information. Default: None.

    Returns
    -------
    :class:`~.grid.components.Storage`, :class:`~.grid.components.Line`
        Created storage instance and newly added line to connect storage.

    """
    storage = set_up_storage(node=mv_grid.station, parameters=parameters,
                             operational_mode=mode)
    line = connect_storage(storage, mv_grid.station)
    return storage, line


def set_up_storage(node, parameters,
                   voltage_level=None, operational_mode=None):
    """
    Sets up a storage instance.

    Parameters
    ----------
    node : :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
        Node the storage will be connected to.
    parameters : :obj:`dict`, optional
        Dictionary with storage parameters. Must at least contain
        'nominal_power'. See :class:`~.grid.network.StorageControl` for more
        information.
    voltage_level : :obj:`str`, optional
        This parameter only needs to be provided if `node` is of type
        :class:`~.grid.components.LVStation`. In that case `voltage_level`
        defines which side of the LV station the storage is connected to. Valid
        options are 'lv' and 'mv'. Default: None.
    operational_mode : :obj:`str`, optional
        Operational mode. See :class:`~.grid.network.StorageControl` for
        possible options and more information. Default: None.

    """

    # if node the storage is connected to is an LVStation voltage_level
    # defines which side the storage is connected to
    if isinstance(node, LVStation):
        if voltage_level == 'lv':
            grid = node.grid
        elif voltage_level == 'mv':
            grid = node.mv_grid
        else:
            raise ValueError(
                "{} is not a valid option for voltage_level.".format(
                    voltage_level))
    else:
        grid = node.grid

    return Storage(operation=operational_mode,
                   id='{}_storage_{}'.format(grid,
                                             len(grid.graph.nodes_by_attribute(
                                                 'storage')) + 1),
                   grid=grid,
                   geom=node.geom,
                   **parameters)


def connect_storage(storage, node):
    """
    Connects storage to the given node.

    The storage is connected by a cable
    The cable the storage is connected with is selected to be able to carry
    the storages nominal power and equal amount of reactive power.
    No load factor is considered.

    Parameters
    ----------
    storage : :class:`~.grid.components.Storage`
        Storage instance to be integrated into the grid.
    node : :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
        Node the storage will be connected to.

    Returns
    -------
    :class:`~.grid.components.Line`
        Newly added line to connect storage.

    """

    # add storage itself to graph
    storage.grid.graph.add_node(storage, type='storage')

    # add 1m connecting line to node the storage is connected to
    if isinstance(storage.grid, MVGrid):
        voltage_level = 'mv'
    else:
        voltage_level = 'lv'

    # necessary apparent power the line must be able to carry is set to be
    # the storages nominal power and equal amount of reactive power devided by
    # the minimum load factor
    lf_dict = storage.grid.network.config['grid_expansion_load_factors']
    lf = min(lf_dict['{}_feedin_case_line'.format(voltage_level)],
             lf_dict['{}_load_case_line'.format(voltage_level)])
    apparent_power_line = sqrt(2) * storage.nominal_power / lf
    line_type, line_count = select_cable(storage.grid.network, voltage_level,
                                         apparent_power_line)
    line = Line(
        id=storage.id,
        type=line_type,
        kind='cable',
        length=1e-3,
        grid=storage.grid,
        quantity=line_count)

    storage.grid.graph.add_edge(node, storage, line=line, type='line')

    return line
