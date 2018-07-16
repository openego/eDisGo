from edisgo.grid.components import Storage, Line, LVStation
from edisgo.grid.grids import MVGrid
from edisgo.grid.tools import select_cable


def storage_at_hvmv_substation(mv_grid, parameters, mode=None):
    """
    Place battery at HV/MV substation bus bar.

    Parameters
    ----------
    mv_grid : :class:`~.grid.grids.MVGrid`
        MV grid instance
    parameters : :obj:`dict`
        Dictionary with storage parameters. See
        :class:`~.grid.network.StorageControl` for more information.
    mode : None or :obj:`str`
        Operational mode. See :class:`~.grid.network.StorageControl` for
        possible options and more information. Default: None.

    Returns
    -------
    :class:`~.grid.components.Storage`
        Created storage instance

    """
    storage = set_up_storage(parameters, mv_grid.station, mode)
    connect_storage(storage, mv_grid.station)
    return storage


def set_up_storage(parameters, node,
                   voltage_level=None, operational_mode=None):
    """
    Sets up a storage instance.

    Parameters
    ----------
    parameters : :obj:`dict`
        Dictionary with storage parameters. See
        :class:`~.grid.network.StorageControl` for more information.
    node : :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
        Node the storage will be connected to.
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
                   id='{}_storage_{}'.format(
                       grid,
                       len(grid.graph.nodes_by_attribute('storage')) + 1),
                   nominal_capacity=parameters['nominal_capacity'],
                   grid=grid,
                   soc_initial=parameters['soc_initial'],
                   efficiency_in=parameters['efficiency_in'],
                   efficiency_out=parameters['efficiency_out'],
                   standing_loss=parameters['standing_loss'],
                   geom=node.geom)


def connect_storage(storage, node):
    """
    Connects storage to the given node.

    Parameters
    ----------
    storage : :class:`~.grid.components.Storage`
        Storage instance to be integrated into the grid.
    node : :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
        Node the storage will be connected to.

    """

    # add storage itself to graph
    node.grid.graph.add_node(storage, type='storage')

    # add 1m connecting line to node the storage is connected to
    if isinstance(node.grid, MVGrid):
        voltage_level = 'mv'
    else:
        voltage_level = 'lv'
    line_type, line_count = select_cable(node.grid.network, voltage_level,
                                         storage.nominal_capacity)
    line = Line(
        id=storage.id,
        type=line_type,
        kind='cable',
        length=1e-3,
        grid=node.grid,
        quantity=line_count)

    node.grid.graph.add_edge(node, storage, line=line, type='line')
