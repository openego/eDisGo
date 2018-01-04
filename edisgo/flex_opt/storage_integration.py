from edisgo.grid.components import Storage, Line
from edisgo.grid.tools import select_cable

import logging


def integrate_storage(network, position, operational_mode, parameters):
    """
    Integrate storage units in the grid and specify its operational mode

    Parameters
    ----------
    network: :class:`~.grid.network.Network`
        The eDisGo container object
    position : str
        Specify storage location. Available options are

        * 'hvmv_substation_busbar': places a storage unit directly at the
          HV/MV station's bus bar, see :func:`storage_at_hvmv_substation`
    operational_mode : str
        Operational mode. See :class:`~.grid.components.StorageOperation for
        possible options and more information.
    parameters : dict
        Parameters specifying characteristics of storage in detail
        The format looks like the following example and requires given
        parameters

        .. code-block:: python

            {
                'nominal_capacity': <float>, # in kWh
                'soc_initial': <float>, # in kWh
                'efficiency_in': <float>, # in per unit 0..1
                'efficiency_out': <float>, # in per unit 0..1
                'standing_loss': <float> # in per unit 0..1
            }

    """

    if position == 'hvmv_substation_busbar':
        storage_at_hvmv_substation(network.mv_grid, parameters,
                                   operational_mode)
    else:
        logging.error("{} is not a valid storage positioning mode".format(
            position))
        raise ValueError("Unknown parameter for storage positioning: {} is "
                         "not a valid storage positioning mode".format(
            position))


def storage_at_hvmv_substation(mv_grid, parameters, mode):
    """
    Place 1 MVA battery at HV/MV substation bus bar

    As this is currently a dummy implementation the storage operation is as
    simple as follows:

     * Feedin > 50 % -> charge at full power
     * Feedin < 50 % -> discharge at full power

    Parameters
    ----------
    mv_grid : :class:`~.grid.grids.MVGrid`
        MV grid instance
    parameters : dict
        Parameters specifying characteristics of storage in detail
    mode : str
        Operational mode. See :class:`~.grid.components.StorageOperation for
        possible options and more information.
    """

    # define storage instance and define it's operational mode
    storage_id = len(mv_grid.graph.nodes_by_attribute('storage')) + 1
    storage = Storage(operation={'mode': mode},
                      id=storage_id,
                      nominal_capacity=parameters['nominal_capacity'],
                      grid=mv_grid,
                      soc_initial=parameters['soc_initial'],
                      efficiency_in=parameters['efficiency_in'],
                      efficiency_out=parameters['efficiency_out'],
                      standing_loss=parameters['standing_loss'],
                      geom=mv_grid.station.geom)

    # add storage itself to graph
    mv_grid.graph.add_node(storage, type='storage')

    # add 1m connecting line to hv/mv substation bus bar
    line_type, _ = select_cable(mv_grid.network, 'mv',
                                storage.nominal_capacity)
    line = Line(
        id=storage_id,
        type=line_type,
        kind='cable',
        length=1e-3,
        grid=mv_grid)

    mv_grid.graph.add_edge(mv_grid.station, storage, line=line, type='line')