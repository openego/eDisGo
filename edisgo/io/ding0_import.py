import pandas as pd
import os
import numpy as np
from pypsa import Network as PyPSANetwork

from edisgo.network.grids import MVGrid, LVGrid

if 'READTHEDOCS' not in os.environ:
    from shapely.wkt import loads as wkt_loads

import logging
logger = logging.getLogger('edisgo')

COLUMNS = {
    'buses_df': ['v_nom', 'x', 'y', 'mv_grid_id', 'lv_grid_id', 'in_building'],
    'generators_df': ['bus', 'control', 'p_nom', 'type', 'subtype',
                      'weather_cell_id'],
    'loads_df': ['bus', 'peak_load', 'sector', 'annual_consumption'],
    'transformers_df': ['bus0', 'bus1', 'x_pu', 'r_pu', 's_nom',
                        'type_info'],
    'lines_df': ['bus0', 'bus1', 'length', 'x', 'r', 's_nom', 'type_info',
                 'kind', 'num_parallel'],
    'switches_df': ['bus_open', 'bus_closed', 'branch', 'type_info'],
    'storage_units_df': []
}


def import_ding0_grid(path, edisgo_obj):
    """
    Import an eDisGo network topology from
    `Ding0 data <https://github.com/openego/ding0>`_.

    This import method is specifically designed to load network topology data in
    the format as `Ding0 <https://github.com/openego/ding0>`_ provides it via
    csv files.

    Parameters
    ----------
    path: :obj:`str`
        path to ding0 network csv files
    edisgo_obj: :class:`~.network.edisgo_obj.Network`
        The eDisGo data container object

    """

    def sort_transformer_buses(transformers_df):
        """
        Sort buses of inserted transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        voltage_bus0 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus0].v_nom.values
        voltage_bus1 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus1].v_nom.values
        transformers_df.loc[voltage_bus1 > voltage_bus0, ['bus0', 'bus1']] = \
            transformers_df.loc[voltage_bus1 > voltage_bus0,
                                ['bus1', 'bus0']].values
        return transformers_df

    def sort_hvmv_transformer_buses(transformers_df):
        """
        Sort buses of inserted HV/MV transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        for transformer in transformers_df.index:
            if not transformers_df.loc[transformer, 'bus1'] in \
                   edisgo_obj.topology.buses_df.index:
                transformers_df.loc[transformer, ['bus0', 'bus1']] = \
                    transformers_df.loc[transformer, ['bus1', 'bus0']].values
        return transformers_df

    grid = PyPSANetwork()
    grid.import_from_csv_folder(path)

    # write dataframes to edisgo_obj
    edisgo_obj.topology.buses_df = grid.buses[COLUMNS['buses_df']]
    # drop slack generator from generators
    slack = [_ for _ in grid.generators.index if 'slack' in _.lower()][0]
    grid.generators.drop(index=[slack], inplace=True)

    edisgo_obj.topology.generators_df = grid.generators[
        COLUMNS['generators_df']]
    edisgo_obj.topology.loads_df = grid.loads[COLUMNS['loads_df']]
    edisgo_obj.topology.transformers_df = sort_transformer_buses(
        grid.transformers.drop(labels=['x_pu','r_pu'], axis=1).rename(
        columns={'r': 'r_pu', 'x': 'x_pu'})[COLUMNS['transformers_df']])
    edisgo_obj.topology.transformers_hvmv_df = sort_hvmv_transformer_buses(
        pd.read_csv(os.path.join(path, 'transformers_hvmv.csv'),
                    index_col=[0]).rename(
            columns={'r': 'r_pu', 'x': 'x_pu'}))
    edisgo_obj.topology.lines_df = grid.lines[COLUMNS['lines_df']]
    edisgo_obj.topology.switches_df = pd.read_csv(
        os.path.join(path, 'switches.csv'), index_col=[0])
    edisgo_obj.topology.storage_units_df = grid.storage_units
    edisgo_obj.topology.grid_district = {
        'population': grid.mv_grid_district_population,
        'geom': wkt_loads(grid.mv_grid_district_geom),
        'srid': grid.srid}

    edisgo_obj.topology._grids = {}

    # set up medium voltage grid
    mv_grid_id = list(set(grid.buses.mv_grid_id))[0]
    edisgo_obj.topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)
    edisgo_obj.topology._grids[str(edisgo_obj.topology.mv_grid)] = \
        edisgo_obj.topology.mv_grid

    # set up low voltage grids
    lv_grid_ids = set(grid.buses.lv_grid_id.dropna())
    for lv_grid_id in lv_grid_ids:
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_obj)
        edisgo_obj.topology.mv_grid._lv_grids.append(lv_grid)
        edisgo_obj.topology._grids[str(lv_grid)] = lv_grid

    # Check data integrity
    _validate_ding0_grid_import(edisgo_obj.topology)


def _validate_ding0_grid_import(topology):
    """
    Check imported data integrity. Checks for duplicated labels and not
    connected components.
    Todo: Check with meth:`_check_integrity_of_pypsa` in pypsa_io

    Parameters
    ----------
    topology: class:`~.network.topology.Topology`
        topology class containing mv and lv grids

    """
    # check for duplicate labels (of components)
    duplicated_labels = []
    if any(topology.buses_df.index.duplicated()):
        duplicated_labels.append(
            topology.buses_df.index[topology.buses_df.index.duplicated()].values)
    if any(topology.generators_df.index.duplicated()):
        duplicated_labels.append(
            topology.generators_df.index[
                topology.generators_df.index.duplicated()].values)
    if any(topology.loads_df.index.duplicated()):
        duplicated_labels.append(
            topology.loads_df.index[topology.loads_df.index.duplicated()].values)
    if any(topology.transformers_df.index.duplicated()):
        duplicated_labels.append(
            topology.transformers_df.index[
                topology.transformers_df.index.duplicated()].values)
    if any(topology.lines_df.index.duplicated()):
        duplicated_labels.append(
            topology.lines_df.index[topology.lines_df.index.duplicated()].values)
    if any(topology.switches_df.index.duplicated()):
        duplicated_labels.append(
            topology.switches_df.index[
                topology.switches_df.index.duplicated()].values)
    if duplicated_labels:
        raise ValueError(
            "{labels} have duplicate entry in one of the components "
            "dataframes.".format(labels=', '.join(
                np.concatenate([list.tolist() for list in duplicated_labels])))
        )

    # check for isolated or not defined buses
    buses = []

    for nodal_component in ["loads", "generators", "storage_units"]:
        df = getattr(topology, nodal_component + "_df")
        missing = df.index[~df.bus.isin(topology.buses_df.index)]
        buses.append(df.bus.values)
        if len(missing) > 0:
            raise ValueError(
                "The following {} have buses which are not defined: "
                "{}.".format(
                    nodal_component, ', '.join(missing.values)))

    for branch_component in ["lines", "transformers"]:
        df = getattr(topology, branch_component + "_df")
        for attr in ["bus0", "bus1"]:
            buses.append(df[attr].values)
            missing = df.index[~df[attr].isin(topology.buses_df.index)]
            if len(missing) > 0:
                raise ValueError(
                    "The following {} have {} which are not defined: "
                    "{}.".format(
                        branch_component, attr, ', '.join(missing.values)))

    for attr in ["bus_open", "bus_closed"]:
        missing = topology.switches_df.index[
            ~topology.switches_df[attr].isin(topology.buses_df.index)]
        buses.append(topology.switches_df[attr].values)
        if len(missing) > 0:
            raise ValueError(
                "The following switches have {} which are not defined: "
                "{}.".format(
                    attr, ', '.join(missing.values)))

    all_buses = np.unique(np.concatenate(buses, axis=None))
    missing = topology.buses_df.index[~topology.buses_df.index.isin(all_buses)]
    if len(missing) > 0:
        raise ValueError("The following buses are isolated: {}.".format(
            ', '.join(missing.values)))
