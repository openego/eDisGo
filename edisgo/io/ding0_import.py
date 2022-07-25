import os

import pandas as pd

from pypsa import Network as PyPSANetwork

from edisgo.network.grids import LVGrid, MVGrid

if "READTHEDOCS" not in os.environ:
    from shapely.wkt import loads as wkt_loads

import logging

logger = logging.getLogger(__name__)


def import_ding0_grid(path, edisgo_obj):
    """
    Import an eDisGo network topology from
    `Ding0 data <https://github.com/openego/ding0>`_.

    This import method is specifically designed to load network topology data in
    the format as `Ding0 <https://github.com/openego/ding0>`_ provides it via
    csv files.

    Parameters
    ----------
    path : str
        Path to ding0 network csv files.
    edisgo_obj: :class:`~.EDisGo`
        The eDisGo data container object.

    """

    def sort_transformer_buses(transformers_df):
        """
        Sort buses of inserted transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        voltage_bus0 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus0
        ].v_nom.values
        voltage_bus1 = edisgo_obj.topology.buses_df.loc[
            transformers_df.bus1
        ].v_nom.values
        transformers_df.loc[
            voltage_bus1 > voltage_bus0, ["bus0", "bus1"]
        ] = transformers_df.loc[voltage_bus1 > voltage_bus0, ["bus1", "bus0"]].values
        return transformers_df

    def sort_hvmv_transformer_buses(transformers_df):
        """
        Sort buses of inserted HV/MV transformers in a way that bus1 always
        represents secondary side of transformer.
        """
        for transformer in transformers_df.index:
            if (
                transformers_df.at[transformer, "bus1"]
                in edisgo_obj.topology.buses_df.index
            ):
                continue

            transformers_df.loc[transformer, ["bus0", "bus1"]] = transformers_df.loc[
                transformer, ["bus1", "bus0"]
            ].values

        return transformers_df

    grid = PyPSANetwork()
    grid.import_from_csv_folder(path)

    # write dataframes to edisgo_obj
    edisgo_obj.topology.buses_df = grid.buses[edisgo_obj.topology.buses_df.columns]
    edisgo_obj.topology.lines_df = grid.lines[edisgo_obj.topology.lines_df.columns]

    grid.loads = grid.loads.drop(columns="p_set").rename(columns={"peak_load": "p_set"})

    edisgo_obj.topology.loads_df = grid.loads[edisgo_obj.topology.loads_df.columns]
    # set loads without type information to be conventional loads
    # this is done, as ding0 currently does not provide information on the type of load
    # but ding0 grids currently also only contain conventional loads
    # ToDo: Change, once information is provided by ding0
    loads_without_type = edisgo_obj.topology.loads_df[
        (edisgo_obj.topology.loads_df.type.isnull())
        | (edisgo_obj.topology.loads_df.type == "")
    ].index
    edisgo_obj.topology.loads_df.loc[loads_without_type, "type"] = "conventional_load"
    # drop slack generator from generators
    slack = grid.generators.loc[grid.generators.control == "Slack"].index
    grid.generators.drop(slack, inplace=True)
    edisgo_obj.topology.generators_df = grid.generators[
        edisgo_obj.topology.generators_df.columns
    ]
    edisgo_obj.topology.storage_units_df = grid.storage_units[
        edisgo_obj.topology.storage_units_df.columns
    ]
    edisgo_obj.topology.transformers_df = sort_transformer_buses(
        grid.transformers.drop(labels=["x_pu", "r_pu"], axis=1).rename(
            columns={"r": "r_pu", "x": "x_pu"}
        )[edisgo_obj.topology.transformers_df.columns]
    )
    edisgo_obj.topology.transformers_hvmv_df = sort_hvmv_transformer_buses(
        pd.read_csv(os.path.join(path, "transformers_hvmv.csv"), index_col=[0]).rename(
            columns={"r": "r_pu", "x": "x_pu"}
        )
    )
    edisgo_obj.topology.switches_df = pd.read_csv(
        os.path.join(path, "switches.csv"), index_col=[0]
    )

    edisgo_obj.topology.grid_district = {
        "population": grid.mv_grid_district_population,
        "geom": wkt_loads(grid.mv_grid_district_geom),
        "srid": grid.srid,
    }

    edisgo_obj.topology._grids = {}

    # set up medium voltage grid
    mv_grid_id = list(set(grid.buses.mv_grid_id))[0]
    edisgo_obj.topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)
    edisgo_obj.topology._grids[
        str(edisgo_obj.topology.mv_grid)
    ] = edisgo_obj.topology.mv_grid

    # set up low voltage grids
    lv_grid_ids = set(grid.buses.lv_grid_id.dropna())
    for lv_grid_id in lv_grid_ids:
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_obj)
        edisgo_obj.topology.mv_grid._lv_grids.append(lv_grid)
        edisgo_obj.topology._grids[str(lv_grid)] = lv_grid

    # Check data integrity
    edisgo_obj.topology.check_integrity()
