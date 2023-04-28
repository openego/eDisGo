import os

import pandas as pd

from pypsa import Network as PyPSANetwork

from edisgo.network.grids import MVGrid

if "READTHEDOCS" not in os.environ:
    from shapely.wkt import loads as wkt_loads

import logging

logger = logging.getLogger(__name__)


def import_ding0_grid(path, edisgo_obj, legacy_ding0_grids=True):
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
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo data container object.
    legacy_ding0_grids : bool
        Allow import of old ding0 grids. Default: True.

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
    if legacy_ding0_grids:
        logger.debug("Use ding0 legacy grid import.")
        # rename column peak_load to p_set
        grid.loads = grid.loads.drop(columns="p_set").rename(
            columns={"peak_load": "p_set"}
        )
        # set loads without type information to be conventional loads
        # this is done, as older ding0 versions do not provide information on the type
        # of load and can be done as these ding0 grids only contain conventional loads
        loads_without_type = grid.loads[
            (grid.loads.type.isnull()) | (grid.loads.type == "")
        ].index
        grid.loads.loc[loads_without_type, "type"] = "conventional_load"
        # rename retail to cts, as it is in newer ding0 versions called cts
        grid.loads.replace(to_replace=["retail"], value="cts", inplace=True)
        # set up columns that are added in new ding0 version
        grid.loads["building_id"] = None
        grid.loads["number_households"] = None
        grid.generators["source_id"] = None
    else:
        edisgo_obj.topology.buses_df["in_building"] = False
        grid.generators = grid.generators.rename(columns={"gens_id": "source_id"})
    edisgo_obj.topology.loads_df = grid.loads[edisgo_obj.topology.loads_df.columns]
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

    # set up medium voltage grid
    mv_grid_id = list(set(grid.buses.mv_grid_id))[0]
    edisgo_obj.topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)

    # check data integrity
    edisgo_obj.topology.check_integrity()
