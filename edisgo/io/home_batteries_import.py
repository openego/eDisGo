from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import saio

from sqlalchemy import func
from sqlalchemy.engine.base import Engine

from edisgo.io.egon_data_import import (
    get_srid_of_db_table,
    session_scope_egon_data,
    sql_grid_geom,
    sql_within,
)
from edisgo.tools.geo import mv_grid_gdf

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def home_batteries_from_database(
    edisgo_obj: EDisGo,
    engine: Engine,
    scenario: str = "eGon2035",
    remove_existing: bool = True,
):
    batteries_gdf = get_home_batteries_from_database(
        edisgo_obj=edisgo_obj, engine=engine, scenario=scenario
    )

    if remove_existing:
        remove_existing_storages(edisgo_obj=edisgo_obj)

    generators_df = edisgo_obj.topology.generators_df.copy()

    batteries_gdf = batteries_gdf.merge(
        right=generators_df[["bus", "building_id"]], how="left", on="building_id"
    )

    if batteries_gdf.bus.isna().any():
        raise LookupError(
            f"The following batteries don't have a matching generator. Please make sure"
            f" to import all generators of the scenario first. Batteries missing "
            f"generator: {batteries_gdf.loc[batteries_gdf.bus.isna()].index.tolist()}"
        )

    cols_to_iterate = [
        "p_nom",
        "bus",
    ]

    for index, p_nom, bus in batteries_gdf[cols_to_iterate].itertuples():
        edisgo_obj.add_component(
            comp_type="storage_unit",
            p_nom=p_nom,
            bus=bus,
            egon_id=index,
        )


def remove_existing_storages(edisgo_obj: EDisGo):
    storage_units_df = edisgo_obj.topology.storage_units_df.copy()

    for name in storage_units_df.index:
        edisgo_obj.remove_component(comp_type="storage_unit", comp_name=name)


def get_home_batteries_from_database(
    edisgo_obj: EDisGo, engine: Engine, scenario: str = "eGon2035"
):
    saio.register_schema("supply", engine)
    saio.register_schema("openstreetmap", engine)

    from saio.openstreetmap import osm_buildings_filtered
    from saio.supply import egon_home_batteries

    sql_geom = sql_grid_geom(edisgo_obj)
    crs = mv_grid_gdf(edisgo_obj).crs

    with session_scope_egon_data(engine) as session:
        srid = get_srid_of_db_table(session, osm_buildings_filtered.geom_point)

        query = session.query(
            func.ST_Transform(
                osm_buildings_filtered.geom_point,
                srid,
            ).label("geom"),
            osm_buildings_filtered.id,
        ).filter(
            sql_within(osm_buildings_filtered.geom_point, sql_geom, srid),
        )

        buildings_gdf = gpd.read_postgis(
            sql=query.statement, con=query.session.bind, crs=f"EPSG:{srid}"
        ).to_crs(crs)

    building_ids = buildings_gdf.id

    with session_scope_egon_data(engine) as session:
        query = (
            session.query(egon_home_batteries)
            .filter(
                egon_home_batteries.scenario == scenario,
                egon_home_batteries.building_id.in_(building_ids),
            )
            .order_by(egon_home_batteries.index)
        )

        batteries_df = pd.read_sql(
            sql=query.statement, con=query.session.bind, index_col="index"
        )

    return gpd.GeoDataFrame(
        batteries_df.merge(
            buildings_gdf, how="left", left_on="building_id", right_on="id"
        ).drop(columns=["id"]),
        geometry="geom",
        crs=buildings_gdf.crs,
    )
