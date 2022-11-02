from __future__ import annotations

import logging
import os

from typing import TYPE_CHECKING

import pandas as pd
import saio

# from sqlalchemy import func
from sqlalchemy.engine.base import Engine

from edisgo.io.egon_data_import import session_scope
from edisgo.tools.tools import mv_grid_gdf

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def dsm_from_database(
    edisgo_obj: EDisGo,
    engine: Engine,
    scenario: str = "eGon2035",
):
    saio.register_schema("grid", engine)

    from saio.grid import (
        egon_etrago_link,
        egon_etrago_link_timeseries,
        egon_etrago_store,
        egon_etrago_store_timeseries,
        egon_hvmv_substation,
    )

    grid_gdf = mv_grid_gdf(edisgo_obj)

    with session_scope(engine) as session:
        query = session.query(
            egon_hvmv_substation.bus_id,
            egon_hvmv_substation.point.label("geom"),
        )  # TODO: wie muss die querry richtig lauten?
        #     .filter(
        #     func.ST_Within(
        #         grid_gdf.geometry.iat[0],
        #         egon_hvmv_substation.point,
        #     )
        # )

        gdf = gpd.read_postgis(
            sql=query.statement, con=query.session.bind, crs=grid_gdf.crs
        )

    bus_ids = gdf.loc[gdf.geometry.within(grid_gdf.geometry.iat[0])].bus_id

    if len(bus_ids) == 0:
        raise ImportError(
            f"There is no egon-data substation within the open_ego grid "
            f"{edisgo_obj.topology.id}. Cannot import any DSM information."
        )

    # pick one randomly (if more than one)
    bus_id = bus_ids.iat[0]

    with session_scope(engine) as session:
        query = session.query(egon_etrago_link).filter(
            egon_etrago_link.scn_name == "eGon2035",
            egon_etrago_link.carrier == "dsm",
            egon_etrago_link.bus0 == bus_id,
        )

        edisgo_obj.dsm.egon_etrago_link = pd.read_sql(
            sql=query.statement, con=query.session.bind
        )

    link_id = edisgo_obj.dsm.egon_etrago_link.at[0, "link_id"]
    store_bus_id = edisgo_obj.dsm.egon_etrago_link.at[0, "bus1"]

    with session_scope(engine) as session:
        query = session.query(egon_etrago_link_timeseries).filter(
            egon_etrago_link_timeseries.scn_name == scenario,
            egon_etrago_link_timeseries.link_id == link_id,
        )

        edisgo_obj.dsm.egon_etrago_link_timeseries = pd.read_sql(
            sql=query.statement, con=query.session.bind
        )

    time_series_data = {
        "p_min_pu": edisgo_obj.dsm.egon_etrago_link_timeseries.at[0, "p_min_pu"],
        "p_max_pu": edisgo_obj.dsm.egon_etrago_link_timeseries.at[0, "p_max_pu"],
        "p_set": edisgo_obj.dsm.egon_etrago_link_timeseries.at[0, "p_set"],
    }

    if time_series_data["p_set"] is None:
        logger.warning(
            "DSM time series for p_set is missing. Using the mean from p_min_pu and "
            "p_max_pu as fallback."
        )

        time_series_data["p_set"] = [
            (p_min_pu + p_max_pu) / 2
            for p_min_pu, p_max_pu in zip(
                time_series_data["p_min_pu"], time_series_data["p_max_pu"]
            )
        ]

    if len(edisgo_obj.timeseries.timeindex) != len(time_series_data["p_min_pu"]):
        raise IndexError(
            f"The length of the time series of the edisgo object ("
            f"{len(edisgo_obj.timeseries.timeindex)}) and the database ("
            f"{len(time_series_data['p_min_pu'])}) do not match. Adjust the length of "
            f"the time series of the edisgo object accordingly."
        )

    edisgo_obj.dsm.dsm_time_series = pd.DataFrame(
        time_series_data, index=edisgo_obj.timeseries.timeindex
    )

    with session_scope(engine) as session:
        query = session.query(egon_etrago_store).filter(
            egon_etrago_store.scn_name == scenario,
            egon_etrago_store.carrier == "dsm",
            egon_etrago_store.bus == store_bus_id,
        )

        edisgo_obj.dsm.egon_etrago_store = pd.read_sql(
            sql=query.statement, con=query.session.bind
        )

    store_id = edisgo_obj.dsm.egon_etrago_store.at[0, "store_id"]
    e_nom = edisgo_obj.dsm.egon_etrago_store.at[0, "e_nom"]

    with session_scope(engine) as session:
        query = session.query(egon_etrago_store_timeseries).filter(
            egon_etrago_store_timeseries.scn_name == scenario,
            egon_etrago_store_timeseries.store_id == store_id,
        )

        edisgo_obj.dsm.egon_etrago_store_timeseries = pd.read_sql(
            sql=query.statement, con=query.session.bind
        )

    time_series_data = {
        "e_min": edisgo_obj.dsm.egon_etrago_store_timeseries.at[0, "e_min_pu"],
        "e_max": edisgo_obj.dsm.egon_etrago_store_timeseries.at[0, "e_max_pu"],
    }

    edisgo_obj.dsm.store_time_series = pd.DataFrame(
        time_series_data, index=edisgo_obj.timeseries.timeindex
    ).mul(e_nom)
