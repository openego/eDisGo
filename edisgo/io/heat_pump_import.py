# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import random

import numpy as np
import pandas as pd

from edisgo.io import db
from edisgo.tools.config import Config
from edisgo.tools.tools import (
    determine_bus_voltage_level,
    determine_grid_integration_voltage_level,
)

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

    from shapely.geometry import Point

logger = logging.getLogger(__name__)


def _query_individual_heat_pumps_oedb(
    edisgo_object, scenario, engine, query_limit=None
):
    """
    Get heat pumps for individual heating per building from oedb.

    Weather cell ID is as well added in this function.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing installed heat pump capacity for all individual heat
        pumps in the grid per building.
        For more information see parameter `hp_individual` in
        :func:`~.io.heat_pump_import._grid_integration`.

    """
    config = Config()
    (egon_hp_capacity_buildings,) = config.import_tables_from_oep(
        engine, ["egon_hp_capacity_buildings"], "demand"
    )
    egon_map_zensus_mvgd_buildings, egon_map_zensus_weather_cell = (
        config.import_tables_from_oep(
            engine,
            ["egon_map_zensus_mvgd_buildings", "egon_map_zensus_weather_cell"],
            "boundaries",
        )
    )
    building_ids = edisgo_object.topology.loads_df.building_id.unique()
    with db.session_scope_egon_data(engine) as session:
        query = (
            session.query(
                egon_hp_capacity_buildings.building_id,
                egon_hp_capacity_buildings.hp_capacity.label("p_set"),
                egon_map_zensus_weather_cell.w_id.label("weather_cell_id"),
            )
            .filter(
                egon_hp_capacity_buildings.scenario == scenario,
                egon_hp_capacity_buildings.building_id.in_(building_ids),
                egon_hp_capacity_buildings.hp_capacity
                <= edisgo_object.config["grid_connection"][
                    "upper_limit_voltage_level_4"
                ],
            )
            .outerjoin(  # join to obtain zensus cell ID
                egon_map_zensus_mvgd_buildings,
                egon_hp_capacity_buildings.building_id
                == egon_map_zensus_mvgd_buildings.building_id,
            )
            .outerjoin(  # join to obtain weather cell ID corresponding to zensus cell
                egon_map_zensus_weather_cell,
                egon_map_zensus_mvgd_buildings.zensus_population_id
                == egon_map_zensus_weather_cell.zensus_population_id,
            )
        )
        if query_limit is not None:
            query = query.limit(query_limit)

        df = pd.read_sql(query.statement, engine, index_col=None)

        # drop duplicated building IDs that exist because
        # egon_map_zensus_mvgd_buildings can contain several entries per building,
        # e.g. for CTS and residential
        return df.drop_duplicates(subset=["building_id"])


def _query_central_heat_sources_oedb(
    edisgo_object, scenario, engine, carrier, query_limit=None
):
    """
    Retrieve central heat-source records and raw spatial data from the database.

    Shared retrieval helper for central heat pumps and resistive heaters. CRS
    conversion, spatial assignments and location fallbacks are performed by
    :func:`_prepare_central_heat_sources`, not by this query helper.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
        Provides the MV grid ID used to select plants, the grid-district CRS
        assigned to weather-cell geometries and the maximum plant capacity from
        ``config["grid_connection"]["upper_limit_voltage_level_4"]``.
    scenario : str
        Scenario for which to retrieve data, e.g. "eGon2035" or "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine for the OEP or an eGon-data database.
    carrier : str
        Technology to select from the eTraGo links: "central_heat_pump" or
        "central_resistive_heater".
    query_limit : int or None, optional
        Maximum number of plant records to retrieve, intended for reduced live
        tests. Does not limit the spatial support queries. Default: None, meaning
        no explicit row limit.

    Returns
    -------
    dict
        Raw tables with the following keys:

        * ``plants`` : pandas.DataFrame
            Plant records with ``bus0`` (electricity bus/MV grid ID), ``bus1``
            (heat bus ID) and ``p_set`` (nominal electrical capacity in MW).
        * ``heat_buses`` : geopandas.GeoDataFrame
            Heat buses referenced by the plants, with ``bus1``, ``geom`` and
            ``scn_name``.
        * ``weather_cells`` : geopandas.GeoDataFrame
            Weather cells with ``weather_cell_id`` and ``geom``.
        * ``district_heating_areas`` : geopandas.GeoDataFrame
            Areas for the scenario, with ``district_heating_id``, ``area_id``,
            ``geom_polygon`` and ``scenario``.
        * ``chp_plants`` : geopandas.GeoDataFrame
            CHP plants for the scenario, with ``district_heating_id`` and
            ``geom``, used later to select heat-source locations.

        If no plants are found, only ``plants`` is returned as an empty
        DataFrame; spatial support queries are skipped. Geometries are not
        reprojected here. Heat buses, district-heating areas and CHP plants use
        their database SRIDs; weather cells are assigned the grid-district CRS.
    """
    config = Config()
    (egon_district_heating_areas,) = config.import_tables_from_oep(
        engine, ["egon_district_heating_areas"], "demand"
    )
    egon_district_heating, egon_era5_weather_cells = config.import_tables_from_oep(
        engine, ["egon_district_heating", "egon_era5_weather_cells"], "supply"
    )
    # Local eGon databases store eTraGo tables in "grid"; the OEP resolves
    # the aliases via "supply".
    etrago_schema = "supply" if "openenergyplatform" in str(engine.url) else "grid"
    egon_etrago_bus, egon_etrago_link = config.import_tables_from_oep(
        engine, ["egon_etrago_bus", "egon_etrago_link"], etrago_schema
    )
    with db.session_scope_egon_data(engine) as session:
        # get heat pumps / resistive heaters in the grid
        query = session.query(
            egon_etrago_link.bus0,
            egon_etrago_link.bus1,
            egon_etrago_link.p_nom.label("p_set"),
        ).filter(
            egon_etrago_link.scn_name == scenario,
            egon_etrago_link.bus0 == edisgo_object.topology.id,
            egon_etrago_link.carrier == carrier,
            egon_etrago_link.p_nom
            <= edisgo_object.config["grid_connection"]["upper_limit_voltage_level_4"],
        )
        if query_limit is not None:
            query = query.limit(query_limit)
        df = pd.read_sql(query.statement, engine, index_col=None)
        if not df.empty:
            # Query for egon_etrago_bus
            srid_etrago_bus = db.get_srid_of_db_table(session, egon_etrago_bus.geom)

            query_etrago_bus = session.query(
                egon_etrago_bus.bus_id.label("bus1"),
                egon_etrago_bus.geom,
                egon_etrago_bus.scn_name,
            ).filter(
                egon_etrago_bus.scn_name == scenario,
                egon_etrago_bus.bus_id.in_(df.bus1),
            )
            gdf_etrago_bus = gpd.read_postgis(
                query_etrago_bus.statement,
                engine,
                index_col=None,
                crs=f"EPSG:{srid_etrago_bus}",
            )

            # Query for egon_era5_weather_cells
            query_weather_cells = session.query(
                egon_era5_weather_cells.w_id.label("weather_cell_id"),
                egon_era5_weather_cells.geom,
            )
            gdf_weather_cells = gpd.read_postgis(
                query_weather_cells.statement,
                engine,
                index_col=None,
                crs=f"EPSG:{edisgo_object.topology.grid_district['srid']}",
            )

            # Query for egon_district_heating_areas
            query_district_heating_areas = session.query(
                egon_district_heating_areas.id.label("district_heating_id"),
                egon_district_heating_areas.area_id,
                egon_district_heating_areas.geom_polygon,
                egon_district_heating_areas.scenario,
            ).filter(egon_district_heating_areas.scenario == scenario)

            srid = db.get_srid_of_db_table(
                session, egon_district_heating_areas.geom_polygon
            )

            gdf_district_heating_areas = gpd.read_postgis(
                query_district_heating_areas.statement,
                engine,
                geom_col="geom_polygon",
                index_col=None,
                crs=f"EPSG:{srid}",
            )

            # check if there is a CHP plant where heat pump / resistive heater can
            # be located
            srid_dh_supply = db.get_srid_of_db_table(
                session, egon_district_heating.geometry
            )
            query = session.query(
                egon_district_heating.district_heating_id,
                egon_district_heating.geometry.label("geom"),
            ).filter(
                egon_district_heating.carrier == "CHP",
                egon_district_heating.scenario == scenario,
            )
            df_geom_chp = gpd.read_postgis(
                query.statement,
                engine,
                index_col=None,
                crs=f"EPSG:{srid_dh_supply}",
            )
            return {
                "plants": df,
                "heat_buses": gdf_etrago_bus,
                "weather_cells": gdf_weather_cells,
                "district_heating_areas": gdf_district_heating_areas,
                "chp_plants": df_geom_chp,
            }
        return {"plants": df}


def _prepare_central_heat_sources(edisgo_object, data):
    """Assign weather/district IDs and locations without database access.

    Each heat bus gets the weather cell it lies in and the nearest district heating
    area. Prefer a CHP plant in the same heating district and within the grid, then
    the heat-bus location if it is within the grid, and finally the HV/MV station.
    All results are in the grid CRS.
    """
    df = data["plants"]
    if df.empty:
        return df.copy()

    mv_grid_geom_srid = edisgo_object.topology.grid_district["srid"]
    gdf_etrago_bus = data["heat_buses"].to_crs(mv_grid_geom_srid)
    gdf_weather_cells = data["weather_cells"].to_crs(mv_grid_geom_srid)
    gdf_district_heating_areas = data["district_heating_areas"].to_crs(
        mv_grid_geom_srid
    )
    df_geom_chp = data["chp_plants"].to_crs(mv_grid_geom_srid)

    # Perform spatial join with weather cells
    gdf_combined = gpd.sjoin(
        gdf_etrago_bus,
        gdf_weather_cells,
        how="left",
        predicate="intersects",
        rsuffix="weather",
    )

    # Assign each heat bus to the nearest district heating area. sjoin_nearest
    # measures distances in units of the CRS, so the join is done in a projected
    # CRS in meters (EPSG:3035, ETRS89-LAEA for Europe), also if the grid is in a
    # geographic CRS. The heat bus geometries are carried along in the grid CRS and
    # set as geometry again afterwards, so that they are returned unchanged.
    metric_crs = "EPSG:3035"
    geom_col = gdf_combined.geometry.name
    df_geom = gpd.sjoin_nearest(
        gdf_combined.assign(_geom_grid_crs=gdf_combined.geometry).to_crs(metric_crs),
        gdf_district_heating_areas.to_crs(metric_crs),
        how="left",
        rsuffix="district_heating",
    )
    df_geom = (
        df_geom.drop(columns=geom_col)
        .set_geometry("_geom_grid_crs")
        .rename_geometry(geom_col)
    )

    df_merge = pd.merge(df_geom, df, how="right", right_on="bus1", left_on="bus1")

    df_geom_chp = df_geom_chp.loc[
        df_geom_chp.district_heating_id.isin(df_geom.district_heating_id), :
    ]

    # set geolocation of central heat pump / resistive heater
    for idx in df_merge.index:
        geom = None
        # if there is a CHP plant and it lies within the grid district, use
        # its geolocation
        df_chp = df_geom_chp[
            df_geom_chp.district_heating_id == df_merge.at[idx, "district_heating_id"]
        ]
        if not df_chp.empty:
            for idx_chp in df_chp.index:
                if edisgo_object.topology.grid_district["geom"].contains(
                    df_chp.at[idx_chp, "geom"]
                ):
                    geom = df_chp.at[idx_chp, "geom"]
                    break
        # if the heat bus lies within the grid district, use its geolocation
        if geom is None and edisgo_object.topology.grid_district["geom"].contains(
            df_merge.at[idx, "geom"]
        ):
            geom = df_merge.at[idx, "geom"]
        # if geom is still None, use geolocation of HV/MV station
        if geom is None:
            hvmv_station = edisgo_object.topology.mv_grid.station
            geom = Point(hvmv_station.x.iloc[0], hvmv_station.y.iloc[0])
        df_merge.at[idx, "geom"] = geom
    return df_merge.loc[
        :,
        ["p_set", "weather_cell_id", "district_heating_id", "geom", "area_id"],
    ]


def _query_central_heat_pumps_oedb(edisgo_object, scenario, engine, query_limit=None):
    """Retrieve central heat-pump records and raw spatial support data."""
    return _query_central_heat_sources_oedb(
        edisgo_object, scenario, engine, "central_heat_pump", query_limit=query_limit
    )


def _query_central_resistive_heaters_oedb(
    edisgo_object, scenario, engine, query_limit=None
):
    """Retrieve central resistive-heater records and raw spatial support data."""
    return _query_central_heat_sources_oedb(
        edisgo_object,
        scenario,
        engine,
        "central_resistive_heater",
        query_limit=query_limit,
    )


def _query_individual_heat_pump_capacity_oedb(edisgo_object, scenario, engine):
    """
    Get total capacity of heat pumps for individual heating from oedb for sanity
    checking.

    """
    config = Config()
    (egon_individual_heating,) = config.import_tables_from_oep(
        engine, ["egon_individual_heating"], "supply"
    )
    with db.session_scope_egon_data(engine) as session:
        query = session.query(
            egon_individual_heating.capacity,
        ).filter(
            egon_individual_heating.scenario == scenario,
            egon_individual_heating.carrier == "heat_pump",
            egon_individual_heating.mv_grid_id == edisgo_object.topology.id,
        )
        cap = query.all()
        if len(cap) == 0:
            return 0.0
        else:
            return np.sum(cap)


def oedb(edisgo_object, scenario, engine, import_types=None):
    """
    Get heat pumps for a specified OEP scenario and integrate them into the grid.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve heat-pump data. Possible options are
        "eGon2035" and "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    import_types : list(str) or None
        Technologies to import. Possible options are "individual_heat_pumps",
        "central_heat_pumps" and "central_resistive_heaters". If None, all are
        imported.

    Returns
    -------
    list(str)
        Names of the integrated heat pumps in
        :attr:`~.network.topology.Topology.loads_df`.
    """

    if import_types is None:
        import_types = [
            "individual_heat_pumps",
            "central_heat_pumps",
            "central_resistive_heaters",
        ]

    if "individual_heat_pumps" in import_types:
        hp_individual = _query_individual_heat_pumps_oedb(
            edisgo_object=edisgo_object, scenario=scenario, engine=engine
        )
        hp_individual_cap = _query_individual_heat_pump_capacity_oedb(
            edisgo_object=edisgo_object, scenario=scenario, engine=engine
        )
        if not np.isclose(hp_individual_cap, hp_individual.p_set.sum(), atol=1e-3):
            logger.warning(
                f"Capacity of individual heat pumps ({hp_individual.p_set.sum()} MW) "
                f"differs from expected capacity ({hp_individual_cap} MW)."
            )
    else:
        hp_individual = pd.DataFrame(columns=["p_set"])

    if "central_heat_pumps" in import_types:
        central_data = _query_central_heat_pumps_oedb(
            edisgo_object=edisgo_object, scenario=scenario, engine=engine
        )
        hp_central = _prepare_central_heat_sources(edisgo_object, central_data)
    else:
        hp_central = pd.DataFrame(columns=["p_set"])

    if "central_resistive_heaters" in import_types:
        resistive_data = _query_central_resistive_heaters_oedb(
            edisgo_object=edisgo_object, scenario=scenario, engine=engine
        )
        resistive_heaters_central = _prepare_central_heat_sources(
            edisgo_object, resistive_data
        )
    else:
        resistive_heaters_central = pd.DataFrame(columns=["p_set"])

    # integrate into grid
    return _grid_integration(
        edisgo_object=edisgo_object,
        hp_individual=hp_individual.sort_values(by="p_set"),
        hp_central=hp_central.sort_values(by="p_set"),
        resistive_heaters_central=resistive_heaters_central.sort_values(by="p_set"),
    )


def _grid_integration(
    edisgo_object,
    hp_individual,
    hp_central,
    resistive_heaters_central,
):
    """
    Integrates heat pumps for individual and district heating into the grid.

    See :attr:`~.edisgo.EDisGo.import_heat_pumps` for more information on grid
    integration.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    hp_individual : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all heat pumps for individual heating per building.
        Columns are:

            * p_set : float
                Nominal electric power of heat pump in MW.
            * building_id : int
                Building ID of the building the heat pump is in.
            * weather_cell_id : int
                Weather cell the heat pump is in used to obtain the COP time series.

    hp_central : :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
        Geodataframe containing all heat pumps in district heating networks.
        Columns are:

            * p_set : float
                Nominal electric power of heat pump in MW.
            * district_heating_id : int
                ID of the district heating network the heat pump is in, used to obtain
                other heat supply technologies from supply.egon_district_heating.
            * area_id : int
                ID of the district heating network the heat pump is in, used to obtain
                heat demand time series from demand.egon_timeseries_district_heating.
            * weather_cell_id : int
                Weather cell the heat pump is in used to obtain the COP time series.
            * geom : :shapely:`Shapely Point object<points>`
                Geolocation of the heat pump in the same coordinate reference system as
                the MV grid district geometry.

    resistive_heaters_central : :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
        Geodataframe containing all resistive heaters in district heating networks.
        Columns are the same as for `hp_central`.

    Returns
    --------
    list(str)
        List with names (as in index of :attr:`~.network.topology.Topology.loads_df`)
        of integrated heat pumps.

    """
    # integrate individual heat pumps
    if not hp_individual.empty:
        # join busses corresponding to building ID
        loads_df = edisgo_object.topology.loads_df
        building_id_busses = (
            loads_df[loads_df.type == "conventional_load"]
            .drop_duplicates(subset=["building_id"])
            .set_index("building_id")
            .loc[:, ["bus"]]
        )
        hp_individual = hp_individual.join(
            building_id_busses, how="left", on="building_id"
        )

        # add further information needed in loads_df
        hp_individual["sector"] = "individual_heating"
        hp_individual["type"] = "heat_pump"
        # add heat pump name as index
        hp_individual["index"] = hp_individual.apply(
            lambda _: f"HP_{_.building_id}", axis=1
        )
        hp_individual.set_index("index", drop=True, inplace=True)

        # check for duplicated load names and choose random name for duplicates
        tmp = hp_individual.index.append(edisgo_object.topology.loads_df.index)
        duplicated_indices = tmp[tmp.duplicated()]
        for duplicate in duplicated_indices:
            # find unique name
            random.seed(a=duplicate)
            new_name = duplicate
            while new_name in tmp:
                new_name = f"{duplicate}_{random.randint(10**1, 10**2)}"
            # change name in hp_individual
            hp_individual.rename(index={duplicate: new_name}, inplace=True)

        # filter heat pumps that are too large to be integrated into LV level
        hp_individual_large = hp_individual[
            hp_individual.p_set
            > edisgo_object.config["grid_connection"]["upper_limit_voltage_level_7"]
        ]
        hp_individual_small = hp_individual[
            hp_individual.p_set
            <= edisgo_object.config["grid_connection"]["upper_limit_voltage_level_7"]
        ]

        # integrate small individual heat pumps at buildings
        edisgo_object.topology.loads_df = pd.concat(
            [edisgo_object.topology.loads_df, hp_individual_small]
        )

        # accumulate integrated heat pump names in a plain list (built into an Index
        # once at the end) to avoid O(n^2) per-iteration Index.append below
        integrated_hps_list: list = list(hp_individual_small.index)

        # integrate large individual heat pumps - if building is already connected to
        # higher voltage level it can be integrated at same bus, otherwise it is
        # integrated based on geolocation
        integrated_hps_own_grid_conn_list: list = []
        # collect same-bus rows and concat them into loads_df once after the loop to
        # avoid O(n^2) per-iteration pd.concat (rows are independent additions, so
        # batching keeps the same resulting frame and row order)
        same_bus_rows: list = []
        for hp in hp_individual_large.index:
            # check if building is already connected to a voltage level equal to or
            # higher than the voltage level the heat pump should be connected to
            bus_building = hp_individual_large.at[hp, "bus"]
            voltage_level_building = determine_bus_voltage_level(
                edisgo_object, bus_building
            )
            voltage_level_hp = determine_grid_integration_voltage_level(
                edisgo_object, hp_individual_large.at[hp, "p_set"]
            )

            if voltage_level_hp >= voltage_level_building:
                # integrate at same bus as building
                same_bus_rows.append(hp_individual_large.loc[[hp], :])
                integrated_hps_list.append(hp)
            else:
                # integrate based on geolocation
                hp_name = edisgo_object.integrate_component_based_on_geolocation(
                    comp_type="heat_pump",
                    voltage_level=voltage_level_hp,
                    geolocation=(
                        edisgo_object.topology.buses_df.at[bus_building, "x"],
                        edisgo_object.topology.buses_df.at[bus_building, "y"],
                    ),
                    add_ts=False,
                    p_set=hp_individual_large.at[hp, "p_set"],
                    weather_cell_id=hp_individual_large.at[hp, "weather_cell_id"],
                    sector="individual_heating",
                    building_id=hp_individual_large.at[hp, "building_id"],
                )
                integrated_hps_list.append(hp_name)
                integrated_hps_own_grid_conn_list.append(hp)

        # concat all collected same-bus rows in one operation (O(n) instead of O(n^2))
        if same_bus_rows:
            edisgo_object.topology.loads_df = pd.concat(
                [edisgo_object.topology.loads_df, *same_bus_rows]
            )
        integrated_hps = pd.Index(integrated_hps_list)
        integrated_hps_own_grid_conn = pd.Index(integrated_hps_own_grid_conn_list)
        # logging messages
        logger.debug(
            f"{sum(hp_individual.p_set):.2f} MW of heat pumps for individual heating "
            f"integrated."
        )
        if len(integrated_hps_own_grid_conn) > 0:
            logger.debug(
                f"Of this, "
                f"{sum(hp_individual.loc[integrated_hps_own_grid_conn, 'p_set']):.2f} "
                f"MW have separate grid connection point."
            )
    else:
        integrated_hps = pd.Index([])

    if not hp_central.empty:
        # integrate central heat pumps
        for hp in hp_central.index:
            # determine voltage level, considering resistive heaters
            p_set = hp_central.at[hp, "p_set"]
            if not resistive_heaters_central.empty:
                rh = resistive_heaters_central[
                    resistive_heaters_central.district_heating_id
                    == hp_central.at[hp, "district_heating_id"]
                ]
                p_set += rh.p_set.sum()
            voltage_level = determine_grid_integration_voltage_level(
                edisgo_object, p_set
            )
            # check if there is a resistive heater as well
            hp_name = edisgo_object.integrate_component_based_on_geolocation(
                comp_type="heat_pump",
                geolocation=hp_central.at[hp, "geom"],
                voltage_level=voltage_level,
                add_ts=False,
                p_set=hp_central.at[hp, "p_set"],
                weather_cell_id=hp_central.at[hp, "weather_cell_id"],
                sector="district_heating",
                district_heating_id=hp_central.at[hp, "district_heating_id"],
                area_id=hp_central.at[hp, "area_id"],
            )
            integrated_hps = integrated_hps.append(pd.Index([hp_name]))

        logger.debug(
            f"{sum(hp_central.p_set):.2f} MW of heat pumps for district heating "
            f"integrated."
        )

    if not resistive_heaters_central.empty:
        # integrate central resistive heaters
        for rh in resistive_heaters_central.index:
            integrated = False
            # check if there already is a component in the same district heating network
            # integrated into the grid and if so, use the same bus
            if "district_heating_id" in edisgo_object.topology.loads_df.columns:
                tmp = edisgo_object.topology.loads_df[
                    edisgo_object.topology.loads_df.district_heating_id
                    == resistive_heaters_central.at[rh, "district_heating_id"]
                ]
                if not tmp.empty:
                    hp_name = edisgo_object.add_component(
                        comp_type="load",
                        type="heat_pump",
                        sector="district_heating_resistive_heater",
                        bus=tmp.bus.iloc[0],
                        p_set=resistive_heaters_central.at[rh, "p_set"],
                        weather_cell_id=resistive_heaters_central.at[
                            rh, "weather_cell_id"
                        ],
                        district_heating_id=resistive_heaters_central.at[
                            rh, "district_heating_id"
                        ],
                        area_id=resistive_heaters_central.at[rh, "area_id"],
                    )
                    integrated = True
            if integrated is False:
                hp_name = edisgo_object.integrate_component_based_on_geolocation(
                    comp_type="heat_pump",
                    geolocation=resistive_heaters_central.at[rh, "geom"],
                    add_ts=False,
                    p_set=resistive_heaters_central.at[rh, "p_set"],
                    weather_cell_id=resistive_heaters_central.at[rh, "weather_cell_id"],
                    sector="district_heating_resistive_heater",
                    district_heating_id=resistive_heaters_central.at[
                        rh, "district_heating_id"
                    ],
                    area_id=resistive_heaters_central.at[rh, "area_id"],
                )
            integrated_hps = integrated_hps.append(pd.Index([hp_name]))

        logger.debug(
            f"{sum(resistive_heaters_central.p_set):.2f} MW of resistive heaters for "
            f"district heating integrated."
        )

    return integrated_hps


def efficiency_resistive_heaters_oedb(scenario, engine):
    """
    Get efficiency of resistive heaters from the
    `OpenEnergy DataBase <https://openenergyplatform.org/database/>`_.

    Parameters
    ----------
    scenario : str
        Scenario for which to retrieve efficiency data. Possible options
        are "eGon2035" and "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    -------
    dict
        Dictionary with efficiency of resistive heaters in district and individual
        heating. Keys of the dictionary are
        "central_resistive_heater" giving the efficiency of resistive heaters in
        district heating and "rural_resistive_heater" giving the efficiency of
        resistive heaters in individual heating systems. Values are of type float and
        given in p.u.

    """
    heat_parameters = _query_resistive_heater_efficiency_oedb(
        scenario=scenario,
        engine=engine,
    )

    return heat_parameters["efficiency"]


def _query_resistive_heater_efficiency_oedb(scenario, engine):
    """Query heat parameters containing resistive-heater efficiencies."""

    config = Config()
    (egon_scenario_parameters,) = config.import_tables_from_oep(
        engine, ["egon_scenario_parameters"], "scenario"
    )

    with db.session_scope_egon_data(engine) as session:
        query = session.query(
            egon_scenario_parameters.heat_parameters,
        ).filter(egon_scenario_parameters.name == scenario)

        return query.first()[0]
