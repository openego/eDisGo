import logging
import os
import random

import numpy as np
import pandas as pd
import saio

from sqlalchemy import func

from edisgo.io import db
from edisgo.tools.tools import (
    determine_bus_voltage_level,
    determine_grid_integration_voltage_level,
)

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

    from shapely.geometry import Point

logger = logging.getLogger(__name__)


def oedb(edisgo_object, scenario, engine, import_types=None):
    """
    Gets heat pumps for specified scenario from oedb and integrates them into the grid.

    See :attr:`~.edisgo.EDisGo.import_heat_pumps` for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve heat pump data. Possible options
        are "eGon2035" and "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    import_types : list(str) or None
        Specifies which technologies to import. Possible options are
        "individual_heat_pumps", "central_heat_pumps" and "central_resistive_heaters".
        If None, all are imported.

    Returns
    --------
    list(str)
        List with names (as in index of :attr:`~.network.topology.Topology.loads_df`)
        of integrated heat pumps.

    """

    def _get_individual_heat_pumps():
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

        df = pd.read_sql(query.statement, engine, index_col=None)

        # drop duplicated building IDs that exist because
        # egon_map_zensus_mvgd_buildings can contain several entries per building,
        # e.g. for CTS and residential
        return df.drop_duplicates(subset=["building_id"])

    def _get_central_heat_pump_or_resistive_heaters(carrier):
        """
        Get heat pumps or resistive heaters in district heating from oedb.

        Weather cell ID and geolocation is as well added in this function.
        Concerning the geolocation - the electricity bus central heat pumps and
        resistive heaters are attached to is not determined based on the geolocation
        (which is in egon_data always set to the centroid of the district heating area
        they are in) but based on the majority of heat demand in an MV grid area.
        Further, large heat pumps and resistive heaters are split into several smaller
        ones in egon_data. The geolocation is however not adapted in egon_data. Due to
        this, it is often the case, that the central heat pumps and resistive heaters
        lie outside the MV grid district area. Therefore, the geolocation is adapted in
        this function. It is first checked, if there is a CHP plant in the same district
        heating area. If this is the case and the CHP plant lies within the MV grid
        district, then the geolocation of the is set to the same geolocation as the
        CHP plant, as it is assumed, that this would be a suitable place for a heat pump
        and resistive heaters as well. If there is no
        CHP plant, then it is checked if the centroid of the district heating area lies
        within the MV grid. If this is the case, then this is used. If neither of these
        options can be used, then the geolocation of the HV/MV station is used, as there
        is no better indicator where the heat pump or resistive heater would be placed.

        Parameters
        ----------
        carrier : str
            Specifies whether to obtain central heat pumps or resistive heaters.
            Possible options are "central_heat_pump" and "central_resistive_heater".

        Returns
        -------
        :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
            Geodataframe containing information on all central heat pumps or
            resistive heaters in the grid per district heating area.
            For more information see parameter `hp_central` or
            `resistive_heater_central` in
            :func:`~.io.heat_pump_import._grid_integration`.

        """
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
        df = pd.read_sql(query.statement, engine, index_col=None)
        if not df.empty:
            # get geom of heat bus, weather_cell_id, district_heating_id and area_id
            srid_etrago_bus = db.get_srid_of_db_table(session, egon_etrago_bus.geom)
            query = (
                session.query(
                    egon_etrago_bus.bus_id.label("bus1"),
                    egon_etrago_bus.geom,
                    egon_era5_weather_cells.w_id.label("weather_cell_id"),
                    egon_district_heating_areas.id.label("district_heating_id"),
                    egon_district_heating_areas.area_id,
                )
                .filter(
                    egon_etrago_bus.scn_name == scenario,
                    egon_district_heating_areas.scenario == scenario,
                    egon_etrago_bus.bus_id.in_(df.bus1),
                )
                .outerjoin(  # join to obtain weather cell ID
                    egon_era5_weather_cells,
                    db.sql_within(
                        egon_etrago_bus.geom,
                        egon_era5_weather_cells.geom,
                        mv_grid_geom_srid,
                    ),
                )
                .outerjoin(  # join to obtain district heating ID
                    egon_district_heating_areas,
                    func.ST_Transform(
                        func.ST_Centroid(egon_district_heating_areas.geom_polygon),
                        srid_etrago_bus,
                    )
                    == egon_etrago_bus.geom,
                )
            )
            df_geom = gpd.read_postgis(
                query.statement,
                engine,
                index_col=None,
                crs=f"EPSG:{srid_etrago_bus}",
            ).to_crs(mv_grid_geom_srid)
            # merge dataframes
            df_merge = pd.merge(
                df_geom, df, how="right", right_on="bus1", left_on="bus1"
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
                egon_district_heating.district_heating_id.in_(
                    df_geom.district_heating_id
                ),
            )
            df_geom_chp = gpd.read_postgis(
                query.statement,
                engine,
                index_col=None,
                crs=f"EPSG:{srid_dh_supply}",
            ).to_crs(mv_grid_geom_srid)

            # set geolocation of central heat pump / resistive heater
            for idx in df_merge.index:
                geom = None
                # if there is a CHP plant and it lies within the grid district, use
                # its geolocation
                df_chp = df_geom_chp[
                    df_geom_chp.district_heating_id
                    == df_merge.at[idx, "district_heating_id"]
                ]
                if not df_chp.empty:
                    for idx_chp in df_chp.index:
                        if edisgo_object.topology.grid_district["geom"].contains(
                            df_chp.at[idx_chp, "geom"]
                        ):
                            geom = df_chp.at[idx_chp, "geom"]
                            break
                # if the heat bus lies within the grid district, use its geolocation
                if geom is None and edisgo_object.topology.grid_district[
                    "geom"
                ].contains(df_merge.at[idx, "geom"]):
                    geom = df_merge.at[idx, "geom"]
                # if geom is still None, use geolocation of HV/MV station
                if geom is None:
                    hvmv_station = edisgo_object.topology.mv_grid.station
                    geom = Point(hvmv_station.x[0], hvmv_station.y[0])
                df_merge.at[idx, "geom"] = geom
            return df_merge.loc[
                :,
                ["p_set", "weather_cell_id", "district_heating_id", "geom", "area_id"],
            ]
        else:
            return df

    def _get_individual_heat_pump_capacity():
        """
        Get total capacity of heat pumps for individual heating from oedb for sanity
        checking.

        """
        query = session.query(egon_individual_heating.capacity,).filter(
            egon_individual_heating.scenario == scenario,
            egon_individual_heating.carrier == "heat_pump",
            egon_individual_heating.mv_grid_id == edisgo_object.topology.id,
        )
        cap = query.all()
        if len(cap) == 0:
            return 0.0
        else:
            return np.sum(cap)

    saio.register_schema("demand", engine)
    from saio.demand import egon_district_heating_areas, egon_hp_capacity_buildings

    saio.register_schema("supply", engine)
    from saio.supply import (
        egon_district_heating,
        egon_era5_weather_cells,
        egon_individual_heating,
    )

    saio.register_schema("boundaries", engine)
    from saio.boundaries import (
        egon_map_zensus_mvgd_buildings,
        egon_map_zensus_weather_cell,
    )

    saio.register_schema("grid", engine)
    from saio.grid import egon_etrago_bus, egon_etrago_link

    building_ids = edisgo_object.topology.loads_df.building_id.unique()
    mv_grid_geom_srid = edisgo_object.topology.grid_district["srid"]

    if import_types is None:
        import_types = [
            "individual_heat_pumps",
            "central_heat_pumps",
            "central_resistive_heaters",
        ]

    # get individual and district heating heat pumps, as well as resistive heaters
    # in district heating
    with db.session_scope_egon_data(engine) as session:
        if "individual_heat_pumps" in import_types:
            hp_individual = _get_individual_heat_pumps()
        else:
            hp_individual = pd.DataFrame(columns=["p_set"])

        if "central_heat_pumps" in import_types:
            hp_central = _get_central_heat_pump_or_resistive_heaters(
                "central_heat_pump"
            )
        else:
            hp_central = pd.DataFrame(columns=["p_set"])

        if "central_resistive_heaters" in import_types:
            resistive_heaters_central = _get_central_heat_pump_or_resistive_heaters(
                "central_resistive_heater"
            )
        else:
            resistive_heaters_central = pd.DataFrame(columns=["p_set"])

    # sanity check
    with db.session_scope_egon_data(engine) as session:
        hp_individual_cap = _get_individual_heat_pump_capacity()
    if not np.isclose(hp_individual_cap, hp_individual.p_set.sum(), atol=1e-3):
        logger.warning(
            f"Capacity of individual heat pumps ({hp_individual.p_set.sum()} MW) "
            f"differs from expected capacity ({hp_individual_cap} MW)."
        )

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
                new_name = f"{duplicate}_{random.randint(10 ** 1, 10 ** 2)}"
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

        integrated_hps = hp_individual_small.index

        # integrate large individual heat pumps - if building is already connected to
        # higher voltage level it can be integrated at same bus, otherwise it is
        # integrated based on geolocation
        integrated_hps_own_grid_conn = pd.Index([])
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
                edisgo_object.topology.loads_df = pd.concat(
                    [edisgo_object.topology.loads_df, hp_individual_large.loc[[hp], :]]
                )
                integrated_hps = integrated_hps.append(pd.Index([hp]))
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
                integrated_hps = integrated_hps.append(pd.Index([hp_name]))
                integrated_hps_own_grid_conn = integrated_hps_own_grid_conn.append(
                    pd.Index([hp])
                )
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
                        bus=tmp.bus[0],
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
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

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
    saio.register_schema("scenario", engine)
    from saio.scenario import egon_scenario_parameters

    # get cop from database
    with db.session_scope_egon_data(engine) as session:
        query = session.query(
            egon_scenario_parameters.heat_parameters,
        ).filter(egon_scenario_parameters.name == scenario)
        eta_dict = query.first()[0]["efficiency"]

    return eta_dict
