import logging
import os
import random

import numpy as np
import pandas as pd
import saio

from edisgo.io import db
from edisgo.tools.tools import (
    determine_bus_voltage_level,
    determine_grid_integration_voltage_level,
)

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

logger = logging.getLogger(__name__)


def oedb(edisgo_object, scenario, engine):
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

    def _get_central_heat_pumps():
        """
        Get heat pumps in district heating from oedb.

        Weather cell ID is as well added in this function.

        Returns
        -------
        :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
            Geodataframe containing installed heat pump capacity for all central heat
            pumps in the grid per district heating area.
            For more information see parameter `hp_central` in
            :func:`~.io.heat_pump_import._grid_integration`.

        """
        query = (
            session.query(
                egon_district_heating.district_heating_id,
                egon_district_heating.capacity.label("p_set"),
                egon_district_heating.geometry.label("geom"),
                egon_era5_weather_cells.w_id.label("weather_cell_id"),
            )
            .filter(
                egon_district_heating.scenario == scenario,
                egon_district_heating.carrier == "heat_pump",
                # filter heat pumps inside MV grid district geometry
                db.sql_within(
                    egon_district_heating.geometry,
                    db.sql_grid_geom(edisgo_object),
                    mv_grid_geom_srid,
                ),
            )
            .outerjoin(  # join to obtain weather cell ID
                egon_era5_weather_cells,
                db.sql_within(
                    egon_district_heating.geometry,
                    egon_era5_weather_cells.geom,
                    mv_grid_geom_srid,
                ),
            )
        )

        df = gpd.read_postgis(query.statement, engine, index_col=None)

        # transform to same SRID as MV grid district geometry
        return df.to_crs(mv_grid_geom_srid)

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
    from saio.demand import egon_hp_capacity_buildings

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

    building_ids = edisgo_object.topology.loads_df.building_id.unique()
    mv_grid_geom_srid = edisgo_object.topology.grid_district["srid"]

    # get individual and district heating heat pumps
    with db.session_scope_egon_data(engine) as session:
        hp_individual = _get_individual_heat_pumps()
        hp_central = _get_central_heat_pumps()

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
    )


def _grid_integration(
    edisgo_object,
    hp_individual,
    hp_central,
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
        Geodataframe containing all heat pumps in district heating network.
        Columns are:

            * p_set : float
                Nominal electric power of heat pump in MW.
            * district_heating_id : int
                ID of the district heating network the heat pump is in.
            * weather_cell_id : int
                Weather cell the heat pump is in used to obtain the COP time series.
            * geom : :shapely:`Shapely Point object<points>`
                Geolocation of the heat pump in the same coordinate reference system as
                the MV grid district geometry.

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
            hp_name = edisgo_object.integrate_component_based_on_geolocation(
                comp_type="heat_pump",
                geolocation=hp_central.at[hp, "geom"],
                add_ts=False,
                p_set=hp_central.at[hp, "p_set"],
                weather_cell_id=hp_central.at[hp, "weather_cell_id"],
                sector="district_heating",
                district_heating_id=hp_central.at[hp, "district_heating_id"],
            )
            integrated_hps = integrated_hps.append(pd.Index([hp_name]))

        logger.debug(
            f"{sum(hp_central.p_set):.2f} MW of heat pumps for district heating "
            f"integrated."
        )
    return integrated_hps
