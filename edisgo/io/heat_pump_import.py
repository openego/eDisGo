import logging

# import pandas as pd

# from edisgo.tools import session_scope

logger = logging.getLogger(__name__)


def oedb(edisgo_object, scenario, **kwargs):
    """
    Gets heat pumps for specified scenario from oedb and integrates them into the grid.

    See :attr:`~.edisgo.EDisGo.import_heat_pumps` for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve heat pump data. Possible options
        are 'eGon2035' and 'eGon100RE'.

    Other Parameters
    -----------------
    allowed_number_of_comp_per_lv_bus : int
        Specifies, how many heat pumps are at most allowed to be placed at
        the same LV bus. Default: 2.

    Returns
    --------
    list(str)
        List with names (as in index of :attr:`~.network.topology.Topology.loads_df`)
        of integrated heat pumps.

    """

    def _get_individual_heat_pumps(session):
        """
        Get heat pumps for individual heating from oedb.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe containing data on all heat pumps for individual heating.
            # ToDo add information on dataframe columns and index

        """
        raise NotImplementedError
        # # build query
        #
        # return pd.read_sql(
        #     query.statement, session.bind, index_col="id"
        # )

    def _get_central_heat_pumps(session):
        """
        Get heat pumps in district heating from oedb.

        # ToDo The building IDs of all buildings served by the district heating
        # network should also be retrieved

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe containing data on all heat pumps for district heating.
            # ToDo add information on dataframe columns and index

        """
        raise NotImplementedError
        # # build query
        #
        # return pd.read_sql(
        #     query.statement, session.bind, index_col="id"
        # )

    raise NotImplementedError

    # srid = edisgo_object.topology.grid_district["srid"]
    #
    # # get individual and district heating heat pumps
    # with session_scope() as session:
    #     hp_individual = _get_individual_heat_pumps(session)
    #     hp_central = _get_central_heat_pumps(session)
    #
    # # integrate into grid
    # return _grid_integration(
    #     edisgo_object=edisgo_object,
    #     hp_individual=hp_individual,
    #     hp_large=hp_central,
    # )


def _grid_integration(
    edisgo_object,
    hp_individual,
    hp_central,
):
    """
    Integrates the heat pumps into the grid.

    Grid connection points of heat pumps for individual heating are determined based
    on the corresponding building ID.

    Grid connection points of heat pumps for district
    heating are determined based on their geolocation and installed capacity. See
    :attr:`~.network.topology.Topology.connect_to_mv` and
    :attr:`~.network.topology.Topology.connect_to_lv` for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    hp_individual : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all heat pumps for individual heating.
        # ToDo add information on dataframe columns and index
        Columns are:

            * p_set : float
                Nominal capacity in MW.
            * building_id : int
                Building ID of the building the heat pump is in.
            * weather_cell_id : int
                Weather cell the heat pump is in.

    hp_central : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all heat pumps in district heating network.
        # ToDo add information on dataframe columns and index
        Index of the dataframe are the generator IDs.
        Columns are:

            * p_set : float
                Nominal capacity in MW.
            * building_id : list(int)
                List of building IDs of the buildings in the district heating network
                the heat pump is in.
            * weather_cell_id : int
                Weather cell the heat pump is in.
            * geom : :shapely:`Shapely Point object<points>`
                Geolocation of generator. For CRS see config_grid.srid.

    Returns
    --------
    list(str)
        List with names (as in index of :attr:`~.network.topology.Topology.loads_df`)
        of integrated heat pumps.

    """

    raise NotImplementedError

    # loads_df = edisgo_object.topology.loads_df
    #
    # # integrate individual heat pumps
    # # get buses corresponding to building IDs
    # integrated_heat_pumps = hp_individual.index
    # buses = loads_df[loads_df.building_id.isin(hp_individual.building_id)]
    # hp_df = pd.DataFrame(
    #     {
    #         "bus": buses,
    #         "p_set": hp_individual.p_set,
    #         "type": "heat_pump",
    #         "weather_cell_id": hp_individual.weather_cell_id,
    #         "sector": "individual_heating",
    #     },
    #     index=integrated_heat_pumps,
    # )
    # # ToDo use add_loads for the following?
    # edisgo_object.topology.loads_df = pd.concat([loads_df, hp_df])
    # # add information on heat pump and corresponding building ID to HeatPump class
    # # ToDo building_ids always as list in DataFrame as is the case for large heat
    # #  pumps?
    # edisgo_object.heat_pump.building_ids_df = pd.concat(
    #     [edisgo_object.heat_pump.building_ids_df, hp_df.building_id])
    # logger.debug(
    #     f"{sum(hp_df.p_set)} MW of heat pumps for individual heating integrated."
    # )
    #
    # # integrate central heat pumps
    # for hp in hp_central:
    #     hp_name = edisgo_object.integrate_component_based_on_geolocation(
    #         comp_type="heat_pump",
    #         geolocation=hp_large.at[hp, "geom"],
    #         add_ts=False,
    #         p_set=hp_large.at[hp, "p_set"],
    #         weather_cell_id=hp_large.at[hp, "weather_cell_id"],
    #         kwargs={"sector": "district_heating"},
    #     )
    #     integrated_heat_pumps = integrated_heat_pumps.append(hp_name)
    #     # add information on heat pump and corresponding building IDs to HeatPump
    #     # class
    #     edisgo_object.heat_pump.building_ids_df = pd.concat(
    #         [edisgo_object.heat_pump.building_ids_df,
    #          pd.DataFrame({"building_ids": hp_large.loc[hp, "building_id"]},
    #                       index=[hp_name])
    #          ]
    #     )
    #
    # logger.debug(
    #     f"{sum(hp_central.p_set)} MW of heat pumps for district heating integrated."
    # )
    # return integrated_heat_pumps
