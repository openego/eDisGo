import logging

# import pandas as pd

# from edisgo.tools import session_scope

logger = logging.getLogger(__name__)


def oedb(edisgo_object, scenario, **kwargs):
    """
    Gets generator park for specified scenario from oedb and integrates them
    into the grid.

    The importer uses SQLAlchemy ORM objects.
    These are defined in
    `ego.io <https://github.com/openego/ego.io/tree/dev/egoio/db_tables/>`_.

    For further information see also :attr:`~.EDisGo.import_generators`.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    generator_scenario : str
        Scenario for which to retrieve generator data. Possible options
        are 'nep2035' and 'ego100'.

    Other Parameters
    ----------------
    remove_decommissioned : bool
        If True, removes generators from network that are not included in
        the imported dataset (=decommissioned). Default: True.
    update_existing : bool
        If True, updates capacity of already existing generators to
        capacity specified in the imported dataset. Default: True.
    p_target : dict or None
        Per default, no target capacity is specified and generators are
        expanded as specified in the respective scenario. However, you may
        want to use one of the scenarios but have slightly more or less
        generation capacity than given in the respective scenario. In that case
        you can specify the desired target capacity per technology type using
        this input parameter. The target capacity dictionary must have
        technology types (e.g. 'wind' or 'solar') as keys and corresponding
        target capacities in MW as values.
        If a target capacity is given that is smaller than the total capacity
        of all generators of that type in the future scenario, only some of
        the generators in the future scenario generator park are installed,
        until the target capacity is reached.
        If the given target capacity is greater than that of all generators
        of that type in the future scenario, then each generator capacity is
        scaled up to reach the target capacity. Be careful to not have much
        greater target capacities as this will lead to unplausible generation
        capacities being connected to the different voltage levels.
        Also be aware that only technologies specified in the dictionary are
        expanded. Other technologies are kept the same.
        Default: None.
    allowed_number_of_comp_per_lv_bus : int
        Specifies, how many generators are at most allowed to be placed at
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

    def _get_large_heat_pumps(session):
        """
        Get heat pumps in district heating from oedb.

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
    #     hp_large = _get_large_heat_pumps(session)
    #
    # return _grid_integration(
    #     edisgo_object=edisgo_object,
    #     hp_individual=hp_individual,
    #     hp_large=hp_large,
    # )


def _grid_integration(
    edisgo_object,
    hp_individual,
    hp_large,
):
    """
    Integrates the heat pumps into the grid.

    Grid connection points of heat pumps for individual heating are determined based
    on the corresponding building ID.

    Grid connection points of heat pumps for district
    heating are determined based on their geolocation and finding the nearest grid
    connection point. See :attr:`~.network.topology.Topology.connect_to_mv` and
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

    hp_large : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all heat pumps in district heating network.
        # ToDo add information on dataframe columns and index
        Index of the dataframe are the generator IDs.
        Columns are:

            * p_set : float
                Nominal capacity in MW.
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
    # edisgo_object.topology.loads_df = pd.concat([loads_df, hp_df])
    # logger.debug(
    #     f"{sum(hp_df.p_set)} MW of heat pumps for individual heating integrated."
    # )
    #
    # # integrate large heat pumps
    # for hp in hp_large:
    #     hp_name = edisgo_object.integrate_component_based_on_geolocation(
    #         comp_type="heat_pump",
    #         geolocation=hp_large.at[hp, "geom"],
    #         add_ts=False,
    #         p_set=hp_large.at[hp, "p_set"],
    #         weather_cell_id=hp_large.at[hp, "weather_cell_id"],
    #         kwargs={"sector": "district_heating"},
    #     )
    #     integrated_heat_pumps = integrated_heat_pumps.append(hp_name)
    #
    # logger.debug(
    #     f"{sum(hp_large.p_set)} MW of heat pumps for district heating " f"integrated."
    # )
    # return integrated_heat_pumps
