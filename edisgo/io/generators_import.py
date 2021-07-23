import os
import pandas as pd
from sqlalchemy import func
import random
import logging

from edisgo.network.timeseries import add_generators_timeseries
from edisgo.tools import session_scope
from edisgo.tools.geo import (
    proj2equidistant,
)

logger = logging.getLogger("edisgo")

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


def oedb(edisgo_object, generator_scenario, **kwargs):
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

    """

    def _import_conv_generators(session):
        """
        Import data for conventional generators from oedb.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe containing data on all conventional MV generators.
            You can find a full list of columns in
            :func:`edisgo.io.import_data.update_grids`.

        """
        # build query
        generators_sqla = (
            session.query(
                orm_conv_generators.columns.id,
                orm_conv_generators.columns.id.label("generator_id"),
                orm_conv_generators.columns.subst_id,
                orm_conv_generators.columns.la_id,
                orm_conv_generators.columns.capacity.label("p_nom"),
                orm_conv_generators.columns.voltage_level,
                orm_conv_generators.columns.fuel.label("generator_type"),
                func.ST_AsText(
                    func.ST_Transform(orm_conv_generators.columns.geom, srid)
                ).label("geom"),
            ).filter(
                orm_conv_generators.columns.subst_id
                == edisgo_object.topology.mv_grid.id
            ).filter(
                orm_conv_generators.columns.voltage_level.in_([4, 5])
            ).filter(
                orm_conv_generators_version)
        )

        return pd.read_sql_query(
            generators_sqla.statement, session.bind, index_col="id"
        )

    def _import_res_generators(session):
        """
        Import data for renewable generators from oedb.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`,
         :pandas:`pandas.DataFrame<DataFrame>`)
            Dataframe containing data on all renewable MV and LV generators.
            You can find a full list of columns in
            :func:`edisgo.io.import_data.update_grids`.

        Notes
        -----
        If subtype is not specified it is set to 'unknown'.

        """

        # build basic query
        generators_sqla = (
            session.query(
                orm_re_generators.columns.id,
                orm_re_generators.columns.id.label("generator_id"),
                orm_re_generators.columns.subst_id,
                orm_re_generators.columns.la_id,
                orm_re_generators.columns.mvlv_subst_id,
                orm_re_generators.columns.electrical_capacity.label("p_nom"),
                orm_re_generators.columns.generation_type.label(
                    "generator_type"),
                orm_re_generators.columns.generation_subtype.label(
                    "subtype"),
                orm_re_generators.columns.voltage_level,
                orm_re_generators.columns.w_id.label("weather_cell_id"),
                func.ST_AsText(
                    func.ST_Transform(
                        orm_re_generators.columns.rea_geom_new, srid
                    )
                ).label("geom"),
                func.ST_AsText(
                    func.ST_Transform(orm_re_generators.columns.geom, srid)
                ).label("geom_em"),
            ).filter(
                orm_re_generators.columns.subst_id
                == edisgo_object.topology.mv_grid.id
            ).filter(
                orm_re_generators_version)
        )

        # extend basic query for MV generators and read data from db
        generators_mv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([4, 5])
        )
        gens_mv = pd.read_sql_query(
            generators_mv_sqla.statement,
            session.bind,
            index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        gens_mv.loc[
            gens_mv["subtype"].isnull(), "subtype"
        ] = "unknown"

        # convert capacity from kW to MW
        gens_mv.p_nom = pd.to_numeric(gens_mv.p_nom) / 1e3

        # extend basic query for LV generators and read data from db
        generators_lv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([6, 7])
        )
        gens_lv = pd.read_sql_query(
            generators_lv_sqla.statement,
            session.bind,
            index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        gens_lv.loc[
            gens_lv["subtype"].isnull(), "subtype"
        ] = "unknown"

        # convert capacity from kW to MW
        gens_lv.p_nom = pd.to_numeric(gens_lv.p_nom) / 1e3

        return gens_mv, gens_lv

    def _validate_generation():
        """
        Validate generation capacity in updated grids.

        The validation uses the cumulative capacity of all generators.

        """

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -1

        capacity_imported = (generators_res_mv["p_nom"].sum() +
                             generators_res_lv["p_nom"].sum() +
                             generators_conv_mv['p_nom'].sum()
                             )

        capacity_grid = edisgo_object.topology.generators_df.p_nom.sum()

        logger.debug(
            "Cumulative generator capacity (updated): {} MW".format(
                round(capacity_imported, 1)
            )
        )

        if abs(capacity_imported - capacity_grid) > cap_diff_threshold:
            raise ValueError(
                "Cumulative capacity of imported generators ({} MW) "
                "differs from cumulative capacity of generators "
                "in updated grid ({} MW) by {} MW.".format(
                    round(capacity_imported, 1),
                    round(capacity_grid, 1),
                    round(capacity_imported - capacity_grid, 1),
                )
            )
        else:
            logger.debug(
                "Cumulative capacity of imported generators validated."
            )

    def _validate_sample_geno_location():
        """
        Checks that newly imported generators are located inside grid district.

        The check is performed for two randomly sampled generators.

        """
        if (
                all(generators_res_lv["geom"].notnull())
                and all(generators_res_mv["geom"].notnull())
                and not generators_res_lv["geom"].empty
                and not generators_res_mv["geom"].empty
        ):
            projection = proj2equidistant(srid)
            # get geom of 1 random MV and 1 random LV generator and transform
            sample_mv_geno_geom_shp = transform(
                projection,
                wkt_loads(
                    generators_res_mv["geom"].dropna().sample(n=1).values[0]
                ),
            )
            sample_lv_geno_geom_shp = transform(
                projection,
                wkt_loads(
                    generators_res_lv["geom"].dropna().sample(n=1).values[0]
                ),
            )

            # get geom of MV grid district
            mvgd_geom_shp = transform(
                projection,
                edisgo_object.topology.grid_district["geom"],
            )

            # check if MVGD contains geno
            if not (
                    mvgd_geom_shp.contains(sample_mv_geno_geom_shp)
                    and mvgd_geom_shp.contains(sample_lv_geno_geom_shp)
            ):
                raise ValueError(
                    "At least one imported generator is not located in the MV "
                    "grid area. Check compatibility of grid and generator "
                    "datasets."
                )

    oedb_data_source = edisgo_object.config["data_source"]["oedb_data_source"]
    srid = edisgo_object.topology.grid_district["srid"]

    # load ORM names
    orm_conv_generators_name = (
            edisgo_object.config[oedb_data_source][
                "conv_generators_prefix"]
            + generator_scenario
            + edisgo_object.config[oedb_data_source][
                "conv_generators_suffix"]
    )
    orm_re_generators_name = (
            edisgo_object.config[oedb_data_source]["re_generators_prefix"]
            + generator_scenario
            + edisgo_object.config[oedb_data_source]["re_generators_suffix"]
    )

    if oedb_data_source == "model_draft":

        # import ORMs
        orm_conv_generators = model_draft.__getattribute__(
            orm_conv_generators_name
        )
        orm_re_generators = model_draft.__getattribute__(
            orm_re_generators_name
        )

        # set dummy version condition (select all generators)
        orm_conv_generators_version = 1 == 1
        orm_re_generators_version = 1 == 1

    elif oedb_data_source == "versioned":

        data_version = edisgo_object.config["versioned"]["version"]

        # import ORMs
        orm_conv_generators = supply.__getattribute__(orm_conv_generators_name)
        orm_re_generators = supply.__getattribute__(orm_re_generators_name)

        # set version condition
        orm_conv_generators_version = (
                orm_conv_generators.columns.version == data_version
        )
        orm_re_generators_version = (
                orm_re_generators.columns.version == data_version
        )

    # get conventional and renewable generators
    with session_scope() as session:
        generators_conv_mv = _import_conv_generators(session)
        generators_res_mv, generators_res_lv = _import_res_generators(session)

    generators_mv = generators_conv_mv.append(generators_res_mv)

    # validate that imported generators are located inside the grid district
    _validate_sample_geno_location()

    _update_grids(
        edisgo_object=edisgo_object,
        imported_generators_mv=generators_mv,
        imported_generators_lv=generators_res_lv,
        **kwargs
    )

    if kwargs.get('p_target', None) is None:
        _validate_generation()

    # update time series if they were already set
    if not edisgo_object.timeseries.generators_active_power.empty:
        add_generators_timeseries(
            edisgo_obj=edisgo_object,
            generator_names=edisgo_object.topology.generators_df.index)


def _update_grids(
        edisgo_object,
        imported_generators_mv,
        imported_generators_lv,
        remove_decommissioned=True,
        update_existing=True,
        p_target=None,
        allowed_number_of_comp_per_lv_bus=2,
        **kwargs
):
    """
    Update network according to new generator dataset.

    Steps are:

        * Step 1: Update capacity of existing generators if `update_existing`
          is True, which it is by default.
        * Step 2: Remove decommissioned generators if `remove_decommissioned`
          is True, which it is by default.
        * Step 3: Integrate new MV generators.
        * Step 4: Integrate new LV generators.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    imported_generators_mv : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all MV generators.
        Index of the dataframe are the generator IDs.
        Columns are:

            * p_nom : float
                Nominal capacity in MW.
            * generator_type : str
                Type of generator (e.g. 'solar').
            * subtype : str
                Subtype of generator (e.g. 'solar_roof_mounted').
            * voltage_level : int
                Voltage level to connect to. Can be 4, 5, 6 or 7.
            * weather_cell_id : int
                Weather cell the generator is in. Only given for fluctuating
                generators.
            * geom : :shapely:`Shapely Point object<points>`
                Geolocation of generator. For CRS see config_grid.srid.
            * geom_em: :shapely:`Shapely Point object<points>`
                Geolocation of generator as given in energy map. For CRS see
                config_grid.srid.

    imported_generators_lv : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all LV generators.
        Index of the dataframe are the generator IDs.
        Columns are the same as in `imported_generators_mv` plus:

            * mvlv_subst_id : int
                ID of MV-LV substation in grid = grid, the generator will be
                connected to.

    remove_decommissioned : bool
        See :func:`edisgo.io.generators_import.oedb` for more information.
    update_existing : bool
        See :func:`edisgo.io.generators_import.oedb` for more information.
    p_target : dict
        See :func:`edisgo.io.generators_import.oedb` for more information.
    allowed_number_of_comp_per_lv_bus : int
        See :func:`edisgo.io.generators_import.oedb` for more information.

    """

    def _check_mv_generator_geom(generator_data):
        """
        Checks if a valid geom is available in dataset.

        If yes, this geom will be used.
        If not, geom from EnergyMap is used if available.

        Parameters
        ----------
        generator_data : :pandas:`pandas.Series<Series>`
            Series with among others 'geom' (geometry from open_eGo data
            processing) and 'geom_em' (geometry from EnergyMap).

        Returns
        -------
        :shapely:`Shapely Point object<points>` or None
            Geometry of generator. None, if no geometry is available.

        """
        # check if geom is available
        if generator_data.geom:
            return generator_data.geom
        else:
            # set geom to EnergyMap's geom, if available
            if generator_data.geom_em:
                logger.debug(
                    "Generator {} has no geom entry, EnergyMap's geom "
                    "entry will be used.".format(generator_data.name)
                )
                return generator_data.geom_em
        return None

    # set capacity difference threshold
    cap_diff_threshold = 10 ** -4

    # get all imported generators
    imported_gens = pd.concat(
        [imported_generators_lv, imported_generators_mv],
        sort=True
    )

    logger.debug("{} generators imported.".format(len(imported_gens)))

    # get existing generators and append ID column
    existing_gens = edisgo_object.topology.generators_df
    existing_gens["id"] = list(
        map(lambda _: int(_.split("_")[-1]), existing_gens.index)
    )

    logger.debug(
        "Cumulative generator capacity (existing): {} MW".format(
            round(existing_gens.p_nom.sum(), 1)
        )
    )

    # check if capacity of any of the imported generators is <= 0
    # (this may happen if dp is buggy) and remove generator if it is
    gens_to_remove = imported_gens[imported_gens.p_nom <= 0]
    for id in gens_to_remove.index:
        # remove from topology (if generator exists)
        if id in existing_gens.id.values:
            gen_name = existing_gens[existing_gens.id == id].index[0]
            edisgo_object.topology.remove_generator(gen_name)
            logger.warning(
                "Capacity of generator {} is <= 0, it is therefore "
                "removed. Check your data source.".format(gen_name)
            )
        # remove from imported generators
        imported_gens.drop(id, inplace=True)
        if id in imported_generators_mv.index:
            imported_generators_mv.drop(id, inplace=True)
        else:
            imported_generators_lv.drop(id, inplace=True)

    # =============================================
    # Step 1: Update existing MV and LV generators
    # =============================================

    if update_existing:
        # filter for generators that need to be updated (i.e. that
        # appear in the imported and existing generators dataframes)
        gens_to_update = existing_gens[
            existing_gens.id.isin(imported_gens.index.values)
        ]

        # calculate capacity difference between existing and imported
        # generators
        gens_to_update["cap_diff"] = (
                imported_gens.loc[gens_to_update.id, "p_nom"].values
                - gens_to_update.p_nom
        )
        # in case there are generators whose capacity does not match, update
        # their capacity
        gens_to_update_cap = gens_to_update[
            abs(gens_to_update.cap_diff) > cap_diff_threshold
            ]

        for id, row in gens_to_update_cap.iterrows():
            edisgo_object.topology._generators_df.loc[
                id, "p_nom"
            ] = imported_gens.loc[row["id"], "p_nom"]

        log_geno_count = len(gens_to_update_cap)
        log_geno_cap = gens_to_update_cap["cap_diff"].sum()
        logger.debug(
            "Capacities of {} of {} existing generators updated "
            "({} MW).".format(
                log_geno_count, len(gens_to_update), round(log_geno_cap, 1)
            )
        )

    # ==================================================
    # Step 2: Remove decommissioned MV and LV generators
    # ==================================================

    # filter for generators that do not appear in the imported but in
    # the existing generators dataframe
    decommissioned_gens = existing_gens[
        ~existing_gens.id.isin(imported_gens.index.values)
    ]

    if not decommissioned_gens.empty and remove_decommissioned:
        for gen in decommissioned_gens.index:
            edisgo_object.topology.remove_generator(gen)
        log_geno_cap = decommissioned_gens.p_nom.sum()
        log_geno_count = len(decommissioned_gens)
        logger.debug(
            "{} decommissioned generators removed ({} MW).".format(
                log_geno_count, round(log_geno_cap, 1)
            )
        )

    # ===================================
    # Step 3: Integrate new MV generators
    # ===================================

    new_gens_mv = imported_generators_mv[
        ~imported_generators_mv.index.isin(list(existing_gens.id))
    ]

    new_gens_lv = imported_generators_lv[
        ~imported_generators_lv.index.isin(list(existing_gens.id))
    ]

    if p_target is not None:
        def update_imported_gens(layer, imported_gens):
            def drop_generators(generator_list, gen_type, total_capacity):
                random.seed(42)
                while (generator_list[
                           generator_list['generator_type'] ==
                           gen_type].p_nom.sum() > total_capacity and
                       len(generator_list[
                               generator_list['generator_type'] ==
                               gen_type]) > 0):
                    generator_list.drop(
                        random.choice(
                            generator_list[
                                generator_list[
                                    'generator_type'] == gen_type].index),
                        inplace=True)

            for gen_type in p_target.keys():
                # Currently installed capacity
                existing_capacity = existing_gens[
                    existing_gens.index.isin(layer) &
                    (existing_gens['type'] == gen_type).values].p_nom.sum()
                # installed capacity in scenario
                expanded_capacity = existing_capacity + imported_gens[
                    imported_gens[
                        'generator_type'] == gen_type].p_nom.sum()
                # target capacity
                target_capacity = p_target[gen_type]
                # required expansion
                required_expansion = target_capacity - existing_capacity

                # No generators to be expanded
                if imported_gens[
                    imported_gens[
                        'generator_type'] == gen_type].p_nom.sum() == 0:
                    continue
                # Reduction in capacity over status quo, so skip all expansion
                if required_expansion <= 0:
                    imported_gens.drop(
                        imported_gens[
                            imported_gens['generator_type'] == gen_type].index,
                        inplace=True)
                    continue
                # More expansion than in NEP2035 required, keep all generators
                # and scale them up later
                if p_target[gen_type] >= expanded_capacity:
                    continue

                # Reduced expansion, remove some generators from expansion
                drop_generators(imported_gens, gen_type, required_expansion)

        new_gens = pd.concat([new_gens_lv, new_gens_mv], sort=True)
        update_imported_gens(
            edisgo_object.topology.generators_df.index,
            new_gens)

        # drop types not in p_target from new_gens
        for gen_type in new_gens.generator_type.unique():
            if not gen_type in p_target.keys():
                new_gens.drop(
                    new_gens[new_gens['generator_type'] == gen_type].index,
                    inplace=True)

        new_gens_lv = new_gens[new_gens.voltage_level.isin([6, 7])]
        new_gens_mv = new_gens[new_gens.voltage_level.isin([4, 5])]

    # iterate over new generators and create them
    number_new_gens = len(new_gens_mv)
    for id in new_gens_mv.index.sort_values(ascending=True):
        # check if geom is available, skip otherwise
        geom = _check_mv_generator_geom(new_gens_mv.loc[id, :])
        if geom is None:
            logger.warning(
                "Generator {} has no geom entry and will "
                "not be imported!".format(id)
            )
            new_gens_mv.drop(id)
            continue
        new_gens_mv.at[id, "geom"] = geom
        edisgo_object.topology.connect_to_mv(
            edisgo_object, dict(new_gens_mv.loc[id, :])
        )

    log_geno_count = len(new_gens_mv)
    log_geno_cap = new_gens_mv["p_nom"].sum()
    logger.debug(
        "{} of {} new MV generators added ({} MW).".format(
            log_geno_count, number_new_gens, round(log_geno_cap, 1)
        )
    )

    # ====================================
    # Step 4: Integrate new LV generators
    # ====================================

    # check if new generators can be allocated to an existing LV grid
    if not imported_generators_lv.empty:
        grid_ids = [_.id for _ in edisgo_object.topology._grids.values()]
        if not any(
                [
                    _ in grid_ids
                    for _ in list(imported_generators_lv["mvlv_subst_id"])
                ]
        ):
            logger.warning(
                "None of the imported LV generators can be allocated "
                "to an existing LV grid. Check compatibility of grid "
                "and generator datasets."
            )

    # iterate over new generators and create them
    for id in new_gens_lv.index.sort_values(ascending=True):
        edisgo_object.topology.connect_to_lv(
            edisgo_object,
            dict(new_gens_lv.loc[id, :]),
            allowed_number_of_comp_per_bus=allowed_number_of_comp_per_lv_bus
        )

    def scale_generators(gen_type, total_capacity):
        idx = edisgo_object.topology.generators_df['type'] == gen_type
        current_capacity = edisgo_object.topology.generators_df[
            idx].p_nom.sum()
        if current_capacity != 0:
            edisgo_object.topology.generators_df.loc[
                idx, 'p_nom'] *= total_capacity / current_capacity

    if p_target is not None:
        for gen_type, target_cap in p_target.items():
            scale_generators(gen_type, target_cap)

    log_geno_count = len(new_gens_lv)
    log_geno_cap = new_gens_lv["p_nom"].sum()
    logger.debug(
        "{} new LV generators added ({} MW).".format(
            log_geno_count, round(log_geno_cap, 1)
        )
    )

    for lv_grid in edisgo_object.topology.mv_grid.lv_grids:
        lv_loads = len(lv_grid.loads_df)
        lv_gens_voltage_level_7 = len(
            lv_grid.generators_df[
                lv_grid.generators_df.bus != lv_grid.station.index[0]
                ]
        )
        # warn if there are more generators than loads in LV grid
        if lv_gens_voltage_level_7 > lv_loads * 2:
            logger.debug(
                "There are {} generators (voltage level 7) but only {} "
                "loads in LV grid {}.".format(
                    lv_gens_voltage_level_7, lv_loads, lv_grid.id
                )
            )
