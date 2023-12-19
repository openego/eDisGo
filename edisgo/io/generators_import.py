from __future__ import annotations

import logging
import os
import random

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import saio

from sqlalchemy import func
from sqlalchemy.engine.base import Engine

from edisgo.io.db import get_srid_of_db_table, session_scope_egon_data
from edisgo.tools import session_scope
from edisgo.tools.geo import find_nearest_bus, proj2equidistant
from edisgo.tools.tools import (
    determine_bus_voltage_level,
    determine_grid_integration_voltage_level,
)

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

    from egoio.db_tables import model_draft, supply
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def oedb_legacy(edisgo_object, generator_scenario, **kwargs):
    """
    Gets generator park for specified scenario from oedb and integrates generators into
    the grid.

    The importer uses SQLAlchemy ORM objects. These are defined in
    `ego.io <https://github.com/openego/ego.io/tree/dev/egoio/db_tables/>`_.
    The data is imported from the tables
    `conventional power plants <https://openenergy-platform.org/dataedit/\
    view/supply/ego_dp_conv_powerplant>`_ and
    `renewable power plants <https://openenergy-platform.org/dataedit/\
    view/supply/ego_dp_res_powerplant>`_.

    When the generator data is retrieved, the following steps are conducted:

        * Step 1: Update capacity of existing generators if `update_existing` is True,
          which it is by default.
        * Step 2: Remove decommissioned generators if
          `remove_decommissioned` is True, which it is by default.
        * Step 3: Integrate new MV generators.
        * Step 4: Integrate new LV generators.

    For more information on how generators are integrated, see
    :attr:`~.network.topology.Topology.connect_to_mv` and
    :attr:`~.network.topology.Topology.connect_to_lv`.

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
        of all generators of that type in the future scenario, only some
        generators in the future scenario generator park are installed,
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
            )
            .filter(
                orm_conv_generators.columns.subst_id
                == edisgo_object.topology.mv_grid.id
            )
            .filter(orm_conv_generators.columns.voltage_level.in_([4, 5]))
            .filter(orm_conv_generators_version)
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
                orm_re_generators.columns.generation_type.label("generator_type"),
                orm_re_generators.columns.generation_subtype.label("subtype"),
                orm_re_generators.columns.voltage_level,
                orm_re_generators.columns.w_id.label("weather_cell_id"),
                func.ST_AsText(
                    func.ST_Transform(orm_re_generators.columns.rea_geom_new, srid)
                ).label("geom"),
                func.ST_AsText(
                    func.ST_Transform(orm_re_generators.columns.geom, srid)
                ).label("geom_em"),
            )
            .filter(
                orm_re_generators.columns.subst_id == edisgo_object.topology.mv_grid.id
            )
            .filter(orm_re_generators_version)
        )

        # extend basic query for MV generators and read data from db
        generators_mv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([4, 5])
        )
        gens_mv = pd.read_sql_query(
            generators_mv_sqla.statement, session.bind, index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        gens_mv.loc[gens_mv["subtype"].isnull(), "subtype"] = "unknown"

        # convert capacity from kW to MW
        gens_mv.p_nom = pd.to_numeric(gens_mv.p_nom) / 1e3

        # extend basic query for LV generators and read data from db
        generators_lv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([6, 7])
        )
        gens_lv = pd.read_sql_query(
            generators_lv_sqla.statement, session.bind, index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        gens_lv.loc[gens_lv["subtype"].isnull(), "subtype"] = "unknown"

        # convert capacity from kW to MW
        gens_lv.p_nom = pd.to_numeric(gens_lv.p_nom) / 1e3

        return gens_mv, gens_lv

    def _validate_generation():
        """
        Validate generation capacity in updated grids.

        The validation uses the cumulative capacity of all generators.

        """

        # set capacity difference threshold
        cap_diff_threshold = 10**-1

        capacity_imported = (
            generators_res_mv["p_nom"].sum()
            + generators_res_lv["p_nom"].sum()
            + generators_conv_mv["p_nom"].sum()
        )

        capacity_grid = edisgo_object.topology.generators_df.p_nom.sum()

        logger.debug(
            f"Cumulative generator capacity (updated): {round(capacity_imported, 1)} MW"
        )

        if abs(capacity_imported - capacity_grid) > cap_diff_threshold:
            raise ValueError(
                f"Cumulative capacity of imported generators ("
                f"{round(capacity_imported, 1)} MW) differs from cumulative capacity of"
                f" generators in updated grid ({round(capacity_grid, 1)} MW) by "
                f"{round(capacity_imported - capacity_grid, 1)} MW."
            )

        else:
            logger.debug("Cumulative capacity of imported generators validated.")

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
                    generators_res_mv["geom"]
                    .dropna()
                    .sample(n=1, random_state=42)
                    .values[0]
                ),
            )
            sample_lv_geno_geom_shp = transform(
                projection,
                wkt_loads(
                    generators_res_lv["geom"]
                    .dropna()
                    .sample(n=1, random_state=42)
                    .values[0]
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
        edisgo_object.config[oedb_data_source]["conv_generators_prefix"]
        + generator_scenario
        + edisgo_object.config[oedb_data_source]["conv_generators_suffix"]
    )
    orm_re_generators_name = (
        edisgo_object.config[oedb_data_source]["re_generators_prefix"]
        + generator_scenario
        + edisgo_object.config[oedb_data_source]["re_generators_suffix"]
    )

    if oedb_data_source == "model_draft":
        # import ORMs
        orm_conv_generators = model_draft.__getattribute__(orm_conv_generators_name)
        orm_re_generators = model_draft.__getattribute__(orm_re_generators_name)

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
        orm_re_generators_version = orm_re_generators.columns.version == data_version

    # get conventional and renewable generators
    with session_scope() as session:
        generators_conv_mv = _import_conv_generators(session)
        generators_res_mv, generators_res_lv = _import_res_generators(session)

    generators_mv = pd.concat(
        [
            generators_conv_mv,
            generators_res_mv,
        ]
    )

    # validate that imported generators are located inside the grid district
    _validate_sample_geno_location()

    _update_grids(
        edisgo_object=edisgo_object,
        imported_generators_mv=generators_mv,
        imported_generators_lv=generators_res_lv,
        **kwargs,
    )

    if kwargs.get("p_target", None) is None:
        _validate_generation()


def _update_grids(
    edisgo_object,
    imported_generators_mv,
    imported_generators_lv,
    remove_decommissioned=True,
    update_existing=True,
    p_target=None,
    allowed_number_of_comp_per_lv_bus=2,
    **kwargs,
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

            * mvlv_subst_id : int or float
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
    cap_diff_threshold = 10**-4

    # get all imported generators
    imported_gens = pd.concat(
        [imported_generators_lv, imported_generators_mv], sort=True
    )

    logger.debug(f"{len(imported_gens)} generators imported.")

    # get existing generators and append ID column
    existing_gens = edisgo_object.topology.generators_df
    existing_gens["id"] = list(
        map(lambda _: int(_.split("_")[-1]), existing_gens.index)
    )

    logger.debug(
        "Cumulative generator capacity (existing): "
        f"{round(existing_gens.p_nom.sum(), 1)} MW"
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
            imported_gens.loc[gens_to_update.id, "p_nom"].values - gens_to_update.p_nom
        )
        # in case there are generators whose capacity does not match, update
        # their capacity
        gens_to_update_cap = gens_to_update[
            abs(gens_to_update.cap_diff) > cap_diff_threshold
        ]

        for id, row in gens_to_update_cap.iterrows():
            edisgo_object.topology._generators_df.loc[id, "p_nom"] = imported_gens.loc[
                row["id"], "p_nom"
            ]

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
    new_gens_mv = new_gens_mv.assign(
        p=new_gens_mv.p_nom,
    )

    new_gens_lv = imported_generators_lv[
        ~imported_generators_lv.index.isin(list(existing_gens.id))
    ]
    new_gens_lv = new_gens_lv.assign(
        p=new_gens_lv.p_nom,
    )

    if p_target is not None:

        def update_imported_gens(layer, imported_gens):
            def drop_generators(generator_list, gen_type, total_capacity):
                random.seed(42)
                while (
                    generator_list[
                        generator_list["generator_type"] == gen_type
                    ].p_nom.sum()
                    > total_capacity
                    and len(
                        generator_list[generator_list["generator_type"] == gen_type]
                    )
                    > 0
                ):
                    generator_list.drop(
                        random.choice(
                            generator_list[
                                generator_list["generator_type"] == gen_type
                            ].index
                        ),
                        inplace=True,
                    )

            for gen_type in p_target.keys():
                # Currently installed capacity
                existing_capacity = existing_gens[
                    existing_gens.index.isin(layer)
                    & (existing_gens["type"] == gen_type).values
                ].p_nom.sum()
                # installed capacity in scenario
                expanded_capacity = (
                    existing_capacity
                    + imported_gens[
                        imported_gens["generator_type"] == gen_type
                    ].p_nom.sum()
                )
                # target capacity
                target_capacity = p_target[gen_type]
                # required expansion
                required_expansion = target_capacity - existing_capacity

                # No generators to be expanded
                if (
                    imported_gens[
                        imported_gens["generator_type"] == gen_type
                    ].p_nom.sum()
                    == 0
                ):
                    continue
                # Reduction in capacity over status quo, so skip all expansion
                if required_expansion <= 0:
                    imported_gens.drop(
                        imported_gens[
                            imported_gens["generator_type"] == gen_type
                        ].index,
                        inplace=True,
                    )
                    continue
                # More expansion than in NEP2035 required, keep all generators
                # and scale them up later
                if p_target[gen_type] >= expanded_capacity:
                    continue

                # Reduced expansion, remove some generators from expansion
                drop_generators(imported_gens, gen_type, required_expansion)

        new_gens = pd.concat([new_gens_lv, new_gens_mv], sort=True)
        update_imported_gens(edisgo_object.topology.generators_df.index, new_gens)

        # drop types not in p_target from new_gens
        for gen_type in new_gens.generator_type.unique():
            if gen_type not in p_target.keys():
                new_gens.drop(
                    new_gens[new_gens["generator_type"] == gen_type].index,
                    inplace=True,
                )

        new_gens_lv = new_gens[new_gens.voltage_level.isin([6, 7])]
        new_gens_mv = new_gens[new_gens.voltage_level.isin([4, 5])]

    # iterate over new generators and create them
    number_new_gens = len(new_gens_mv)
    for id in new_gens_mv.index.sort_values(ascending=True):
        # check if geom is available, skip otherwise
        geom = _check_mv_generator_geom(new_gens_mv.loc[id, :])
        if geom is None:
            logger.warning(
                "Generator {} has no geom entry and will not be imported!".format(id)
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
        grid_ids = edisgo_object.topology._lv_grid_ids
        if not any(
            [
                int(_) in grid_ids
                for _ in list(imported_generators_lv["mvlv_subst_id"])
                if not np.isnan(_)
            ]
        ):
            logger.warning(
                "None of the imported LV generators can be allocated "
                "to an existing LV grid. Check compatibility of grid "
                "and generator datasets."
            )

    substations = edisgo_object.topology.buses_df.loc[
        edisgo_object.topology.transformers_df.bus1.unique()
    ]

    new_gens_lv.geom = new_gens_lv.geom.apply(wkt_loads)

    new_gens_lv = gpd.GeoDataFrame(
        new_gens_lv,
        geometry="geom",
        crs=f"EPSG:{edisgo_object.topology.grid_district['srid']}",
    )

    # iterate over new generators and create them
    for id in new_gens_lv.index.sort_values(ascending=True):
        comp_data = dict(new_gens_lv.loc[id, :])
        try:
            nearest_substation, _ = find_nearest_bus(comp_data["geom"], substations)
            comp_data["mvlv_subst_id"] = int(nearest_substation.split("_")[-2])
        except AttributeError:
            pass
        edisgo_object.topology.connect_to_lv(
            edisgo_object,
            comp_data,
            allowed_number_of_comp_per_bus=allowed_number_of_comp_per_lv_bus,
        )

    def scale_generators(gen_type, total_capacity):
        idx = edisgo_object.topology.generators_df["type"] == gen_type
        current_capacity = edisgo_object.topology.generators_df[idx].p_nom.sum()
        if current_capacity != 0:
            edisgo_object.topology.generators_df.loc[idx, "p_nom"] *= (
                total_capacity / current_capacity
            )

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
            lv_grid.generators_df[lv_grid.generators_df.bus != lv_grid.station.index[0]]
        )
        # warn if there are more generators than loads in LV grid
        if lv_gens_voltage_level_7 > lv_loads * 2:
            logger.debug(
                "There are {} generators (voltage level 7) but only {} "
                "loads in LV grid {}.".format(
                    lv_gens_voltage_level_7, lv_loads, lv_grid.id
                )
            )


def oedb(
    edisgo_object: EDisGo,
    scenario: str,
    engine: Engine,
    max_capacity=20,
):
    """
    Gets generator park for specified scenario from oedb and integrates generators into
    the grid.

    The data is imported from the tables supply.egon_chp_plants,
    supply.egon_power_plants and supply.egon_power_plants_pv_roof_building.

    For the grid integration it is distinguished between PV rooftop plants and all
    other power plants.
    For PV rooftop the following steps are conducted:

    * Removes decommissioned PV rooftop plants (plants whose source ID cannot
      be matched to a source ID of an existing plant).
    * Updates existing PV rooftop plants. The following two cases are distinguished:

      * Nominal power increases: It is checked, if plant needs to be connected to a
        higher voltage level and if that is the case, the existing plant is removed from
        the grid and the new one integrated based on the geolocation.
      * Nominal power decreases: Nominal power of existing plant is overwritten.
    * Integrates new PV rooftop plants at corresponding building ID. If the plant needs
      to be connected to a higher voltage level than the building, it is integrated
      based on the geolocation.

    For all other power plants the following steps are conducted:

    * Removes decommissioned power and CHP plants (all plants that do not have a source
      ID or whose source ID can not be matched to a new plant and are not of subtype
      pv_rooftop, as these are handled in a separate function)
    * Updates existing power plants (plants whose source ID is in
      can be matched; solar, wind and CHP plants never have a source ID in
      the future scenarios and are therefore never updated). The following two cases
      are distinguished:

      * Nominal power increases: It is checked, if plant needs to be connected to a
        higher voltage level and if that is the case, the existing plant is removed from
        the grid and the new one integrated based on the geolocation.
      * Nominal power decreases: Nominal power of existing plant is overwritten.
    * Integrates new power and CHP plants based on the geolocation.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve generator data. Possible options
        are "eGon2035" and "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    max_capacity : float
        Maximum capacity in MW of power plants to retrieve from database. In general,
        the generators that are retrieved from the database are selected based on the
        voltage level they are in. In some cases, the voltage level is not correct as
        it was wrongly set in the MaStR dataset. To avoid having unrealistically large
        generators in the grids, an upper limit is also set. Per default this is 20 MW.

    Notes
    ------
    Note, that PV rooftop plants are queried using the building IDs not the MV grid ID
    as in egon_data buildings are mapped to a grid based on the
    zensus cell they are in whereas in ding0 buildings are mapped to a grid based on
    the geolocation. As it can happen that buildings lie outside an MV grid but within
    a zensus cell that is assigned to that MV grid, they are mapped differently in
    egon_data and ding0, and it is therefore better to query using the building IDs.

    """

    def _get_egon_power_plants():
        with session_scope_egon_data(engine) as session:
            srid_table = get_srid_of_db_table(session, egon_power_plants.geom)
            query = (
                session.query(
                    egon_power_plants.id.label("generator_id"),
                    egon_power_plants.source_id,
                    egon_power_plants.carrier.label("type"),
                    egon_power_plants.el_capacity.label("p_nom"),
                    egon_power_plants.weather_cell_id,
                    egon_power_plants.geom,
                )
                .filter(
                    egon_power_plants.scenario == scenario,
                    egon_power_plants.voltage_level >= 4,
                    egon_power_plants.el_capacity <= max_capacity,
                    egon_power_plants.bus_id == edisgo_object.topology.id,
                )
                .order_by(egon_power_plants.id)
            )
            power_plants_gdf = gpd.read_postgis(
                sql=query.statement, con=engine, crs=f"EPSG:{srid_table}"
            ).to_crs(srid_edisgo)
        # rename wind_onshore to wind
        power_plants_gdf["type"] = power_plants_gdf["type"].str.replace("_onshore", "")
        # set subtype
        mapping = {
            "wind": "wind_onshore",
            "solar": "solar_ground_mounted",
        }
        power_plants_gdf = power_plants_gdf.assign(
            subtype=power_plants_gdf["type"].map(mapping)
        )
        # unwrap source ID
        if not power_plants_gdf.empty:
            power_plants_gdf["source_id"] = power_plants_gdf.apply(
                lambda _: (
                    list(_["source_id"].values())[0]
                    if isinstance(_["source_id"], dict)
                    else None
                ),
                axis=1,
            )
        return power_plants_gdf

    def _get_egon_pv_rooftop():
        # egon_power_plants_pv_roof_building - queried using building IDs instead of
        # grid ID because it can happen that buildings lie outside an MV grid but within
        # a zensus cell that is assigned to that MV grid and are therefore sometimes
        # mapped to the MV grid they lie within and sometimes to the MV grid the zensus
        # cell is mapped to
        building_ids = edisgo_object.topology.loads_df.building_id.unique()
        with session_scope_egon_data(engine) as session:
            query = (
                session.query(
                    egon_power_plants_pv_roof_building.index.label("generator_id"),
                    egon_power_plants_pv_roof_building.building_id,
                    egon_power_plants_pv_roof_building.gens_id.label("source_id"),
                    egon_power_plants_pv_roof_building.capacity.label("p_nom"),
                    egon_power_plants_pv_roof_building.weather_cell_id,
                )
                .filter(
                    egon_power_plants_pv_roof_building.scenario == scenario,
                    egon_power_plants_pv_roof_building.building_id.in_(building_ids),
                    egon_power_plants_pv_roof_building.voltage_level >= 4,
                    egon_power_plants_pv_roof_building.capacity <= max_capacity,
                )
                .order_by(egon_power_plants_pv_roof_building.index)
            )
            pv_roof_df = pd.read_sql(sql=query.statement, con=engine)
        # add type and subtype
        pv_roof_df = pv_roof_df.assign(
            type="solar",
            subtype="pv_rooftop",
        )
        return pv_roof_df

    def _get_egon_chp_plants():
        with session_scope_egon_data(engine) as session:
            srid_table = get_srid_of_db_table(session, egon_chp_plants.geom)
            query = (
                session.query(
                    egon_chp_plants.id.label("generator_id"),
                    egon_chp_plants.carrier.label("type"),
                    egon_chp_plants.district_heating_area_id.label(
                        "district_heating_id"
                    ),
                    egon_chp_plants.el_capacity.label("p_nom"),
                    egon_chp_plants.th_capacity.label("p_nom_th"),
                    egon_chp_plants.geom,
                )
                .filter(
                    egon_chp_plants.scenario == scenario,
                    egon_chp_plants.voltage_level >= 4,
                    egon_chp_plants.el_capacity <= max_capacity,
                    egon_chp_plants.electrical_bus_id == edisgo_object.topology.id,
                )
                .order_by(egon_chp_plants.id)
            )
            chp_gdf = gpd.read_postgis(
                sql=query.statement, con=query.session.bind, crs=f"EPSG:{srid_table}"
            ).to_crs(srid_edisgo)
        return chp_gdf

    saio.register_schema("supply", engine)
    from saio.supply import (
        egon_chp_plants,
        egon_power_plants,
        egon_power_plants_pv_roof_building,
    )

    # get generator data from database
    srid_edisgo = edisgo_object.topology.grid_district["srid"]
    pv_rooftop_df = _get_egon_pv_rooftop()
    power_plants_gdf = _get_egon_power_plants()
    chp_gdf = _get_egon_chp_plants()

    # determine number of generators and installed capacity in future scenario
    # for validation of grid integration
    total_p_nom_scenario = (
        pv_rooftop_df.p_nom.sum() + power_plants_gdf.p_nom.sum() + chp_gdf.p_nom.sum()
    )
    total_gen_count_scenario = len(pv_rooftop_df) + len(power_plants_gdf) + len(chp_gdf)

    # integrate into grid (including removal of decommissioned plants and update of
    # still existing power plants)
    _integrate_pv_rooftop(edisgo_object, pv_rooftop_df)
    _integrate_power_and_chp_plants(edisgo_object, power_plants_gdf, chp_gdf)

    # check number of generators and installed capacity in grid
    gens_in_grid = edisgo_object.topology.generators_df
    if not len(gens_in_grid) == total_gen_count_scenario:
        raise ValueError(
            f"Number of power plants in future scenario is not correct. Should be "
            f"{total_gen_count_scenario} instead of {len(gens_in_grid)}."
        )
    if not np.isclose(gens_in_grid.p_nom.sum(), total_p_nom_scenario, atol=1e-4):
        raise ValueError(
            f"Capacity of power plants in future scenario not correct. Should be "
            f"{total_p_nom_scenario} instead of "
            f"{gens_in_grid.p_nom.sum()}."
        )

    return


def _integrate_pv_rooftop(edisgo_object, pv_rooftop_df):
    """
    This function updates generator park for PV rooftop plants.
    See function :func:`~.io.generators_import.oedb` for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    pv_rooftop_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing data on PV rooftop plants per building.
        Columns are:

            * p_nom : float
                Nominal power in MW.
            * building_id : int
                Building ID of the building the PV plant is allocated.
            * generator_id : int
                ID of the PV plant from database.
            * type : str
                Generator type, which here is always "solar".
            * subtype
                Further specification of generator type, which here is always
                "pv_rooftop".
            * weather_cell_id : int
                Weather cell the PV plant is in used to obtain the potential feed-in
                time series.
            * source_id : int
                MaStR ID of the PV plant.

    """
    # match building ID to existing solar generators
    loads_df = edisgo_object.topology.loads_df
    busses_building_id = (
        loads_df[loads_df.type == "conventional_load"]
        .drop_duplicates(subset=["building_id"])
        .set_index("bus")
        .loc[:, ["building_id"]]
    )
    gens_df = edisgo_object.topology.generators_df[
        edisgo_object.topology.generators_df.subtype == "pv_rooftop"
    ].copy()
    gens_df_building_id = gens_df.loc[:, ["bus"]].join(
        busses_building_id, how="left", on="bus"
    )
    # using update to make sure to not overwrite existing building ID information
    if "building_id" not in gens_df.columns:
        gens_df["building_id"] = None
    gens_df.update(gens_df_building_id, overwrite=False)

    # remove decommissioned PV rooftop plants
    gens_decommissioned = gens_df[
        ~gens_df.source_id.isin(pv_rooftop_df.source_id.unique())
    ]
    for gen in gens_decommissioned.index:
        edisgo_object.remove_component(comp_type="generator", comp_name=gen)

    # update existing PV rooftop plants
    gens_existing = gens_df[gens_df.source_id.isin(pv_rooftop_df.source_id.unique())]
    # merge new information
    gens_existing.index.name = "gen_name"
    pv_rooftop_df.index.name = "gen_index_new"
    gens_existing = pd.merge(
        gens_existing.reset_index(),
        pv_rooftop_df.reset_index(),
        how="left",
        on="source_id",
        suffixes=("_old", ""),
    ).set_index("gen_name")
    # add building id
    edisgo_object.topology.generators_df.loc[
        gens_existing.index, "building_id"
    ] = gens_existing.building_id
    # update plants where capacity decreased
    gens_decreased_cap = gens_existing.query("p_nom < p_nom_old")
    if len(gens_decreased_cap) > 0:
        edisgo_object.topology.generators_df.loc[
            gens_decreased_cap.index, "p_nom"
        ] = gens_decreased_cap.p_nom
    # update plants where capacity increased
    gens_increased_cap = gens_existing.query("p_nom > p_nom_old")
    for gen in gens_increased_cap.index:
        voltage_level_new = determine_grid_integration_voltage_level(
            edisgo_object, gens_increased_cap.at[gen, "p_nom"]
        )
        voltage_level_old = determine_bus_voltage_level(
            edisgo_object, gens_increased_cap.at[gen, "bus"]
        )
        if voltage_level_new >= voltage_level_old:
            # simply update p_nom if plant doesn't need to be connected to higher
            # voltage level
            edisgo_object.topology.generators_df.at[
                gen, "p_nom"
            ] = gens_increased_cap.at[gen, "p_nom"]
        else:
            # if plant needs to be connected to higher voltage level, remove existing
            # plant and integrate new component based on geolocation
            bus = gens_increased_cap.at[gen, "bus"]
            x_coord = edisgo_object.topology.buses_df.at[bus, "x"]
            y_coord = edisgo_object.topology.buses_df.at[bus, "y"]
            edisgo_object.remove_component(comp_type="generator", comp_name=gen)
            edisgo_object.integrate_component_based_on_geolocation(
                comp_type="generator",
                voltage_level=voltage_level_new,
                geolocation=(
                    x_coord,
                    y_coord,
                ),
                add_ts=False,
                generator_id=gens_increased_cap.at[gen, "generator_id"],
                p_nom=gens_increased_cap.at[gen, "p_nom"],
                building_id=gens_increased_cap.at[gen, "building_id"],
                generator_type=gens_increased_cap.at[gen, "type"],
                subtype=gens_increased_cap.at[gen, "subtype"],
                weather_cell_id=gens_increased_cap.at[gen, "weather_cell_id"],
                source_id=gens_increased_cap.at[gen, "source_id"],
            )

    # integrate new PV rooftop plants into grid
    new_pv_rooftop_plants = pv_rooftop_df[
        ~pv_rooftop_df.index.isin(gens_existing.gen_index_new)
    ]
    if len(new_pv_rooftop_plants) > 0:
        _, new_pv_own_grid_conn = _integrate_new_pv_rooftop_to_buildings(
            edisgo_object, new_pv_rooftop_plants
        )
    else:
        new_pv_own_grid_conn = []

    # check number and installed capacity of PV rooftop plants in grid
    pv_rooftop_gens_in_grid = edisgo_object.topology.generators_df[
        edisgo_object.topology.generators_df.subtype == "pv_rooftop"
    ]
    if not len(pv_rooftop_gens_in_grid) == len(pv_rooftop_df):
        raise ValueError(
            f"Number of PV rooftop plants in future scenario is not correct. Should be "
            f"{len(pv_rooftop_df)} instead of {len(pv_rooftop_gens_in_grid)}."
        )
    if not np.isclose(
        pv_rooftop_gens_in_grid.p_nom.sum(), pv_rooftop_df.p_nom.sum(), atol=1e-4
    ):
        raise ValueError(
            f"Capacity of PV rooftop plants in future scenario is not correct. Should "
            f"be {pv_rooftop_df.p_nom.sum()} instead of "
            f"{pv_rooftop_gens_in_grid.p_nom.sum()}."
        )

    # logging messages
    logger.debug(
        f"{pv_rooftop_gens_in_grid.p_nom.sum():.2f} MW of PV rooftop plants "
        f"integrated. Of this, {gens_existing.p_nom.sum():.2f} MW could be matched to "
        f"an existing PV rooftop plant."
    )
    if len(new_pv_own_grid_conn) > 0:
        logger.debug(
            f"Of the PV rooftop plants that could not be matched to an existing PV "
            f"plant, "
            f"{sum(pv_rooftop_gens_in_grid.loc[new_pv_own_grid_conn, 'p_nom']):.2f} "
            f"MW was integrated at a new bus."
        )


def _integrate_new_pv_rooftop_to_buildings(edisgo_object, pv_rooftop_df):
    """
    Integrates new PV rooftop plants based on corresponding building ID.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    pv_rooftop_df : :pandas:`pandas.DataFrame<DataFrame>`
        See :attr:`~.io.generators_import._integrate_pv_rooftop` for more information.

    Returns
    -------
    (list(str), list(str))
        Two lists with names (as in index of
        :attr:`~.network.topology.Topology.generators_df`) of all integrated PV rooftop
        plants and PV rooftop plants integrated to a different grid connection point
        than the building.

    """
    # join busses corresponding to building ID
    loads_df = edisgo_object.topology.loads_df
    building_id_busses = (
        loads_df[loads_df.type == "conventional_load"]
        .drop_duplicates(subset=["building_id"])
        .set_index("building_id")
        .loc[:, ["bus"]]
    )
    pv_rooftop_df = pv_rooftop_df.join(building_id_busses, how="left", on="building_id")

    # add further information needed in generators_df
    pv_rooftop_df["control"] = "PQ"
    # add generator name as index
    pv_rooftop_df["index"] = pv_rooftop_df.apply(
        lambda _: f"Generator_pv_rooftop_{_.building_id}", axis=1
    )
    pv_rooftop_df.set_index("index", drop=True, inplace=True)

    # add voltage level
    for gen in pv_rooftop_df.index:
        pv_rooftop_df.at[
            gen, "voltage_level"
        ] = determine_grid_integration_voltage_level(
            edisgo_object, pv_rooftop_df.at[gen, "p_nom"]
        )

    # check for duplicated generator names and choose random name for duplicates
    tmp = pv_rooftop_df.index.append(edisgo_object.topology.storage_units_df.index)
    duplicated_indices = tmp[tmp.duplicated()]
    for duplicate in duplicated_indices:
        # find unique name
        random.seed(a=duplicate)
        new_name = duplicate
        while new_name in tmp:
            new_name = f"{duplicate}_{random.randint(10 ** 1, 10 ** 2)}"
        # change name in pv_rooftop_df
        pv_rooftop_df.rename(index={duplicate: new_name}, inplace=True)

    # filter PV plants that are too large to be integrated into LV
    pv_rooftop_large = pv_rooftop_df[pv_rooftop_df.voltage_level < 7]
    pv_rooftop_small = pv_rooftop_df[pv_rooftop_df.voltage_level == 7]

    # integrate small batteries at buildings
    cols = [
        "bus",
        "control",
        "p_nom",
        "weather_cell_id",
        "building_id",
        "type",
        "subtype",
        "source_id",
    ]
    edisgo_object.topology.generators_df = pd.concat(
        [edisgo_object.topology.generators_df, pv_rooftop_small.loc[:, cols]]
    )
    integrated_plants = pv_rooftop_small.index

    # integrate larger PV rooftop plants - if load is already connected to
    # higher voltage level it can be integrated at same bus, otherwise it is
    # integrated based on geolocation
    integrated_plants_own_grid_conn = pd.Index([])
    for pv_pp in pv_rooftop_large.index:
        # check if building is already connected to a voltage level equal to or
        # higher than the voltage level the PV plant should be connected to
        bus = pv_rooftop_large.at[pv_pp, "bus"]
        voltage_level_bus = determine_bus_voltage_level(edisgo_object, bus)
        voltage_level_pv = pv_rooftop_large.at[pv_pp, "voltage_level"]

        if voltage_level_pv >= voltage_level_bus:
            # integrate at same bus as load
            edisgo_object.topology.generators_df = pd.concat(
                [
                    edisgo_object.topology.generators_df,
                    pv_rooftop_large.loc[[pv_pp], cols],
                ]
            )
            integrated_plants = integrated_plants.append(pd.Index([pv_pp]))
        else:
            # integrate based on geolocation
            pv_pp_name = edisgo_object.integrate_component_based_on_geolocation(
                comp_type="generator",
                voltage_level=voltage_level_pv,
                geolocation=(
                    edisgo_object.topology.buses_df.at[bus, "x"],
                    edisgo_object.topology.buses_df.at[bus, "y"],
                ),
                add_ts=False,
                generator_id=pv_rooftop_large.at[pv_pp, "generator_id"],
                p_nom=pv_rooftop_large.at[pv_pp, "p_nom"],
                building_id=pv_rooftop_large.at[pv_pp, "building_id"],
                generator_type=pv_rooftop_large.at[pv_pp, "type"],
                subtype=pv_rooftop_large.at[pv_pp, "subtype"],
                weather_cell_id=pv_rooftop_large.at[pv_pp, "weather_cell_id"],
                source_id=pv_rooftop_large.at[pv_pp, "source_id"],
            )
            integrated_plants = integrated_plants.append(pd.Index([pv_pp_name]))
            integrated_plants_own_grid_conn = integrated_plants_own_grid_conn.append(
                pd.Index([pv_pp_name])
            )

    # check if all PV plants were integrated
    if not len(pv_rooftop_df) == len(integrated_plants):
        raise ValueError("Not all PV rooftop plants could be integrated into the grid.")

    return integrated_plants, integrated_plants_own_grid_conn


def _integrate_power_and_chp_plants(edisgo_object, power_plants_gdf, chp_gdf):
    """
    This function updates generator park for all power plants except PV rooftop.
    See function :func:`~.io.generators_import.oedb` for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    power_plants_gdf : :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
        Dataframe containing data on power plants.
        Columns are:

            * p_nom : float
                Nominal power in MW.
            * generator_id : int
                ID of the power plant from database.
            * type : str
                Generator type, e.g. "wind".
            * subtype
                Further specification of generator type, e.g. "wind_onshore".
            * weather_cell_id : int
                Weather cell the power plant is in used to obtain the potential feed-in
                time series. Only given for solar and wind generators.
            * source_id : int
                MaStR ID of the power plant.
            * geom : geometry
                Geolocation of power plant.
    chp_gdf : :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
        Dataframe containing data on CHP plants.
        Columns are:

            * p_nom : float
                Nominal power in MW.
            * p_nom_th : float
                Thermal nominal power in MW.
            * generator_id : int
                ID of the CHP plant from database.
            * type : str
                Generator type, e.g. "gas".
            * district_heating_id : int
                ID of district heating network the CHP plant is in.
            * geom : geometry
                Geolocation of power plant.

    """

    def _integrate_new_chp_plant(edisgo_object, comp_data):
        edisgo_object.integrate_component_based_on_geolocation(
            comp_type="generator",
            generator_id=comp_data.at["generator_id"],
            geolocation=comp_data.at["geom"],
            add_ts=False,
            p_nom=comp_data.at["p_nom"],
            p_nom_th=comp_data.at["p_nom_th"],
            generator_type=comp_data.at["type"],
            district_heating_id=comp_data.at["district_heating_id"],
        )

    def _integrate_new_power_plant(edisgo_object, comp_data):
        edisgo_object.integrate_component_based_on_geolocation(
            comp_type="generator",
            generator_id=comp_data.at["generator_id"],
            geolocation=comp_data.at["geom"],
            add_ts=False,
            p_nom=comp_data.at["p_nom"],
            generator_type=comp_data.at["type"],
            subtype=comp_data.at["subtype"],
            weather_cell_id=comp_data.at["weather_cell_id"],
            source_id=comp_data.at["source_id"],
        )

    # determine number of generators and installed capacity in future scenario
    # for validation of grid integration
    total_p_nom_scenario = power_plants_gdf.p_nom.sum() + chp_gdf.p_nom.sum()
    total_gen_count_scenario = len(power_plants_gdf) + len(chp_gdf)

    # remove all power plants that are not PV rooftop and do not have a source ID
    gens_df = edisgo_object.topology.generators_df[
        edisgo_object.topology.generators_df.subtype != "pv_rooftop"
    ].copy()
    if "source_id" not in gens_df.columns:
        gens_df["source_id"] = None
    gens_decommissioned = gens_df[gens_df.source_id.isna()]
    for gen in gens_decommissioned.index:
        edisgo_object.remove_component(comp_type="generator", comp_name=gen)

    # try matching power plants with source ID, to update power plants that exist in
    # status quo and future scenario
    existing_gens_with_source = gens_df[~gens_df.source_id.isna()]
    if len(existing_gens_with_source) > 0:
        # join dataframes at source ID
        existing_gens_with_source.index.name = "gen_name"
        power_plants_gdf.index.name = "gen_index_new"
        existing_gens_with_source_matched = pd.merge(
            existing_gens_with_source.reset_index(),
            power_plants_gdf.reset_index(),
            how="inner",
            on="source_id",
            suffixes=("_old", ""),
        ).set_index("gen_name")

        # remove existing gens where source ID could not be matched
        existing_gens_without_source_matched = [
            _
            for _ in existing_gens_with_source.index
            if _ not in existing_gens_with_source_matched.index
        ]
        for gen in existing_gens_without_source_matched:
            edisgo_object.remove_component(comp_type="generator", comp_name=gen)

        # where source ID could be matched, check if capacity increased or decreased
        # update plants where capacity decreased
        gens_decreased_cap = existing_gens_with_source_matched.query(
            "p_nom < p_nom_old"
        )
        if len(gens_decreased_cap) > 0:
            edisgo_object.topology.generators_df.loc[
                gens_decreased_cap.index, "p_nom"
            ] = gens_decreased_cap.p_nom
        # update plants where capacity increased
        gens_increased_cap = existing_gens_with_source_matched.query(
            "p_nom > p_nom_old"
        )
        for gen in gens_increased_cap.index:
            voltage_level_new = determine_grid_integration_voltage_level(
                edisgo_object, gens_increased_cap.at[gen, "p_nom"]
            )
            voltage_level_old = determine_bus_voltage_level(
                edisgo_object, gens_increased_cap.at[gen, "bus"]
            )
            if voltage_level_new >= voltage_level_old:
                # simply update p_nom if plant doesn't need to be connected to higher
                # voltage level
                edisgo_object.topology.generators_df.at[
                    gen, "p_nom"
                ] = gens_increased_cap.at[gen, "p_nom"]
            else:
                # if plant needs to be connected to higher voltage level, remove
                # existing plant and integrate new component based on geolocation
                edisgo_object.remove_component(comp_type="generator", comp_name=gen)
                _integrate_new_power_plant(edisgo_object, gens_increased_cap.loc[gen])
    else:
        existing_gens_with_source_matched = pd.DataFrame(
            columns=["gen_index_new", "p_nom"]
        )

    # gens where source ID could not be matched are all new
    new_power_plants = power_plants_gdf[
        ~power_plants_gdf.index.isin(existing_gens_with_source_matched.gen_index_new)
    ]
    for gen in new_power_plants.index:
        _integrate_new_power_plant(edisgo_object, new_power_plants.loc[gen])

    # add all CHP plants based on geolocation
    for gen in chp_gdf.index:
        _integrate_new_chp_plant(edisgo_object, chp_gdf.loc[gen])

    # check number of power and CHP plants in grid as well as installed capacity
    gens_in_grid = edisgo_object.topology.generators_df[
        edisgo_object.topology.generators_df.subtype != "pv_rooftop"
    ]
    if not len(gens_in_grid) == total_gen_count_scenario:
        raise ValueError(
            f"Number of power plants in future scenario is not correct. Should be "
            f"{total_gen_count_scenario} instead of {len(gens_in_grid)}."
        )
    if not np.isclose(gens_in_grid.p_nom.sum(), total_p_nom_scenario, atol=1e-4):
        raise ValueError(
            f"Capacity of power plants in future scenario not correct. Should be "
            f"{total_p_nom_scenario} instead of "
            f"{gens_in_grid.p_nom.sum()}."
        )

    # logging messages
    cap_matched = existing_gens_with_source_matched.p_nom.sum()
    logger.debug(
        f"{total_p_nom_scenario:.2f} MW of power and CHP plants integrated. Of this, "
        f"{cap_matched:.2f} MW could be matched to existing power plants."
    )
