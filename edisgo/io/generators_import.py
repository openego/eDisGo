import pandas as pd
from sqlalchemy import func
import logging
import os

from egoio.db_tables import model_draft, supply
from edisgo.tools import session_scope

from edisgo.network.connect import add_and_connect_mv_generator, \
    add_and_connect_lv_generator
from edisgo.tools.geo import proj2equidistant

logger = logging.getLogger('edisgo')

if 'READTHEDOCS' not in os.environ:
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


def oedb(edisgo_object):
    """Import generator data from the Open Energy Database (OEDB).

    The importer uses SQLAlchemy ORM objects.
    These are defined in ego.io,
    see https://github.com/openego/ego.io/tree/dev/egoio/db_tables

    Parameters
    ----------
    network: :class:`~.network.topology.Topology`
        The eDisGo container object

    Notes
    ------
    Right now only solar and wind generators can be imported.

    """

    def _import_conv_generators(session):
        """
        Import conventional (conv) generators from oedb.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            List of medium-voltage generators.

        Notes
        -----
        You can find a full list of columns in
        :func:`edisgo.io.import_data._update_grids`.

        """

        # build query
        generators_sqla = session.query(
            orm_conv_generators.columns.id,
            orm_conv_generators.columns.subst_id,
            orm_conv_generators.columns.la_id,
            orm_conv_generators.columns.capacity,
            orm_conv_generators.columns.type,
            orm_conv_generators.columns.voltage_level,
            orm_conv_generators.columns.fuel,
            func.ST_AsText(func.ST_Transform(
                orm_conv_generators.columns.geom, srid))
        ). \
            filter(orm_conv_generators.columns.subst_id ==
                   edisgo_object.topology.mv_grid.id). \
            filter(orm_conv_generators.columns.voltage_level.in_(
            [4, 5, 6, 7])). \
            filter(orm_conv_generators_version)

        # read data from db
        generators_mv = pd.read_sql_query(generators_sqla.statement,
                                          session.bind,
                                          index_col='id')

        return generators_mv

    def _import_res_generators(session):
        """
        Import renewable (res) generators from oedb.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            List of medium-voltage generators
        :pandas:`pandas.DataFrame<dataframe>`
            List of low-voltage generators

        Notes
        -----
        You can find a full list of columns in
        :func:`edisgo.io.import_data._update_grids`

        If subtype is not specified it's set to 'unknown'.

        """

        # Create filter for generation technologies
        # ToDo: This needs to be removed when all generators can be imported
        types_filter = orm_re_generators.columns.generation_type.in_(
            ['solar', 'wind'])

        # build basic query
        generators_sqla = session.query(
            orm_re_generators.columns.id,
            orm_re_generators.columns.subst_id,
            orm_re_generators.columns.la_id,
            orm_re_generators.columns.mvlv_subst_id,
            orm_re_generators.columns.electrical_capacity,
            orm_re_generators.columns.generation_type,
            orm_re_generators.columns.generation_subtype,
            orm_re_generators.columns.voltage_level,
            orm_re_generators.columns.w_id,
            func.ST_AsText(func.ST_Transform(
                orm_re_generators.columns.rea_geom_new, srid)).label('geom'),
            func.ST_AsText(func.ST_Transform(
            orm_re_generators.columns.geom, srid)).label('geom_em')). \
                filter(orm_re_generators.columns.subst_id ==
                       edisgo_object.topology.mv_grid.id). \
            filter(orm_re_generators_version). \
            filter(types_filter)

        # extend basic query for MV generators and read data from db
        generators_mv_sqla = generators_sqla. \
            filter(orm_re_generators.columns.voltage_level.in_([4, 5]))
        generators_mv = pd.read_sql_query(generators_mv_sqla.statement,
                                          session.bind,
                                          index_col='id')

        # define generators with unknown subtype as 'unknown'
        generators_mv.loc[generators_mv[
                              'generation_subtype'].isnull(),
                          'generation_subtype'] = 'unknown'

        # convert capacity from kW to MW
        generators_mv.electrical_capacity = \
            pd.to_numeric(generators_mv.electrical_capacity) / 1e3

        # extend basic query for LV generators and read data from db
        generators_lv_sqla = generators_sqla. \
            filter(orm_re_generators.columns.voltage_level.in_([6, 7]))
        generators_lv = pd.read_sql_query(generators_lv_sqla.statement,
                                       session.bind,
                                       index_col='id')

        # define generators with unknown subtype as 'unknown'
        generators_lv.loc[generators_lv[
                              'generation_subtype'].isnull(),
                          'generation_subtype'] = 'unknown'

        # convert capacity from kW to MW
        generators_lv.electrical_capacity = \
            pd.to_numeric(generators_lv.electrical_capacity) / 1e3

        return generators_mv, generators_lv

    def _update_grids(edisgo_object, imported_generators_mv,
                      imported_generators_lv,
                      remove_missing=True):
        """
        Update network according to new generator dataset.

        It
            * adds new generators to network if they do not exist
            * updates existing generators if parameters have changed
            * removes existing generators from network which do not exist in
              the imported dataset

        Steps:

            * Step 1: MV generators: Update existing, create new,
              remove decommissioned
            * Step 2: LV generators (single units): Update existing, remove
              decommissioned
            * Step 3: LV generators (in aggregated MV generators):
              Update existing, remove decommissioned
              (aggregated MV generators = originally LV generators from
              aggregated Load Areas which were aggregated during import from
              ding0.)
            * Step 4: LV generators (single units + aggregated MV generators):
              Create new

        Parameters
        ----------
        edisgo_object: :class:`~.network.topology.Topology`
            The eDisGo container object

        generators_mv: :pandas:`pandas.DataFrame<dataframe>`
            List of MV generators
            Columns:
                * id: :obj:`int` (index column)
                * electrical_capacity: :obj:`float` (unit: kW)
                * generation_type: :obj:`str` (e.g. 'solar')
                * generation_subtype: :obj:`str` (e.g. 'solar_roof_mounted')
                * voltage level: :obj:`int` (range: 4..7,)
                * geom: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)
                * geom_em: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)

        generators_lv: :pandas:`pandas.DataFrame<dataframe>`
            List of LV generators
            Columns:
                * id: :obj:`int` (index column)
                * mvlv_subst_id: :obj:`int` (id of MV-LV substation in grid
                  = grid which the generator will be connected to)
                * electrical_capacity: :obj:`float` (unit: kW)
                * generation_type: :obj:`str` (e.g. 'solar')
                * generation_subtype: :obj:`str` (e.g. 'solar_roof_mounted')
                * voltage level: :obj:`int` (range: 4..7,)
                * geom: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)
                * geom_em: :shapely:`Shapely Point object<points>`
                  (CRS see config_grid.cfg)

        remove_missing: :obj:`bool`
            If true, remove generators from network which are not included in
            the imported dataset.

        """

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -4

        # get all imported generators
        imported_gens = pd.concat(
            [imported_generators_lv, imported_generators_mv])

        logger.debug('{} generators imported.'.format(
            len(imported_gens)))

        # get existing generators in MV and LV grids and append ID column
        existing_gens = edisgo_object.topology.generators_df
        existing_gens['id'] = list(
            map(lambda _: int(_.split('_')[-1]), existing_gens.index))

        logger.debug('Cumulative generator capacity (existing): {} MW'
                     .format(round(existing_gens.p_nom.sum(), 1)))

        # check if capacity of any of the imported generators is <= 0
        # (this may happen if dp is buggy) and remove generator if it is
        gens_to_remove = imported_gens[imported_gens.electrical_capacity <= 0]
        for id in gens_to_remove.index:
            # remove from topology (if generator exists)
            if id in existing_gens.id.values:
                gen_name = existing_gens[existing_gens.id == id].index[0]
                edisgo_object.topology.remove_generator(gen_name)
                logger.warning(
                    'Capacity of generator {} is <= 0, it is therefore '
                    'removed. Check your data source.'.format(gen_name))
            # remove from imported generators
            imported_gens.drop(id, inplace=True)
            if id in imported_generators_mv.index:
                imported_generators_mv.drop(id, inplace=True)
            else:
                imported_generators_lv.drop(id, inplace=True)

        # =============================================
        # Step 1: Update existing MV and LV generators
        # =============================================

        # filter for generators that need to be updated (i.e. that
        # appear in the imported and existing generators dataframes)
        gens_to_update = existing_gens[existing_gens.id.isin(
            imported_gens.index.values)]

        # calculate capacity difference between existing and imported
        # generators
        gens_to_update['cap_diff'] = \
            imported_gens.loc[
                gens_to_update.id, 'electrical_capacity'].values - \
            gens_to_update.p_nom
        # in case there are generators whose capacity does not match, update
        # their capacity
        gens_to_update_cap = gens_to_update[
            abs(gens_to_update.cap_diff) > cap_diff_threshold]

        for id, row in gens_to_update_cap.iterrows():
            edisgo_object.topology._generators_df.loc[id, 'p_nom'] = \
                imported_gens.loc[row['id'], 'electrical_capacity']

        log_geno_count = len(gens_to_update_cap)
        log_geno_cap = gens_to_update_cap['cap_diff'].sum()
        logger.debug(
            'Capacities of {} of {} existing generators updated ({} MW).'
                .format(log_geno_count, len(gens_to_update),
                        round(log_geno_cap, 1)))

        # ==================================================
        # Step 2: Remove decommissioned MV and LV generators
        # ==================================================

        # filter for MV generators that do not appear in the imported but in
        # the existing generators dataframe
        decommissioned_gens = existing_gens[~existing_gens.id.isin(
            imported_gens.index.values)]

        if not decommissioned_gens.empty and remove_missing:
            for gen in decommissioned_gens.index:
                edisgo_object.topology.remove_generator(gen)
            log_geno_cap = decommissioned_gens.p_nom.sum()
            log_geno_count = len(decommissioned_gens)
            logger.debug('{} decommissioned generators removed ({} MW).'
                         .format(log_geno_count,
                                 round(log_geno_cap, 1)))

        # ===============================
        # Step 3: Add new MV generators
        # ===============================

        new_gens_mv = imported_generators_mv[
            ~imported_generators_mv.index.isin(list(existing_gens.id))]
        number_new_gens = len(new_gens_mv)

        # iterate over new generators and create them
        for id in new_gens_mv.index:
            # check if geom is available, skip otherwise
            geom = _check_mv_generator_geom(new_gens_mv.loc[id, :])
            if geom is None:
                logger.warning('Generator {} has no geom entry and will'
                               'not be imported!'.format(id))
                new_gens_mv.drop(id)
                continue
            new_gens_mv.at[id, 'geom'] = geom
            add_and_connect_mv_generator(edisgo_object, new_gens_mv.loc[id, :])

        log_geno_count = len(new_gens_mv)
        log_geno_cap = new_gens_mv['electrical_capacity'].sum()
        logger.debug('{} of {} new MV generators added ({} MW).'
                     .format(log_geno_count,
                             number_new_gens,
                             round(log_geno_cap, 1)))

        # ===============================
        # Step 4: Add new LV generators
        # ===============================

        new_gens_lv = imported_generators_lv[
            ~imported_generators_lv.index.isin(list(existing_gens.id))]

        # check if new generators can be allocated to an existing LV grid
        grid_ids = [_.id for _ in edisgo_object.topology._grids.values()]
        if not any([_ in grid_ids
                    for _ in list(imported_generators_lv['mvlv_subst_id'])]):
            logger.warning(
                'None of the imported LV generators can be allocated '
                'to an existing LV grid. Check compatibility of grid '
                'and generator datasets.')

        # iterate over new generators and create them
        for id in new_gens_lv.index:
            add_and_connect_lv_generator(
                edisgo_object, new_gens_lv.loc[id, :])

        log_geno_count = len(new_gens_lv)
        log_geno_cap = new_gens_lv['electrical_capacity'].sum()
        logger.debug('{} new LV generators added ({} MW).'
                     .format(log_geno_count,
                             round(log_geno_cap, 1)))

        for lv_grid in edisgo_object.topology.mv_grid.lv_grids:
            lv_loads = len(lv_grid.loads_df)
            lv_gens_voltage_level_7 = len(lv_grid.generators_df[
                lv_grid.generators_df.bus != lv_grid.station.index[0]])
            # warn if there're more generators than loads in LV grid
            if lv_gens_voltage_level_7 > lv_loads * 2:
                logger.debug(
                    'There are {} generators (voltage level 7) but only {} '
                    'loads in LV grid {}.'.format(
                        lv_gens_voltage_level_7,
                        lv_loads,
                        lv_grid.id))

    def _check_mv_generator_geom(generator_data):
        """
        Checks if a valid geom is available in dataset.

        If yes, this geom will be used.
        If not, geom from EnergyMap is used if available.

        Parameters
        ----------
        generator_data : series
            Series with geom (geometry from open_eGo dataprocessing) and
            geom_em (geometry from EnergyMap)

        Returns
        -------
        :shapely:`Shapely Point object<points>` or None
            Geom of generator. None, if no geom is available.

        """
        # check if geom is available
        if generator_data.geom:
            return generator_data.geom
        else:
            # set geom to EnergyMap's geom, if available
            if generator_data.geom_em:
                logger.debug(
                    'Generator {} has no geom entry, EnergyMap\'s geom '
                    'entry will be used.'.format(generator_data.name))
                return generator_data.geom_em
        return None

    def _validate_generation():
        """
        Validate generation capacity in updated grids.

        The validation uses the cumulative capacity of all generators.

        """
        # ToDo: Validate conv. genos too!

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -1

        capacity_imported = generators_res_mv['electrical_capacity'].sum() + \
                            generators_res_lv['electrical_capacity'].sum() #+ \
                            #generators_conv_mv['capacity'].sum()

        capacity_grid = edisgo_object.topology.generators_df.p_nom.sum()

        logger.debug('Cumulative generator capacity (updated): {} MW'
                     .format(round(capacity_imported, 1)))

        if abs(capacity_imported - capacity_grid) > cap_diff_threshold:
            raise ValueError(
                'Cumulative capacity of imported generators ({} MW) '
                'differ from cumulative capacity of generators '
                'in updated grid ({} MW) by {} MW.'
                    .format(round(capacity_imported, 1),
                            round(capacity_grid, 1),
                            round(capacity_imported - capacity_grid, 1)))
        else:
            logger.debug(
                'Cumulative capacity of imported generators validated.')

    def _validate_sample_geno_location():
        """
        Checks that newly imported generators are located inside grid district.

        The check is performed for two randomly sampled generators.

        """
        if all(generators_res_lv['geom'].notnull()) \
                and all(generators_res_mv['geom'].notnull()) \
                and not generators_res_lv['geom'].empty \
                and not generators_res_mv['geom'].empty:

            # get geom of 1 random MV and 1 random LV generator and transform
            sample_mv_geno_geom_shp = transform(
                proj2equidistant(srid),
                wkt_loads(
                    generators_res_mv['geom'].dropna().sample(n=1).values[0]))
            sample_lv_geno_geom_shp = transform(
                proj2equidistant(srid),
                wkt_loads(
                    generators_res_lv['geom'].dropna().sample(n=1).values[0]))

            # get geom of MV grid district
            mvgd_geom_shp = transform(
                proj2equidistant(srid),
                edisgo_object.topology.grid_district['geom'])

            # check if MVGD contains geno
            if not (mvgd_geom_shp.contains(sample_mv_geno_geom_shp) and
                    mvgd_geom_shp.contains(sample_lv_geno_geom_shp)):
                raise ValueError(
                    'At least one imported generator is not located in the MV '
                    'grid area. Check compatibility of grid and generator '
                    'datasets.')

    logging.warning('Right now only solar and wind generators can be '
                    'imported from the oedb.')

    oedb_data_source = edisgo_object.config['data_source']['oedb_data_source']
    scenario = edisgo_object.topology.generator_scenario
    srid = edisgo_object.topology.grid_district['srid']

    # load ORM names
    orm_conv_generators_name = \
        edisgo_object.config[oedb_data_source]['conv_generators_prefix'] + \
        scenario + \
        edisgo_object.config[oedb_data_source]['conv_generators_suffix']
    orm_re_generators_name = \
        edisgo_object.config[oedb_data_source]['re_generators_prefix'] + \
        scenario + \
        edisgo_object.config[oedb_data_source]['re_generators_suffix']

    if oedb_data_source == 'model_draft':

        # import ORMs
        orm_conv_generators = model_draft.__getattribute__(
            orm_conv_generators_name)
        orm_re_generators = model_draft.__getattribute__(
            orm_re_generators_name)

        # set dummy version condition (select all generators)
        orm_conv_generators_version = 1 == 1
        orm_re_generators_version = 1 == 1

    elif oedb_data_source == 'versioned':

        data_version = edisgo_object.config['versioned']['version']

        # import ORMs
        orm_conv_generators = supply.__getattribute__(
            orm_conv_generators_name)
        orm_re_generators = supply.__getattribute__(
            orm_re_generators_name)

        # set version condition
        orm_conv_generators_version = orm_conv_generators.columns.version == \
                                      data_version
        orm_re_generators_version = orm_re_generators.columns.version == \
                                    data_version

    # get conventional and renewable generators
    with session_scope() as session:
        # generators_conv_mv = _import_conv_generators(session)
        generators_res_mv, generators_res_lv = _import_res_generators(
            session)

    # generators_mv = generators_conv_mv.append(generators_res_mv)

    # validate that imported generators are located inside the grid district
    _validate_sample_geno_location()

    _update_grids(edisgo_object=edisgo_object,
                  #generators_mv=generators_mv,
                  imported_generators_mv=generators_res_mv,
                  imported_generators_lv=generators_res_lv)

    _validate_generation()
