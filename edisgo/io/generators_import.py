import pandas as pd
from sqlalchemy import func
import logging
import os
from math import isnan
import random

from egoio.db_tables import model_draft, supply
from edisgo.tools import session_scope

from ..network.components import Generator
from ..network.connect import add_and_connect_mv_generator, connect_lv_generators
from ..network.tools import select_cable
from ..tools.geo import proj2equidistant

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
            generators_mv.electrical_capacity / 1e3

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
            generators_lv.electrical_capacity / 1e3

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

        # get existing generators in MV and LV grids
        existing_gens = edisgo_object.topology.generators_df

        # print current capacity
        logger.debug('Cumulative generator capacity (existing): {} MW'
                     .format(round(existing_gens.p_nom.sum(), 1)))

        # ======================================
        # Step 1: MV generators (existing)
        # ======================================

        logger.debug('==> MV generators')
        logger.debug('{} generators imported.'.format(
            len(imported_generators_mv)))

        # get existing MV generators and append column with ID as in oedb
        existing_gens_mv = edisgo_object.topology.mv_grid.generators_df
        existing_gens_mv['id'] = list(
            map(lambda _: int(_.split('_')[-1]), existing_gens_mv.index))
        # filter for MV generators that only need to be updated (i.e. that
        # appear in the imported and existing generators dataframes)
        gens_to_update = existing_gens_mv[existing_gens_mv.id.isin(
            imported_generators_mv.index.values)]

        # check if new capacity of any of the imported generators is <= 0
        # (this may happen if dp is buggy)
        gens_to_remove = imported_generators_mv.loc[gens_to_update.id, :][
            imported_generators_mv.loc[
            gens_to_update.id, :].electrical_capacity <= 0]
        for id in gens_to_remove.index:
            gen_name = gens_to_update[gens_to_update.id == id].index[0]
            edisgo_object.topology.remove_generator(gen_name)
            logger.warning(
                'Capacity of generator {} is <= 0, it is therefore removed. '
                'Check your data source.'.format(gen_name))

        # calculate capacity difference between existing and imported
        # generators
        gens_to_update['cap_diff'] = \
            imported_generators_mv.loc[
                gens_to_update.id, 'electrical_capacity'].values - \
            gens_to_update.p_nom
        # in case there are generators whose capacity does not match, update
        # their capacity
        gens_to_update_cap = gens_to_update[
            abs(gens_to_update.cap_diff) > cap_diff_threshold]

        for id, row in gens_to_update_cap.iterrows():
                edisgo_object.topology._generators_df.loc[id, 'p_nom'] = \
                    imported_generators_mv.loc[
                        row['id'], 'electrical_capacity']

        log_geno_count = len(gens_to_update_cap)
        log_geno_cap = gens_to_update_cap['cap_diff'].sum()
        logger.debug(
            'Capacities of {} of {} existing generators updated ({} MW).'
                .format(log_geno_count, len(gens_to_update),
                        round(log_geno_cap, 1)))

        # ======================================
        # Step 2: MV generators (new)
        # ======================================

        new_gens_mv = imported_generators_mv[
            ~imported_generators_mv.index.isin(list(existing_gens_mv.id))]
        number_new_gens = len(new_gens_mv)

        # iterate over new generators and create them
        for id in new_gens_mv.index:
            # check if geom is available, skip otherwise
            geom = _check_geom(new_gens_mv.loc[id, :])
            if geom is None:
                logger.warning('Generator {} has no geom entry and will'
                               'not be imported!'.format(id))
                new_gens_mv.drop(id)
                continue
            new_gens_mv.loc[id, 'geom'] = geom
            add_and_connect_mv_generator(edisgo_object, new_gens_mv.loc[id, :])

        log_geno_count = len(new_gens_mv)
        log_geno_cap = new_gens_mv['electrical_capacity'].sum()
        logger.debug('{} of {} new generators added ({} MW).'
                     .format(log_geno_count,
                             number_new_gens,
                             round(log_geno_cap, 1)))

        # ======================================
        # Step 3: MV generators (decommissioned)
        # ======================================

        # remove decommissioned generators
        # (genos which exist in grid but not in the new dataset)

        # filter for MV generators that do not appear in the imported but in
        # the existing generators dataframe
        decommissioned_gens_mv = existing_gens_mv[~existing_gens_mv.id.isin(
            imported_generators_mv.index.values)]

        if not decommissioned_gens_mv.empty and remove_missing:
            for gen in decommissioned_gens_mv.index:
                edisgo_object.topology.remove_generator(gen)
            log_geno_cap = decommissioned_gens_mv.p_nom.sum()
            log_geno_count = len(decommissioned_gens_mv)
            logger.debug('{} decommissioned generators removed ({} MW).'
                         .format(log_geno_count,
                                 round(log_geno_cap, 1)))

        # # =============================================
        # # Step 2: LV generators (single existing units)
        # # =============================================
        # logger.debug('==> LV generators')
        # logger.debug('{} generators imported.'.format(str(len(generators_lv))))
        # # get existing genos (status quo DF format)
        # g_lv_existing = g_lv[g_lv['id'].isin(list(generators_lv.index.values))]
        # # get existing genos (new genos DF format)
        # generators_lv_existing = generators_lv[generators_lv.index.isin(list(g_lv_existing['id']))]
        #
        # # TEMP: BACKUP 1 GENO FOR TESTING
        # # temp_geno = g_lv.iloc[0]
        #
        # # remove existing ones from grid's geno list
        # g_lv = g_lv[~g_lv.isin(g_lv_existing)].dropna()
        #
        # # iterate over exiting generators and check whether capacity has changed
        # log_geno_count = 0
        # log_geno_cap = 0
        # for id, row in generators_lv_existing.iterrows():
        #
        #     geno_existing = g_lv_existing[g_lv_existing['id'] == id]['obj'].iloc[0]
        #
        #     # check if capacity equals; if not: update capacity
        #     if abs(row['electrical_capacity'] - \
        #                    geno_existing.nominal_capacity) < cap_diff_threshold:
        #         continue
        #     else:
        #         log_geno_cap += row['electrical_capacity'] - geno_existing.nominal_capacity
        #         log_geno_count += 1
        #         geno_existing.nominal_capacity = row['electrical_capacity']
        #
        # logger.debug('Capacities of {} of {} existing generators (single units) updated ({} kW).'
        #              .format(str(log_geno_count),
        #                      str(len(generators_lv_existing) - log_geno_count),
        #                      str(round(log_geno_cap, 1))
        #                      )
        #              )
        #
        # # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        # # g_lv.loc[len(g_lv)] = temp_geno
        #
        # # remove decommissioned genos
        # # (genos which exist in grid but not in the new dataset)
        # log_geno_cap = 0
        # if not g_lv.empty and remove_missing:
        #     log_geno_count = 0
        #     for _, row in g_lv.iterrows():
        #         log_geno_cap += row['obj'].nominal_capacity
        #         row['obj'].grid.graph.remove_node(row['obj'])
        #         log_geno_count += 1
        #     logger.debug('{} of {} decommissioned generators (single units) removed ({} kW).'
        #                  .format(str(log_geno_count),
        #                          str(len(g_lv)),
        #                          str(round(log_geno_cap, 1))
        #                          )
        #                  )
        #
        # # ====================================================================================
        # # Step 3: LV generators (existing in aggregated units (originally from aggregated LA))
        # # ====================================================================================
        # g_lv_agg = network.dingo_import_data
        # g_lv_agg_existing = g_lv_agg[g_lv_agg['id'].isin(list(generators_lv.index.values))]
        # generators_lv_agg_existing = generators_lv[generators_lv.index.isin(list(g_lv_agg_existing['id']))]
        #
        # # TEMP: BACKUP 1 GENO FOR TESTING
        # # temp_geno = g_lv_agg.iloc[0]
        #
        # g_lv_agg = g_lv_agg[~g_lv_agg.isin(g_lv_agg_existing)].dropna()
        #
        # log_geno_count = 0
        # log_agg_geno_list = []
        # log_geno_cap = 0
        # for id, row in generators_lv_agg_existing.iterrows():
        #
        #     # check if capacity equals; if not: update capacity off agg. geno
        #     cap_diff = row['electrical_capacity'] - \
        #                g_lv_agg_existing[g_lv_agg_existing['id'] == id]['capacity'].iloc[0]
        #     if abs(cap_diff) < cap_diff_threshold:
        #         continue
        #     else:
        #         agg_geno = g_lv_agg_existing[g_lv_agg_existing['id'] == id]['agg_geno'].iloc[0]
        #         agg_geno.nominal_capacity += cap_diff
        #         log_geno_cap += cap_diff
        #
        #         log_geno_count += 1
        #         log_agg_geno_list.append(agg_geno)
        #
        # logger.debug('Capacities of {} of {} existing generators (in {} of {} aggregated units) '
        #              'updated ({} kW).'
        #              .format(str(log_geno_count),
        #                      str(len(generators_lv_agg_existing) - log_geno_count),
        #                      str(len(set(log_agg_geno_list))),
        #                      str(len(g_lv_agg_existing['agg_geno'].unique())),
        #                      str(round(log_geno_cap, 1))
        #                      )
        #              )
        #
        # # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        # # g_lv_agg.loc[len(g_lv_agg)] = temp_geno
        #
        # # remove decommissioned genos
        # # (genos which exist in grid but not in the new dataset)
        # log_geno_cap = 0
        # if not g_lv_agg.empty and remove_missing:
        #     log_geno_count = 0
        #     for _, row in g_lv_agg.iterrows():
        #         row['agg_geno'].nominal_capacity -= row['capacity']
        #         log_geno_cap += row['capacity']
        #         # remove LV geno id from id string of agg. geno
        #         id = row['agg_geno'].id.split('-')
        #         ids = id[2].split('_')
        #         ids.remove(str(int(row['id'])))
        #         row['agg_geno'].id = '-'.join([id[0], id[1], '_'.join(ids)])
        #
        #         # after removing the LV geno from agg geno, is the agg. geno empty?
        #         # if yes, remove it from grid
        #         if not ids:
        #             row['agg_geno'].grid.graph.remove_node(row['agg_geno'])
        #
        #         log_geno_count += 1
        #     logger.debug('{} of {} decommissioned generators in aggregated generators removed ({} kW).'
        #                  .format(str(log_geno_count),
        #                          str(len(g_lv_agg)),
        #                          str(round(log_geno_cap, 1))
        #                          )
        #                  )
        #
        # # ====================================================================
        # # Step 4: LV generators (new single units + genos in aggregated units)
        # # ====================================================================
        # # new genos
        # log_geno_count =\
        #     log_agg_geno_new_count =\
        #     log_agg_geno_upd_count = 0
        #
        # # TEMP: BACKUP 1 GENO FOR TESTING
        # #temp_geno = generators_lv[generators_lv.index == g_lv_existing.iloc[0]['id']]
        #
        # generators_lv_new = generators_lv[~generators_lv.index.isin(list(g_lv_existing['id'])) &
        #                                   ~generators_lv.index.isin(list(g_lv_agg_existing['id']))]
        #
        # # TEMP: INSERT BACKUPPED GENO IN DF FOR TESTING
        # #generators_lv_new = generators_lv_new.append(temp_geno)
        #
        # # dict for new agg. generators
        # agg_geno_new = {}
        # # get LV grid districts
        # lv_grid_dict = _build_lv_grid_dict(network)
        #
        # # get predefined random seed and initialize random generator
        # seed = int(network.config['grid_connection']['random_seed'])
        # random.seed(a=seed)
        #
        # # check if none of new generators can be allocated to an existing  LV grid
        # if not any([_ in lv_grid_dict.keys()
        #             for _ in list(generators_lv_new['mvlv_subst_id'])]):
        #     logger.warning('None of the imported generators can be allocated '
        #                    'to an existing LV grid. Check compatibility of grid '
        #                    'and generator datasets.')
        #
        # # iterate over new (single unit or part of agg. unit) generators and create them
        # log_geno_cap = 0
        # for id, row in generators_lv_new.iterrows():
        #     lv_geno_added_to_agg_geno = False
        #
        #     # new unit is part of agg. LA (mvlv_subst_id is different from existing
        #     # ones in LV grids of non-agg. load areas)
        #     if (row['mvlv_subst_id'] not in lv_grid_dict.keys() and
        #             row['la_id'] and not isnan(row['la_id']) and
        #             row['mvlv_subst_id'] and not isnan(row['mvlv_subst_id'])):
        #
        #         # check if new unit can be added to existing agg. generator
        #         # (LA id, type and subtype match) -> update existing agg. generator.
        #         # Normally, this case should not occur since `subtype` of new genos
        #         # is set to a new value (e.g. 'solar')
        #         for _, agg_row in g_mv_agg.iterrows():
        #             if (agg_row['la_id'] == int(row['la_id']) and
        #                     agg_row['obj'].type == row['generation_type'] and
        #                     agg_row['obj'].subtype == row['generation_subtype']):
        #
        #                 agg_row['obj'].nominal_capacity += row['electrical_capacity']
        #                 agg_row['obj'].id += '_{}'.format(str(id))
        #                 log_agg_geno_upd_count += 1
        #                 lv_geno_added_to_agg_geno = True
        #
        #         if not lv_geno_added_to_agg_geno:
        #             la_id = int(row['la_id'])
        #             if la_id not in agg_geno_new:
        #                 agg_geno_new[la_id] = {}
        #             if row['voltage_level'] not in agg_geno_new[la_id]:
        #                 agg_geno_new[la_id][row['voltage_level']] = {}
        #             if row['generation_type'] not in agg_geno_new[la_id][row['voltage_level']]:
        #                 agg_geno_new[la_id][row['voltage_level']][row['generation_type']] = {}
        #             if row['generation_subtype'] not in \
        #                     agg_geno_new[la_id][row['voltage_level']][row['generation_type']]:
        #                 agg_geno_new[la_id][row['voltage_level']][row['generation_type']]\
        #                     .update({row['generation_subtype']: {'ids': [int(id)],
        #                                                          'capacity': row['electrical_capacity']
        #                                                          }
        #                      }
        #                 )
        #             else:
        #                 agg_geno_new[la_id][row['voltage_level']][row['generation_type']] \
        #                     [row['generation_subtype']]['ids'].append(int(id))
        #                 agg_geno_new[la_id][row['voltage_level']][row['generation_type']] \
        #                     [row['generation_subtype']]['capacity'] += row['electrical_capacity']
        #
        #     # new generator is a single (non-aggregated) unit
        #     else:
        #         # check if geom is available
        #         geom = _check_geom(id, row)
        #
        #         if row['generation_type'] in ['solar', 'wind']:
        #             gen = GeneratorFluctuating(
        #                 id=id,
        #                 grid=None,
        #                 nominal_capacity=row['electrical_capacity'],
        #                 type=row['generation_type'],
        #                 subtype=row['generation_subtype'],
        #                 v_level=int(row['voltage_level']),
        #                 weather_cell_id=row['w_id'],
        #                 geom=wkt_loads(geom) if geom else geom)
        #         else:
        #             gen = Generator(id=id,
        #                             grid=None,
        #                             nominal_capacity=row[
        #                                 'electrical_capacity'],
        #                             type=row['generation_type'],
        #                             subtype=row['generation_subtype'],
        #                             v_level=int(row['voltage_level']),
        #                             geom=wkt_loads(geom) if geom else geom)
        #
        #         # TEMP: REMOVE MVLV SUBST ID FOR TESTING
        #         #row['mvlv_subst_id'] = None
        #
        #         # check if MV-LV substation id exists. if not, allocate to
        #         # random one
        #         lv_grid = _check_mvlv_subst_id(
        #             generator=gen,
        #             mvlv_subst_id=row['mvlv_subst_id'],
        #             lv_grid_dict=lv_grid_dict)
        #
        #         gen.grid = lv_grid
        #
        #         lv_grid.graph.add_node(gen, type='generator')
        #
        #         log_geno_count += 1
        #     log_geno_cap += row['electrical_capacity']
        #
        # # there are new agg. generators to be created
        # if agg_geno_new:
        #
        #     pfac_mv_gen = network.config['reactive_power_factor']['mv_gen']
        #
        #     # add aggregated generators
        #     for la_id, val in agg_geno_new.items():
        #         for v_level, val2 in val.items():
        #             for type, val3 in val2.items():
        #                 for subtype, val4 in val3.items():
        #                     if type in ['solar', 'wind']:
        #                         gen = GeneratorFluctuating(
        #                             id='agg-' + str(la_id) + '-' + '_'.join([
        #                                 str(_) for _ in val4['ids']]),
        #                             grid=network.mv_grid,
        #                             nominal_capacity=val4['capacity'],
        #                             type=type,
        #                             subtype=subtype,
        #                             v_level=4,
        #                             # ToDo: get correct w_id
        #                             weather_cell_id=row['w_id'],
        #                             geom=network.mv_grid.station.geom)
        #                     else:
        #                         gen = Generator(
        #                             id='agg-' + str(la_id) + '-' + '_'.join([
        #                                 str(_) for _ in val4['ids']]),
        #                             nominal_capacity=val4['capacity'],
        #                             type=type,
        #                             subtype=subtype,
        #                             geom=network.mv_grid.station.geom,
        #                             grid=network.mv_grid,
        #                             v_level=4)
        #
        #                     network.mv_grid.graph.add_node(
        #                         gen, type='generator_aggr')
        #
        #                     # select cable type
        #                     line_type, line_count = select_cable(
        #                         edisgo_obj=edisgo_obj,
        #                         level='mv',
        #                         apparent_power=gen.nominal_capacity /
        #                         pfac_mv_gen)
        #
        #                     # connect generator to MV station
        #                     line = Line(id='line_aggr_generator_la_' + str(la_id) + '_vlevel_{v_level}_'
        #                                 '{subtype}'.format(
        #                                  v_level=v_level,
        #                                  subtype=subtype),
        #                                 type=line_type,
        #                                 kind='cable',
        #                                 quantity=line_count,
        #                                 length=1e-3,
        #                                 grid=network.mv_grid)
        #
        #                     network.mv_grid.graph.add_edge(network.mv_grid.station,
        #                                                    gen,
        #                                                    line=line,
        #                                                    type='line_aggr')
        #
        #                     log_agg_geno_new_count += len(val4['ids'])
        #                     log_geno_cap += val4['capacity']
        #
        # logger.debug('{} of {} new generators added ({} single units, {} to existing '
        #              'agg. generators and {} units as new aggregated generators) '
        #              '(total: {} kW).'
        #              .format(str(log_geno_count +
        #                          log_agg_geno_new_count +
        #                          log_agg_geno_upd_count),
        #                      str(len(generators_lv_new)),
        #                      str(log_geno_count),
        #                      str(log_agg_geno_upd_count),
        #                      str(log_agg_geno_new_count),
        #                      str(round(log_geno_cap, 1))
        #                      )
        #              )

    def _check_geom(generator_data):
        """
        Checks if a valid geom is available in dataset.

        If yes, this geom will be used.
        If not:

            * MV generators: use geom from EnergyMap.
            * LV generators: set geom to None. It is re-set in
                :func:`edisgo.io.import_data._check_mvlv_subst_id`
                to MV-LV station's geom. EnergyMap's geom is not used
                since it is more inaccurate than the station's geom.

        Parameters
        ----------
        generator_data : series
            Series with geom (geometry from open_eGo dataprocessing, shapely
            Point) and
            geom_em (geometry from EnergyMap, shapely Point) and voltage_level
            (integer)

        Returns
        -------
        :shapely:`Shapely Point object<points>` or None
            Geom of generator. None, if no geom is available.

        """
        # check if geom is available
        if generator_data.geom:
            return generator_data.geom
        else:
            # MV generators: set geom to EnergyMap's geom, if available
            if generator_data.voltage_level in [4, 5]:
                # check if original geom from Energy Map is available
                if generator_data.geom_em:
                    logger.debug(
                        'Generator {} has no geom entry, EnergyMap\'s geom '
                        'entry will be used.'.format(generator_data.name))
                    return generator_data.geom_em
        return None

    def _check_mvlv_subst_id(generator, mvlv_subst_id, lv_grid_dict):
        """
        Checks if MV/LV substation id of single LV generator is valid.

        In case it is not valid or missing, a random one from existing stations
        in LV grids will be assigned.

        Parameters
        ----------
        generator : :class:`~.network.components.Generator`
            LV generator
        mvlv_subst_id : :obj:`int`
            MV-LV substation id
        lv_grid_dict : :obj:`dict`
            Dict of existing LV grids
            Format: {:obj:`int`: :class:`~.network.grids.LVGrid`}

        Returns
        -------
        :class:`~.network.grids.LVGrid`
            LV network of generator

        """

        if mvlv_subst_id and not isnan(mvlv_subst_id):
            # assume that given LA exists
            try:
                # get LV grid
                lv_grid = lv_grid_dict[mvlv_subst_id]

                # if no geom, use geom of station
                if not generator.geom:
                    generator.geom = lv_grid.station.geom
                    logger.debug(
                        "Generator {} has no geom entry, stations' geom will "
                        "be used.".format(generator.id))
                return lv_grid

            # if LA/LVGD does not exist, choose random LVGD and move generator
            # to station of LVGD
            # this occurs due to exclusion of LA with peak load < 1kW
            except:
                lv_grid = random.choice(list(lv_grid_dict.values()))
                generator.geom = lv_grid.station.geom

                logger.warning('Generator {} cannot be assigned to '
                               'non-existent LV Grid and was '
                               'allocated to a random LV Grid ({}); '
                               'geom was set to stations\' geom.'
                               .format(repr(generator),
                                       repr(lv_grid)))
                return lv_grid

        else:
            lv_grid = random.choice(list(lv_grid_dict.values()))
            generator.geom = lv_grid.station.geom

            logger.warning('Generator {} has no mvlv_subst_id and was '
                           'allocated to a random LV Grid ({}); '
                           'geom was set to stations\' geom.'
                           .format(repr(generator),
                                   repr(lv_grid)))
            return lv_grid

    def _validate_generation():
        """
        Validate generators in updated grids.

        The validation uses the cumulative capacity of all generators.

        """
        # ToDo: Valdate conv. genos too!

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -1

        capacity_imported = generators_res_mv['electrical_capacity'].sum()# + \
                            #generators_res_lv['electrical_capacity'].sum() #+ \
                            #generators_conv_mv['capacity'].sum()

        # ToDo: change to all generators once lv import works
        capacity_grid = \
            edisgo_object.topology.mv_grid.generators_df.p_nom.sum()

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


# probably not needed anymore
# def _build_generator_list(network):
#     """Builds DataFrames with all generators in MV and LV grids
#
#     Returns
#     -------
#     :pandas:`pandas.DataFrame<dataframe>`
#             A DataFrame with id of and reference to MV generators
#     :pandas:`pandas.DataFrame<dataframe>`
#             A DataFrame with id of and reference to LV generators
#     :pandas:`pandas.DataFrame<dataframe>`
#             A DataFrame with id of and reference to aggregated LV generators
#     """
#
#     genos_mv = pd.DataFrame(columns=
#                             ('id', 'obj'))
#     genos_lv = pd.DataFrame(columns=
#                             ('id', 'obj'))
#     genos_lv_agg = pd.DataFrame(columns=
#                                 ('la_id', 'id', 'obj'))
#
#     # MV genos
#     for geno in network.mv_grid.graph.nodes_by_attribute('generator'):
#             genos_mv.loc[len(genos_mv)] = [int(geno.id), geno]
#     for geno in network.mv_grid.graph.nodes_by_attribute('generator_aggr'):
#             la_id = int(geno.id.split('-')[1].split('_')[-1])
#             genos_lv_agg.loc[len(genos_lv_agg)] = [la_id, geno.id, geno]
#
#     # LV genos
#     for lv_grid in network.mv_grid.lv_grids:
#         for geno in lv_grid.generators:
#             genos_lv.loc[len(genos_lv)] = [int(geno.id), geno]
#
#     return genos_mv, genos_lv, genos_lv_agg


def _build_lv_grid_dict(network):
    """Creates dict of LV grids

    LV grid ids are used as keys, LV grid references as values.

    Parameters
    ----------
    network: :class:`~.network.topology.Topology`
        The eDisGo container object

    Returns
    -------
    :obj:`dict`
        Format: {:obj:`int`: :class:`~.network.grids.LVGrid`}
    """

    lv_grid_dict = {}
    for lv_grid in network.mv_grid.lv_grids:
        lv_grid_dict[lv_grid.id] = lv_grid
    return lv_grid_dict

