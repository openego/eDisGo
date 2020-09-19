import os
import pandas as pd
import numpy as np
from sqlalchemy import func
import random
import logging

from edisgo.network.grids import LVGrid
from edisgo.network.timeseries import add_generators_timeseries
from edisgo.tools import session_scope
from edisgo.tools.geo import (
    calc_geo_dist_vincenty,
    calc_geo_lines_in_buffer,
    proj2equidistant,
    proj2equidistant_reverse,
    find_nearest_bus,
)

logger = logging.getLogger("edisgo")

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


def oedb(edisgo_object, **kwargs):
    """Import generator data from the Open Energy Database (OEDB).

    The importer uses SQLAlchemy ORM objects.
    These are defined in ego.io,
    see https://github.com/openego/ego.io/tree/dev/egoio/db_tables

    Parameters
    ----------
    edisgo_object: :class:`~.EDisGo`
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
        generators_sqla = (
            session.query(
                orm_conv_generators.columns.id,
                orm_conv_generators.columns.subst_id,
                orm_conv_generators.columns.la_id,
                (orm_conv_generators.columns.capacity).label("electrical_capacity"),
                orm_conv_generators.columns.type,
                orm_conv_generators.columns.voltage_level,
                (orm_conv_generators.columns.fuel).label("generation_type"),
                func.ST_AsText(
                    func.ST_Transform(orm_conv_generators.columns.geom, srid)
                ).label("geom"),
            )
                .filter(
                orm_conv_generators.columns.subst_id
                == edisgo_object.topology.mv_grid.id
            )
                .filter(
                orm_conv_generators.columns.voltage_level.in_([4, 5])
            )
                .filter(orm_conv_generators_version)
        )

        # read data from db
        generators_mv = pd.read_sql_query(
            generators_sqla.statement, session.bind, index_col="id"
        )

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


        # build basic query
        generators_sqla = (
            session.query(
                orm_re_generators.columns.id,
                orm_re_generators.columns.subst_id,
                orm_re_generators.columns.la_id,
                orm_re_generators.columns.mvlv_subst_id,
                orm_re_generators.columns.electrical_capacity,
                orm_re_generators.columns.generation_type,
                orm_re_generators.columns.generation_subtype,
                orm_re_generators.columns.voltage_level,
                orm_re_generators.columns.w_id,
                func.ST_AsText(
                    func.ST_Transform(
                        orm_re_generators.columns.rea_geom_new, srid
                    )
                ).label("geom"),
                func.ST_AsText(
                    func.ST_Transform(orm_re_generators.columns.geom, srid)
                ).label("geom_em"),
            )
                .filter(
                orm_re_generators.columns.subst_id
                == edisgo_object.topology.mv_grid.id
            )
                .filter(orm_re_generators_version)
        )

        # extend basic query for MV generators and read data from db
        generators_mv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([4, 5])
        )
        generators_mv = pd.read_sql_query(
            generators_mv_sqla.statement, session.bind, index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        generators_mv.loc[
            generators_mv["generation_subtype"].isnull(), "generation_subtype"
        ] = "unknown"

        # convert capacity from kW to MW
        generators_mv.electrical_capacity = (
                pd.to_numeric(generators_mv.electrical_capacity) / 1e3
        )

        # extend basic query for LV generators and read data from db
        generators_lv_sqla = generators_sqla.filter(
            orm_re_generators.columns.voltage_level.in_([6, 7])
        )
        generators_lv = pd.read_sql_query(
            generators_lv_sqla.statement, session.bind, index_col="id"
        )

        # define generators with unknown subtype as 'unknown'
        generators_lv.loc[
            generators_lv["generation_subtype"].isnull(), "generation_subtype"
        ] = "unknown"

        # convert capacity from kW to MW
        generators_lv.electrical_capacity = (
                pd.to_numeric(generators_lv.electrical_capacity) / 1e3
        )

        return generators_mv, generators_lv

    def _validate_generation():
        """
        Validate generation capacity in updated grids.

        The validation uses the cumulative capacity of all generators.

        """

        # set capacity difference threshold
        cap_diff_threshold = 10 ** -1

        capacity_imported = (
                generators_res_mv["electrical_capacity"].sum()
                + generators_res_lv["electrical_capacity"].sum()
        ) + generators_conv_mv['electrical_capacity'].sum()

        capacity_grid = edisgo_object.topology.generators_df.p_nom.sum()

        logger.debug(
            "Cumulative generator capacity (updated): {} MW".format(
                round(capacity_imported, 1)
            )
        )

        if abs(capacity_imported - capacity_grid) > cap_diff_threshold:
            raise ValueError(
                "Cumulative capacity of imported generators ({} MW) "
                "differ from cumulative capacity of generators "
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

            # get geom of 1 random MV and 1 random LV generator and transform
            sample_mv_geno_geom_shp = transform(
                proj2equidistant(srid),
                wkt_loads(
                    generators_res_mv["geom"].dropna().sample(n=1).values[0]
                ),
            )
            sample_lv_geno_geom_shp = transform(
                proj2equidistant(srid),
                wkt_loads(
                    generators_res_lv["geom"].dropna().sample(n=1).values[0]
                ),
            )

            # get geom of MV grid district
            mvgd_geom_shp = transform(
                proj2equidistant(srid),
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
    scenario = edisgo_object.topology.generator_scenario
    srid = edisgo_object.topology.grid_district["srid"]

    # load ORM names
    orm_conv_generators_name = (
            edisgo_object.config[oedb_data_source][
                "conv_generators_prefix"]
            + scenario
            + edisgo_object.config[oedb_data_source][
                "conv_generators_suffix"]
    )
    orm_re_generators_name = (
        edisgo_object.config[oedb_data_source]["re_generators_prefix"]
        + scenario
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

    update_grids(
        edisgo_object=edisgo_object,
        imported_generators_mv=generators_mv,
        imported_generators_lv=generators_res_lv,
        remove_missing=kwargs.get("remove_missing", True),
    )

    _validate_generation()

    # update time series if they were already set
    if not edisgo_object.timeseries.generators_active_power.empty:
        add_generators_timeseries(
            edisgo_obj=edisgo_object,
            generator_names=edisgo_object.topology.generators_df.index)


def add_and_connect_mv_generator(edisgo_object, generator,
                                 comp_type="Generator"):
    """
    Add and connect new MV generator to existing grid.

    ToDo: Change name to add_and_connect_mv_component and move to some other
     module.

    This function connects

        * generators of voltage level 4
            * to HV-MV station

        * generators of voltage level 5
            * to nearest MV bus or line
            * in case generator is connected to a line, the line is split and
              a new branch tee is added to connect new generator to

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    generator : pd.Series
        Pandas series with generator information such as electrical_capacity
        in MW and generation_type. (name, electrical_capacity, generation_type,
        generation_subtype, w_id, geom, voltage_level). Geom is expected as
        string.

    Returns
    -------
    comp_name: `str`
        The name of the newly connected generator

    """

    # ToDo use select_cable instead of standard line?
    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data["mv_cables"].loc[
        edisgo_object.config["grid_expansion_standard_equipment"][
            "mv_line"
        ]
    ]

    geom = wkt_loads(generator.geom)

    # create new bus for new generator
    gen_bus = False
    if generator.voltage_level == 5:
        # in case of voltage_level 5 check if an existing bus can be used
        # to connect generator to

        # TODO: Make this a meaningful value
        DISTANCE_THRESHOLD = 0.01

        # Check if we can connect to nearest bus
        nearest_bus, distance = find_nearest_bus(
            geom, edisgo_object.topology.mv_grid.buses_df)
        if distance < DISTANCE_THRESHOLD:
            gen_bus = nearest_bus

    if not gen_bus:

        if comp_type == "Generator":
            # FIXME: Needs to be passed as 'name' here, but is referred to as 'generator_id' by add_generator function. Should be unified across function calls.
            gen_bus = "Bus_Generator_{}".format(generator.name)
        else:
            gen_bus = "Bus_ChargingPoint_{}".format(
                len(edisgo_object.topology.charging_points_df))

        edisgo_object.topology.add_bus(
            bus_name=gen_bus,
            v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
            x=geom.x,
            y=geom.y,
        )

    # add component
    if comp_type == "Generator":
        comp_name = edisgo_object.topology.add_generator(
            generator_id=generator.name,
            bus=gen_bus,
            p_nom=generator["electrical_capacity"],
            # FIXME: Needs to be passed as 'generation_type' here, but is referred to as 'generator_type' by add_generator function. Should be unified across function calls.
            generator_type=generator["generation_type"],
            # FIXME: Needs to be passed as 'generation_subtype' here, but is referred to as 'subtype' by add_generator function. Should be unified across function calls.
            subtype=generator["generation_subtype"],
            # FIXME: Needs to be passed as 'w_id' here, but is referred to as 'weather_cell_id' by add_generator function. Should be unified across function calls.
            weather_cell_id=generator.w_id,
        )
    # add charging point
    else:
        comp_name = edisgo_object.topology.add_charging_point(
            bus=gen_bus,
            p_nom=generator.electrical_capacity,
        )

    # ===== voltage level 4: generator is connected to MV station =====
    if generator.voltage_level == 4:

        # add line

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=gen_bus,
            bus_target=edisgo_object.topology.mv_grid.station.index[0],
        )

        line_name = edisgo_object.topology.add_line(
            bus0=edisgo_object.topology.mv_grid.station.index[0],
            bus1=gen_bus,
            length=line_length,
            kind="cable",
            type_info=std_line_type.name,
        )

        # add line to equipment changes to track costs
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name],
        )

    # == voltage level 5: generator is connected to MV grid (next-neighbor) ==
    elif generator.voltage_level == 5:
        # in case generator was connected to existing bus, it's bus does
        # not need to be connected anymore
        if distance >= DISTANCE_THRESHOLD:
            # get branches within the predefined radius `generator_buffer_radius`
            # get params from config
            lines = calc_geo_lines_in_buffer(
                edisgo_object=edisgo_object,
                bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
                grid=edisgo_object.topology.mv_grid,
            )

            # calc distance between generator and grid's lines -> find nearest line
            conn_objects_min_stack = find_nearest_conn_objects(
                edisgo_object=edisgo_object,
                bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
                lines=lines,
            )

            # connect
            # go through the stack (from nearest to farthest connection target
            # object)
            # ToDo: line connecting a generator in voltage level 4 to the MV
            #  station should be discarded as a valid connection object
            generator_connected = False
            for dist_min_obj in conn_objects_min_stack:
                target_obj_result = connect_mv_node(
                    edisgo_object=edisgo_object,
                    bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
                    target_obj=dist_min_obj,
                )

                if target_obj_result is not None:
                    generator_connected = True
                    break

            if not generator_connected:
                logger.debug(
                    "Generator {} could not be connected, try to "
                    "increase the parameter `conn_buffer_radius` in "
                    "config file `config_grid.cfg` to gain more possible "
                    "connection points.".format(comp_name)
                )
    return comp_name


def add_and_connect_lv_generator(
    edisgo_object, generator, allow_multiple_genos_per_load=True
):
    """
    Add and connect new LV generator to existing grids.

    It connects

        * generators with an MV-LV station ID that does not exist (i.e.
          generators in an aggregated load area)
            * to MV-LV station

        * generators of voltage level 6
            * to MV-LV station

        * generators of voltage level 7
            * with a nom. capacity of <=30 kW to LV loads of type residential
            * with a nom. capacity of >30 kW and <=100 kW to LV loads of type
                retail, industrial or agricultural
            * to the MV-LV station if no appropriate load is available
              (fallback)

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    generator : pd.Series
        Pandas series with generator information such as electrical_capacity
        in MW and generation_type.
    allow_multiple_genos_per_load : :obj:`bool`
        If True, more than one generator can be connected to one load.

    Returns
    -------
    str
        Representative of new component.

    Notes
    -----
    For the allocation, loads are selected randomly (sector-wise) using a
    predefined seed to ensure reproducibility.

    """

    # get list of LV grid IDs
    lv_grid_ids = [_.id for _ in edisgo_object.topology.mv_grid.lv_grids]

    add_generator_data = {
        "generator_id": generator.name,
        "p_nom": generator.electrical_capacity,
        "generator_type": generator.generation_type,
        "subtype": generator.generation_subtype,
        "weather_cell_id": generator.w_id,
    }

    # determine LV grid the generator should be connected in

    # if substation ID (= LV grid ID) is given but it does not match an
    # existing LV grid ID (i.e. it is an aggregated LV grid), connect
    # generator to HV-MV substation
    if (
        generator.mvlv_subst_id
        and generator.mvlv_subst_id not in lv_grid_ids
    ):
        # add generator
        comp_name = edisgo_object.topology.add_generator(
            bus=edisgo_object.topology.mv_grid.station.index[0],
            **add_generator_data
        )
        return comp_name

    # if substation ID (= LV grid ID) is given and it matches an existing LV
    # grid ID (i.e. it is not an aggregated LV grid), set grid to connect
    # generator to to specified grid (in case the generator has no geometry
    # it is connected to the grid's station)
    elif (
        generator.mvlv_subst_id and generator.mvlv_subst_id in lv_grid_ids
    ):

        # get LV grid
        lv_grid = edisgo_object.topology._grids[
            "LVGrid_{}".format(int(generator.mvlv_subst_id))
        ]

        # if no geom is given, connect to LV grid's station
        if not generator.geom:
            # add generator
            comp_name = edisgo_object.topology.add_generator(
                bus=lv_grid.station.index[0], **add_generator_data
            )
            logger.debug(
                "Generator {} has no geom entry and will be connected to "
                "grid's LV stations.".format(generator.name)
            )
            return comp_name

    # if no MV-LV substation ID is given, choose random LV grid and connect
    # to station
    else:
        random.seed(a=generator.name)
        lv_grid_id = random.choice(lv_grid_ids)
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_object)
        comp_name = edisgo_object.topology.add_generator(
            bus=lv_grid.station.index[0], **add_generator_data
        )
        logger.warning(
            "Generator {} has no mvlv_subst_id. It is therefore allocated to "
            "a random LV Grid ({}); geom was set to stations' geom.".format(
                generator.name, lv_grid_id
            )
        )
        return comp_name

    # generator is of v_level 6 -> connect to grid's LV station
    if generator.voltage_level == 6:

        gen_bus = "Bus_Generator_{}".format(generator.name)
        geom = wkt_loads(generator.geom)
        edisgo_object.topology.add_bus(
            bus_name=gen_bus,
            v_nom=lv_grid.nominal_voltage,
            x=geom.x,
            y=geom.y,
            lv_grid_id=lv_grid.id,
        )

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=gen_bus,
            bus_target=lv_grid.station.index[0],
        )

        # get standard equipment
        std_line_type = edisgo_object.topology.equipment_data[
            "lv_cables"
        ].loc[
            edisgo_object.config["grid_expansion_standard_equipment"][
                "lv_line"
            ]
        ]
        line_name = edisgo_object.topology.add_line(
            bus0=lv_grid.station.index[0],
            bus1=gen_bus,
            length=line_length,
            kind="cable",
            type_info=std_line_type.name,
        )

        # add line to equipment changes to track costs
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name],
        )

        # add generator
        comp_name = edisgo_object.topology.add_generator(
            bus=gen_bus, **add_generator_data
        )
        return comp_name

    # generator is of v_level 7 -> assign generator to load
    # generators with P <= 30kW are connected to residential loads, if
    # available; generators with 30kW <= P <= 100kW are connected to
    # retail, industrial, or agricultural load, if available
    # in case no load is available the generator is connected to random
    # bus in LV grid
    # if load to connect to is available, the generator is connected to
    # load with less than two generators
    # if every load has two or more generators choose the first load
    # from random sample
    elif generator.voltage_level == 7:

        lv_loads = lv_grid.loads_df
        if generator.electrical_capacity <= 0.030:
            target_loads = lv_loads[lv_loads.sector == "residential"]
        else:
            target_loads = lv_loads[
                lv_loads.sector.isin(
                    ["industrial", "agricultural", "retail"]
                )
            ]

        # generate random list (unique elements) of possible target loads
        # to connect generators to
        random.seed(a=int(generator.name))
        if len(target_loads) > 0:
            lv_loads_rnd = random.sample(
                sorted(list(target_loads.index)),
                len(target_loads))
        else:
            logger.debug(
                "No load to connect LV generator to. The "
                "generator is therefore connected to random LV bus."
            )
            gen_bus = random.choice(
                lv_grid.buses_df[~lv_grid.buses_df.in_building].index
            )
            # add generator
            comp_name = edisgo_object.topology.add_generator(
                bus=gen_bus, **add_generator_data
            )
            return comp_name

        # search through list of loads for load with less
        # than two generators
        lv_conn_target = None
        while len(lv_loads_rnd) > 0 and lv_conn_target is None:

            lv_load = lv_loads_rnd.pop()

            # determine number of generators of LV load
            load_bus = target_loads.at[lv_load, "bus"]
            if np.logical_not(
                edisgo_object.topology.buses_df.at[load_bus, "in_building"]
            ):
                neighbours = list(
                    edisgo_object.topology.get_neighbours(load_bus)
                )
                branch_tee_in_building = neighbours[0]
                if len(neighbours) > 1 or np.logical_not(
                    edisgo_object.topology.buses_df.at[
                        branch_tee_in_building, "in_building"
                    ]
                ):
                    raise ValueError(
                        "Expected neighbour to be branch tee in building."
                    )
            else:
                branch_tee_in_building = load_bus
            generators_at_load = edisgo_object.topology.generators_df[
                edisgo_object.topology.generators_df.bus.isin(
                    [load_bus, branch_tee_in_building]
                )
            ]
            if len(generators_at_load) < 2:
                lv_conn_target = branch_tee_in_building

        if lv_conn_target is None:
            logger.debug(
                "No valid connection target found for generator {}. "
                "Connected to LV station.".format(generator.name)
            )

            station_bus = lv_grid.station.index[0]

            gen_bus = "Bus_Generator_{}".format(generator.name)
            lv_conn_target = gen_bus
            geom = wkt_loads(generator.geom)
            edisgo_object.topology.add_bus(
                bus_name=gen_bus,
                v_nom=lv_grid.nominal_voltage,
                x=geom.x,
                y=geom.y,
                lv_grid_id=lv_grid.id,
            )

            line_length = calc_geo_dist_vincenty(
                edisgo_object=edisgo_object,
                bus_source=gen_bus,
                bus_target=station_bus,
            )

            # get standard equipment
            std_line_type = edisgo_object.topology.equipment_data[
                "lv_cables"
            ].loc[
                edisgo_object.config["grid_expansion_standard_equipment"][
                    "lv_line"
                ]
            ]
            line_name = edisgo_object.topology.add_line(
                bus0=station_bus,
                bus1=gen_bus,
                length=line_length,
                kind="cable",
                type_info=std_line_type.name,
            )

            # add line to equipment changes to track costs
            add_line_to_equipment_changes(
                edisgo_object=edisgo_object,
                line=edisgo_object.topology.lines_df.loc[line_name],
            )

        # add generator
        comp_name = edisgo_object.topology.add_generator(
            bus=lv_conn_target, **add_generator_data
        )
        return comp_name


def update_grids(
    edisgo_object,
    imported_generators_mv,
    imported_generators_lv,
    remove_missing=True,
):
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
        [imported_generators_lv, imported_generators_mv]
    )

    logger.debug("{} generators imported.".format(len(imported_gens)))

    # get existing generators in MV and LV grids and append ID column
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
    gens_to_remove = imported_gens[imported_gens.electrical_capacity <= 0]
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

    # filter for generators that need to be updated (i.e. that
    # appear in the imported and existing generators dataframes)
    gens_to_update = existing_gens[
        existing_gens.id.isin(imported_gens.index.values)
    ]

    # calculate capacity difference between existing and imported
    # generators
    gens_to_update["cap_diff"] = (
        imported_gens.loc[gens_to_update.id, "electrical_capacity"].values
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
        ] = imported_gens.loc[row["id"], "electrical_capacity"]

    log_geno_count = len(gens_to_update_cap)
    log_geno_cap = gens_to_update_cap["cap_diff"].sum()
    logger.debug(
        "Capacities of {} of {} existing generators updated ({} MW).".format(
            log_geno_count, len(gens_to_update), round(log_geno_cap, 1)
        )
    )

    # ==================================================
    # Step 2: Remove decommissioned MV and LV generators
    # ==================================================

    # filter for MV generators that do not appear in the imported but in
    # the existing generators dataframe
    decommissioned_gens = existing_gens[
        ~existing_gens.id.isin(imported_gens.index.values)
    ]

    if not decommissioned_gens.empty and remove_missing:
        for gen in decommissioned_gens.index:
            edisgo_object.topology.remove_generator(gen)
        log_geno_cap = decommissioned_gens.p_nom.sum()
        log_geno_count = len(decommissioned_gens)
        logger.debug(
            "{} decommissioned generators removed ({} MW).".format(
                log_geno_count, round(log_geno_cap, 1)
            )
        )

    # ===============================
    # Step 3: Add new MV generators
    # ===============================

    new_gens_mv = imported_generators_mv[
        ~imported_generators_mv.index.isin(list(existing_gens.id))
    ]
    number_new_gens = len(new_gens_mv)

    # iterate over new generators and create them
    for id in new_gens_mv.index:
        # check if geom is available, skip otherwise
        geom = check_mv_generator_geom(new_gens_mv.loc[id, :])
        if geom is None:
            logger.warning(
                "Generator {} has no geom entry and will"
                "not be imported!".format(id)
            )
            new_gens_mv.drop(id)
            continue
        new_gens_mv.at[id, "geom"] = geom
        add_and_connect_mv_generator(
            edisgo_object, new_gens_mv.loc[id, :]
        )

    log_geno_count = len(new_gens_mv)
    log_geno_cap = new_gens_mv["electrical_capacity"].sum()
    logger.debug(
        "{} of {} new MV generators added ({} MW).".format(
            log_geno_count, number_new_gens, round(log_geno_cap, 1)
        )
    )

    # ===============================
    # Step 4: Add new LV generators
    # ===============================

    new_gens_lv = imported_generators_lv[
        ~imported_generators_lv.index.isin(list(existing_gens.id))
    ]

    # check if new generators can be allocated to an existing LV grid
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
    for id in new_gens_lv.index:
        add_and_connect_lv_generator(
            edisgo_object, new_gens_lv.loc[id, :]
        )

    log_geno_count = len(new_gens_lv)
    log_geno_cap = new_gens_lv["electrical_capacity"].sum()
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


def check_mv_generator_geom(generator_data):
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
                "Generator {} has no geom entry, EnergyMap's geom "
                "entry will be used.".format(generator_data.name)
            )
            return generator_data.geom_em
    return None


def add_line_to_equipment_changes(edisgo_object, line):
    """
    Add line to the equipment changes.

    All changes of equipment are stored in edisgo.results.equipment_changes
    which is used later to determine network expansion costs.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    line : pd.Series
        Data of line to add.
        Series has same rows as columns of topology.lines_df. Line
        representative is the series name.

    """
    edisgo_object.results.equipment_changes = edisgo_object.results.equipment_changes.append(
        pd.DataFrame(
            {
                "iteration_step": [0],
                "change": ["added"],
                "equipment": [line.type_info],
                "quantity": [1],
            },
            index=[line.name],
        )
    )


def del_line_from_equipment_changes(edisgo_object, line_repr):
    """
    Delete line from the equipment changes if it exists.

    This is needed when a line was already added to
    Results.equipment_changes but another component is later connected
    to this line. Therefore, the line needs to be split which changes the
    representative of the line and the line data.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    line_repr : str
        Line representative as in topology.lines_df.index.

    """
    if line_repr in edisgo_object.results.equipment_changes.index:
        edisgo_object.results.equipment_changes = edisgo_object.results.equipment_changes.drop(
            line_repr
        )


def find_nearest_conn_objects(edisgo_object, bus, lines):
    """
    Searches all lines for the nearest possible connection object per line.

    It picks out 1 object out of 3 possible objects: 2 branch-adjacent buses
    and 1 potentially created branch tee on the line (using perpendicular
    projection). The resulting stack (list) is sorted ascending by distance
    from bus.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus : pandas Series
        Data of bus to connect.
        Series has same rows as columns of topology.buses_df.
    lines : list(str)
        List of line representatives from topology.lines_df.index

    Returns
    -------
    :obj:`list` of :obj:`dict`
        List of connection objects (each object is represented by dict with
        representative, shapely object and distance to node.

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/network/mv_grid/mv_connect.py#L38>`_.

    """

    # threshold which is used to determine if 2 objects are at the same
    # position (see below for details on usage)
    conn_diff_tolerance = edisgo_object.config["grid_connection"][
        "conn_diff_tolerance"
    ]

    conn_objects_min_stack = []

    srid = edisgo_object.topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    for line in lines:

        line_bus0 = edisgo_object.topology.buses_df.loc[
            edisgo_object.topology.lines_df.loc[line, "bus0"]
        ]
        line_bus1 = edisgo_object.topology.buses_df.loc[
            edisgo_object.topology.lines_df.loc[line, "bus1"]
        ]

        # create shapely objects for 2 buses and line between them,
        # transform to equidistant CRS
        line_bus0_shp = transform(
            proj2equidistant(srid), Point(line_bus0.x, line_bus0.y)
        )
        line_bus1_shp = transform(
            proj2equidistant(srid), Point(line_bus1.x, line_bus1.y)
        )
        line_shp = LineString([line_bus0_shp, line_bus1_shp])

        # create dict with line & 2 adjacent buses and their shapely objects
        # and distances
        conn_objects = {
            "s1": {
                "repr": line_bus0.name,
                "shp": line_bus0_shp,
                "dist": bus_shp.distance(line_bus0_shp) * 0.999,
            },
            "s2": {
                "repr": line_bus1.name,
                "shp": line_bus1_shp,
                "dist": bus_shp.distance(line_bus1_shp) * 0.999,
            },
            "b": {
                "repr": line,
                "shp": line_shp,
                "dist": bus_shp.distance(line_shp),
            },
        }

        # remove line from the dict of possible conn. objects if it is too
        # close to the bus (necessary to assure that connection target is
        # reproducible)
        if (
            abs(conn_objects["s1"]["dist"] - conn_objects["b"]["dist"])
            < conn_diff_tolerance
            or abs(conn_objects["s2"]["dist"] - conn_objects["b"]["dist"])
            < conn_diff_tolerance
        ):
            del conn_objects["b"]

        # remove MV station as possible connection point
        if (
            conn_objects["s1"]["repr"]
            == edisgo_object.topology.mv_grid.station.index[0]
        ):
            del conn_objects["s1"]
        elif (
            conn_objects["s2"]["repr"]
            == edisgo_object.topology.mv_grid.station.index[0]
        ):
            del conn_objects["s2"]

        # find nearest connection point in conn_objects
        conn_objects_min = min(
            conn_objects.values(), key=lambda v: v["dist"]
        )

        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [
        _ for _ in sorted(conn_objects_min_stack, key=lambda x: x["dist"])
    ]

    return conn_objects_min_stack


def connect_mv_node(edisgo_object, bus, target_obj):
    """
    Connects MV generators to target object in MV network

    If the target object is a bus, a new line is created to it.
    If the target object is a line, the node is connected to a newly created
    bus (using perpendicular projection) on this line.
    New lines are created using standard equipment.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus : pandas Series
        Data of bus to connect.
        Series has same rows as columns of topology.buses_df.
    target_obj : :class:`~.network.components.Component`
        Object that node shall be connected to

    Returns
    -------
    :class:`~.network.components.Component` or None
        Node that node was connected to

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/network/mv_grid/mv_connect.py#L311>`_.

    """

    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data["mv_cables"].loc[
        edisgo_object.config["grid_expansion_standard_equipment"][
            "mv_line"
        ]
    ]
    std_line_kind = "cable"

    srid = edisgo_object.topology.grid_district["srid"]
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    # MV line is nearest connection point => split old line into 2 segments
    # (delete old line and create 2 new ones)
    if isinstance(target_obj["shp"], LineString):

        line_data = edisgo_object.topology.lines_df.loc[
            target_obj["repr"], :
        ]

        # find nearest point on MV line
        conn_point_shp = target_obj["shp"].interpolate(
            target_obj["shp"].project(bus_shp)
        )
        conn_point_shp = transform(
            proj2equidistant_reverse(srid), conn_point_shp
        )

        # create new branch tee bus
        branch_tee_repr = "BranchTee_{}".format(target_obj["repr"])
        edisgo_object.topology.add_bus(
            bus_name=branch_tee_repr,
            v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
            x=conn_point_shp.x,
            y=conn_point_shp.y,
        )

        # add new line between newly created branch tee and line's bus0
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=line_data.bus0,
            bus_target=branch_tee_repr,
        )

        line_name_bus0 = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=line_data.bus0,
            length=line_length,
            kind=line_data.kind,
            type_info=line_data.type_info,
        )

        # add line to equipment changes
        # ToDo @Anya?
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name_bus0, :],
        )

        # add new line between newly created branch tee and line's bus0
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=line_data.bus1,
            bus_target=branch_tee_repr,
        )

        line_name_bus1 = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=line_data.bus1,
            length=line_length,
            kind=line_data.kind,
            type_info=line_data.type_info,
        )

        # add line to equipment changes
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name_bus1, :],
        )

        # add new line for new bus
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus.name,
            bus_target=branch_tee_repr,
        )

        new_line_name = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=bus.name,
            length=line_length,
            kind=std_line_kind,
            type_info=std_line_type.name,
        )

        # add line to equipment changes
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[new_line_name, :],
        )

        # remove old line from topology and equipment changes
        edisgo_object.topology.remove_line(line_data.name)

        del_line_from_equipment_changes(
            edisgo_object=edisgo_object, line_repr=line_data.name
        )

        return branch_tee_repr

    # node ist nearest connection point
    else:

        # add new branch for satellite (station to station)
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus.name,
            bus_target=target_obj["repr"],
        )

        new_line_name = edisgo_object.topology.add_line(
            bus0=target_obj["repr"],
            bus1=bus.name,
            length=line_length,
            kind=std_line_kind,
            type_info=std_line_type.name,
        )

        # add line to equipment changes
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[new_line_name, :],
        )

        return target_obj["repr"]





