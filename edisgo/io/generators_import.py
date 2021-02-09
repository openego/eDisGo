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
)

logger = logging.getLogger("edisgo")

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads


def oedb(edisgo_object, **kwargs):
    """
    Import generator data from the Open Energy Database (OEDB).

    The importer uses SQLAlchemy ORM objects.
    These are defined in ego.io,
    see https://github.com/openego/ego.io/tree/dev/egoio/db_tables

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`

    Other Parameters
    ----------------
    remove_missing : bool
        See :func:`edisgo.io.generators_import.update_grids` for more
        information.
    update_existing : bool
        See :func:`edisgo.io.generators_import.update_grids` for more
        information.
    p_target : dict
        See :func:`edisgo.io.generators_import.update_grids` for more
        information.
    allowed_number_of_comp_per_lv_bus : int
        See :func:`edisgo.io.generators_import.update_grids` for more
        information.

    Notes
    ------
    Right now only solar and wind generators can be imported.

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
        If subtype is not specified it's set to 'unknown'.

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

        capacity_imported = (
                                    generators_res_mv["p_nom"].sum()
                                    + generators_res_lv["p_nom"].sum()
                            ) + generators_conv_mv['p_nom'].sum()

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
        **kwargs
    )

    if kwargs.get('p_target', None) is None:
        _validate_generation()

    # update time series if they were already set
    if not edisgo_object.timeseries.generators_active_power.empty:
        add_generators_timeseries(
            edisgo_obj=edisgo_object,
            generator_names=edisgo_object.topology.generators_df.index)


def connect_to_mv(edisgo_object, comp_data, comp_type="Generator"):
    """
    Add and connect new generator or charging point to MV grid.

    # ToDo Update docstring
    This function connects

        * components of voltage level 4
            * to HV-MV station

        * components of voltage level 5
            * to nearest MV bus or line
            * in case component is connected to a line, the line is split and
              a new branch tee is added to connect new components to

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    comp_data : dict
        Dictionary with all information on component.
        The dictionary must contain all required arguments
        of method :attr:`~.network.topology.Topology.add_generator`
        respectively
        :attr:`~.network.topology.Topology.add_charging_point`, except the
        `bus` that is assigned in this function, and may contain all other
        parameters of those methods. Additionally the dictionary must contain
        the voltage level to connect in and geometry.
    comp_type : str
        Type of added component. Can be 'Generator' or 'ChargingPoint'.
        Default: 'Generator'.

    Returns
    -------
    str
        The identifier of the newly connected component.

    """
    # ToDo connect charging points via transformer?

    # ToDo use select_cable instead of standard line?
    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data["mv_cables"].loc[
        edisgo_object.config["grid_expansion_standard_equipment"][
            "mv_line"
        ]
    ]

    # create new bus for new component
    if not type(comp_data["geom"]) is Point:
        geom = wkt_loads(comp_data["geom"])
    else:
        geom = comp_data["geom"]

    if comp_type == "Generator":
        if comp_data["generator_id"] is not None:
            bus = "Bus_Generator_{}".format(comp_data["generator_id"])
        else:
            bus = "Bus_Generator_{}".format(
                len(edisgo_object.topology.generators_df))
    else:
        bus = "Bus_ChargingPoint_{}".format(
            len(edisgo_object.topology.charging_points_df))

    edisgo_object.topology.add_bus(
        bus_name=bus,
        v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
        x=geom.x,
        y=geom.y,
    )

    # add component to newly created bus
    if comp_type == "Generator":
        comp_name = edisgo_object.topology.add_generator(
            bus=bus,
            **comp_data
        )
    else:
        comp_name = edisgo_object.topology.add_charging_point(
            bus=bus,
            **comp_data
        )

    # ===== voltage level 4: component is connected to MV station =====
    if comp_data["voltage_level"] == 4:

        # add line
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus,
            bus_target=edisgo_object.topology.mv_grid.station.index[0],
        )

        line_name = edisgo_object.topology.add_line(
            bus0=edisgo_object.topology.mv_grid.station.index[0],
            bus1=bus,
            length=line_length,
            kind="cable",
            type_info=std_line_type.name,
        )

        # add line to equipment changes to track costs
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name],
        )

    # == voltage level 5: component is connected to MV grid (next-neighbor) ==
    elif comp_data["voltage_level"] == 5:

        # get branches within the predefined radius `connection_buffer_radius`
        lines = calc_geo_lines_in_buffer(
            edisgo_object=edisgo_object,
            bus=edisgo_object.topology.buses_df.loc[bus, :],
            grid=edisgo_object.topology.mv_grid,
        )

        # calc distance between component and grid's lines -> find nearest line
        conn_objects_min_stack = find_nearest_conn_objects(
            edisgo_object=edisgo_object,
            bus=edisgo_object.topology.buses_df.loc[bus, :],
            lines=lines,
        )

        # connect
        # go through the stack (from nearest to farthest connection target
        # object)
        comp_connected = False
        for dist_min_obj in conn_objects_min_stack:
            # do not allow connection to virtual busses
            if not "virtual" in dist_min_obj["repr"]:
                target_obj_result = connect_mv_node(
                    edisgo_object=edisgo_object,
                    bus=edisgo_object.topology.buses_df.loc[bus, :],
                    target_obj=dist_min_obj,
                )

                if target_obj_result is not None:
                    comp_connected = True
                    break

        if not comp_connected:
            logger.error(
                "Component {} could not be connected, try to "
                "increase the parameter `conn_buffer_radius` in "
                "config file `config_grid.cfg` to gain more possible "
                "connection points.".format(comp_name)
            )
    return comp_name


def connect_to_lv(edisgo_object, comp_data, comp_type="Generator",
                  allowed_number_of_comp_per_bus=2):
    """
    Add and connect new generator or charging point to LV grid.

    It connects

        * generators with no or an MV-LV station ID that does not exist (i.e.
          generators in an aggregated load area)
            * to HV-MV station

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
    comp_data : dict
        Dictionary with all information on component.
        The dictionary must contain all required arguments
        of method :attr:`~.network.topology.Topology.add_generator`
        respectively
        :attr:`~.network.topology.Topology.add_charging_point`, except the
        `bus` that is assigned in this function, and may contain all other
        parameters of those methods.
    comp_type : str
        Type of added component. Can be 'Generator' or 'ChargingPoint'.
        Default: 'Generator'.
    allowed_number_of_comp_per_bus : int
        Specifies, how many generators respectively charging points are
        at most allowed to be placed at the same bus. Default: 2.

    Returns
    -------
    str
        The identifier of the newly connected component.

    Notes
    -----
    For the allocation, loads are selected randomly (sector-wise) using a
    predefined seed to ensure reproducibility.

    """

    def _connect_to_station():
        """
        Connects new component to substation via an own bus.
        """

        # add bus for new component
        if comp_type == "Generator":
            if comp_data["generator_id"] is not None:
                bus = "Bus_Generator_{}".format(comp_data["generator_id"])
            else:
                bus = "Bus_Generator_{}".format(
                    len(edisgo_object.topology.generators_df))
        else:
            bus = "Bus_ChargingPoint_{}".format(
                len(edisgo_object.topology.charging_points_df))

        if not type(comp_data["geom"]) is Point:
            geom = wkt_loads(comp_data["geom"])
        else:
            geom = comp_data["geom"]

        edisgo_object.topology.add_bus(
            bus_name=bus,
            v_nom=lv_grid.nominal_voltage,
            x=geom.x,
            y=geom.y,
            lv_grid_id=lv_grid.id,
        )

        # add line to connect new component
        station_bus = lv_grid.station.index[0]
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus,
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
            bus1=bus,
            length=line_length,
            kind="cable",
            type_info=std_line_type.name,
        )

        # add line to equipment changes to track costs
        add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name],
        )

        # add new component
        comp_name = add_func(
            bus=bus, **comp_data
        )
        return comp_name

    # get list of LV grid IDs
    lv_grid_ids = [_.id for _ in edisgo_object.topology.mv_grid.lv_grids]

    if comp_type == "Generator":
        add_func = edisgo_object.topology.add_generator
    elif comp_type == "ChargingPoint":
        add_func = edisgo_object.topology.add_charging_point
    else:
        logger.error(
            "Component type {} is not a valid option.".format(comp_type)
        )

    if comp_data["mvlv_subst_id"]:

        # if substation ID (= LV grid ID) is given and it matches an existing
        # LV grid ID (i.e. it is no aggregated LV grid), set grid to connect
        # component to to specified grid (in case the component has no geometry
        # it is connected to the grid's station)
        if comp_data["mvlv_subst_id"] in lv_grid_ids:

            # get LV grid
            lv_grid = edisgo_object.topology._grids[
                "LVGrid_{}".format(int(comp_data["mvlv_subst_id"]))
            ]

            # if no geom is given, connect to LV grid's station
            if not comp_data["geom"]:
                comp_name = add_func(
                    bus=lv_grid.station.index[0], **comp_data
                )
                logger.debug(
                    "Component {} has no geom entry and will be connected to "
                    "grid's LV station.".format(comp_name)
                )
                return comp_name

        # if substation ID (= LV grid ID) is given but it does not match an
        # existing LV grid ID (i.e. it is an aggregated LV grid), connect
        # component to HV-MV substation
        # ToDo: Keep it like this?
        else:
            comp_name = add_func(
                bus=edisgo_object.topology.mv_grid.station.index[0],
                **comp_data
            )
            return comp_name

    # if no MV-LV substation ID is given (and there is therefore also no
    # geometry data), choose random LV grid and connect to station
    else:
        if comp_type == "Generator":
            random.seed(a=comp_data["generator_id"])
        else:
            # ToDo: Seed shouldn't depend on number of charging points, but
            #  there is currently no better solution
            random.seed(a=len(edisgo_object.topology.charging_points_df))
        lv_grid_id = random.choice(lv_grid_ids)
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_object)
        comp_name = add_func(
            bus=lv_grid.station.index[0], **comp_data
        )
        logger.warning(
            "Component {} has no mvlv_subst_id. It is therefore allocated to "
            "a random LV Grid ({}).".format(
                comp_name, lv_grid_id
            )
        )
        return comp_name

    # v_level 6 -> connect to grid's LV station
    if comp_data["voltage_level"] == 6:
        comp_name = _connect_to_station()
        return comp_name

    # v_level 7 -> assign generator to load
    # Generators:
    # Generators with P <= 30 kW are connected to residential loads, if
    # available; generators with 30 kW <= P <= 100 kW are connected to
    # retail, industrial, or agricultural loads, if available.
    # Charging Points:
    # Charging points with use case 'home' are connected to residential loads,
    # if available; charging points with use case 'work' are connected to
    # retail, industrial, or agricultural loads, if available; charging points
    # with other use cases ('public' or 'fast') are connected somewhere in the
    # grid.
    # In case the above described criteria do not give a bus to connect to,
    # the generator or charging point is connected to a random bus in the LV
    # grid.
    # If there are valid buses, the generator or charging point is connected
    # to a bus out of the valid buses with less than or equal the allowed
    # number of generators / charging points at one bus.
    # If every one of the valid buses already has the allowed number of
    # generators / charging points, the new component is directly
    # connected to the substation.
    elif comp_data["voltage_level"] == 7:

        # get valid buses to connect new component to
        lv_loads = lv_grid.loads_df
        if comp_type == "Generator":
            if comp_data["p_nom"] <= 0.030:
                tmp = lv_loads[lv_loads.sector == "residential"]
                target_buses = tmp.bus.values
            else:
                tmp = lv_loads[
                    lv_loads.sector.isin(
                        ["industrial", "agricultural", "retail"]
                    )
                ]
                target_buses = tmp.bus.values
        else:
            if comp_data["use_case"] is "home":
                tmp = lv_loads[lv_loads.sector == "residential"]
                target_buses = tmp.bus.values
            elif comp_data["use_case"] is "work":
                tmp = lv_loads[
                    lv_loads.sector.isin(
                        ["industrial", "agricultural", "retail"]
                    )
                ]
                target_buses = tmp.bus.values
            else:
                target_buses = lv_grid.buses_df[
                    ~lv_grid.buses_df.in_building].index

        # generate random list (unique elements) of possible target buses
        # to connect components to
        if comp_type == "Generator":
            random.seed(a=comp_data["generator_id"])
        else:
            random.seed(
                a="{}_{}".format(comp_data["use_case"], comp_data["p_nom"]))

        if len(target_buses) > 0:
            lv_buses_rnd = random.sample(
                sorted(list(target_buses)),
                len(target_buses))
        else:
            logger.debug(
                "No valid bus to connect new LV component to. The "
                "component is therefore connected to random LV bus."
            )
            bus = random.choice(
                lv_grid.buses_df[~lv_grid.buses_df.in_building].index
            )
            comp_name = add_func(
                bus=bus, **comp_data
            )
            return comp_name

        # search through list of target buses for bus with less
        # than two generators / charging points
        lv_conn_target = None

        # ToDo: Once export in ding0 connects generators directly to bus with
        #  load, the following distinction does not need to be made anymore.
        if comp_type == "Generator" or (
                comp_type == "ChargingPoint" and
                comp_data["use_case"] in ["home", "work"]):

            while len(lv_buses_rnd) > 0 and lv_conn_target is None:

                lv_bus = lv_buses_rnd.pop()

                # determine number of generators / charging points at LV bus
                if not lv_grid.buses_df.at[lv_bus, "in_building"]:
                    neighbours = list(
                        edisgo_object.topology.get_neighbours(lv_bus)
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
                    branch_tee_in_building = lv_bus
                # ToDo: Do generators at loads exported from ding0 have own
                #  bus? If so, the following needs to be changed.
                if comp_type == "Generator":
                    comps_at_load = edisgo_object.topology.generators_df[
                        edisgo_object.topology.generators_df.bus.isin(
                            [lv_bus, branch_tee_in_building]
                        )
                    ]
                else:
                    comps_at_load = edisgo_object.topology.charging_points_df[
                        edisgo_object.topology.charging_points_df.bus.isin(
                            [lv_bus, branch_tee_in_building]
                        )
                    ]
                if len(comps_at_load) <= allowed_number_of_comp_per_bus:
                    lv_conn_target = branch_tee_in_building

        else:

            while len(lv_buses_rnd) > 0 and lv_conn_target is None:

                lv_bus = lv_buses_rnd.pop()

                # determine number of charging points at LV bus
                comps_at_load = edisgo_object.topology.charging_points_df[
                    edisgo_object.topology.charging_points_df.bus == lv_bus]
                # ToDo: Increase number of generators/charging points allowed
                #  at one load in case all loads already have one
                #  generator/charging point
                if len(comps_at_load) <= allowed_number_of_comp_per_bus:
                    lv_conn_target = lv_bus

        if lv_conn_target is None:
            logger.debug(
                "No valid connection target found for new component. "
                "Connected to LV station."
            )
            comp_name = _connect_to_station()
        else:
            comp_name = add_func(
                bus=lv_conn_target, **comp_data
            )
        return comp_name


def update_grids(
        edisgo_object,
        imported_generators_mv,
        imported_generators_lv,
        remove_missing=True,
        update_existing=True,
        p_target=None,
        allowed_number_of_comp_per_lv_bus=2
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
    edisgo_object : :class:`~.EDisGo`
    imported_generators_mv : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all MV generators.
        Index of the dataframe are the generator IDs.
        Columns:

            * p_nom : float
                (unit: MW)
            * generator_type: :obj:`str` (e.g. 'solar')
            * subtype: :obj:`str` (e.g. 'solar_roof_mounted')
            * voltage_level: :obj:`int` (range: 4..7,)
            * geom: :shapely:`Shapely Point object<points>`
              (CRS see config_grid.cfg)
            * geom_em: :shapely:`Shapely Point object<points>`
              (CRS see config_grid.cfg)

    imported_generators_lv : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing all LV generators.
        Index of the dataframe are the generator IDs.
        Columns are:

            * mvlv_subst_id: :obj:`int` (id of MV-LV substation in grid
              = grid which the generator will be connected to)
            * p_nom: :obj:`float` (unit: kW)
            * generator_type: :obj:`str` (e.g. 'solar')
            * subtype: :obj:`str` (e.g. 'solar_roof_mounted')
            * voltage_level: :obj:`int` (range: 4..7,)
            * geom: :shapely:`Shapely Point object<points>`
              (CRS see config_grid.cfg)
            * geom_em: :shapely:`Shapely Point object<points>`
              (CRS see config_grid.cfg)

    remove_missing : bool
        If true, remove generators from network which are not included in
        the imported dataset. Default: True.
    update_existing : bool
        If true, updates capacity of already existing generators to
        capacity specified in the imported dataset. Default: True.
    p_target : dict
        Dictionary with total installed target capacity per technology (e.g.
        'wind' or 'solar') in MW. Keys of the dictionary are the technology
        types, values are the corresponding total installed target capacities.
        ToDo: Explain further how this is handled.
    allowed_number_of_comp_per_lv_bus : int
        Specifies, how many generators respectively charging points are
        at most allowed to be placed at the same LV bus. Default: 2.

    """

    # set capacity difference threshold
    cap_diff_threshold = 10 ** -4

    # get all imported generators
    imported_gens = pd.concat(
        [imported_generators_lv, imported_generators_mv],
        sort=True
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

    new_gens_lv = imported_generators_lv[
        ~imported_generators_lv.index.isin(list(existing_gens.id))
    ]

    if p_target is not None:
        def update_imported_gens(layer, imported_gens):
            def drop_generators(generator_list, gen_type, total_capacity):
                random.seed(42)
                while (generator_list[
                           generator_list[
                               'generator_type'] == gen_type].p_nom.sum() > total_capacity and
                       len(generator_list[generator_list[
                                              'generator_type'] == gen_type]) > 0):
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
                # Total capacity in 2030 scenario as described by expansion factor
                target_capacity = p_target[gen_type]
                # Amount of required expansion
                required_expansion = (
                        target_capacity - existing_capacity)

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
    for id in new_gens_mv.index:
        # check if geom is available, skip otherwise
        geom = check_mv_generator_geom(new_gens_mv.loc[id, :])
        if geom is None:
            logger.warning(
                "Generator {} has no geom entry and will "
                "not be imported!".format(id)
            )
            new_gens_mv.drop(id)
            continue
        new_gens_mv.at[id, "geom"] = geom
        connect_to_mv(
            edisgo_object, dict(new_gens_mv.loc[id, :])
        )

    log_geno_count = len(new_gens_mv)
    log_geno_cap = new_gens_mv["p_nom"].sum()
    logger.debug(
        "{} of {} new MV generators added ({} MW).".format(
            log_geno_count, number_new_gens, round(log_geno_cap, 1)
        )
    )

    # ===============================
    # Step 4: Add new LV generators
    # ===============================

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
    for id in new_gens_lv.index:
        connect_to_lv(
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
