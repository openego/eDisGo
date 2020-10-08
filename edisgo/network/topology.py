import logging
import random
import pandas as pd
import numpy as np
import os
import warnings

import edisgo
from edisgo.network.grids import MVGrid, LVGrid
from edisgo.network.components import Generator, Load
from edisgo.tools.tools import (
    calculate_line_resistance,
    calculate_line_reactance,
    calculate_apparent_power,
    check_bus_for_removal,
    check_line_for_removal,
)
from edisgo.io.ding0_import import _validate_ding0_grid_import

logger = logging.getLogger("edisgo")


class Topology:
    """
    Used as container for all data related to a single
    :class:`~.network.grids.MVGrid`.

    Parameters
    -----------
    config : :class:`~.tools.config.Config`
        Config object with configuration data from config files.

    Attributes
    -----------
    _grids : dict
        Dictionary containing all grids (keys are grid representatives and
        values the grid objects)

    """

    # ToDo Implement update (and add) functions for component dataframes to
    # avoid using protected variables in other classes and modules

    def __init__(self, **kwargs):

        # load configuration and equipment data
        self._equipment_data = self._load_equipment_data(
            kwargs.get("config", None)
        )

    @staticmethod
    def _load_equipment_data(config=None):
        """
        Load equipment data for transformers, cables etc.

        Parameters
        -----------
        config : :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        Returns
        -------
        :obj:`dict`
            Dictionary with :pandas:`pandas.DataFrame<DataFrame>` containing
            equipment data. Keys of the dictionary are 'mv_transformers',
            'mv_overhead_lines', 'mv_cables', 'lv_transformers', and
            'lv_cables'.

        Notes
        ------
        This function calculates electrical values of transformer from standard
        values (so far only for LV transformers, not necessary for MV as MV
        impedances are not used).

        $z_{pu}$ is calculated as follows:

        .. math:: z_{pu} = \frac{u_{kr}}{100}

        using the following simplification:

        .. math:: z_{pu} = \frac{Z}{Z_{nom}}

        with

        .. math:: Z = \frac{u_{kr}}{100} \cdot \frac{U_n^2}{S_{nom}}

        and

        .. math:: Z_{nom} = \frac{U_n^2}{S_{nom}}

        $r_{pu}$ is calculated as follows:

        .. math:: r_{pu} = \frac{P_k}{S_{nom}}

        using the simplification of

        .. math:: r_{pu} = \frac{R}{Z_{nom}}

        with

        .. math:: R = \frac{P_k}{3 I_{nom}^2} = P_k \cdot \frac{U_{nom}^2}{S_{nom}^2}

        $x_{pu}$ is calculated as follows:

        .. math::  x_{pu} = \sqrt(z_{pu}^2-r_{pu}^2)


        """

        equipment = {
            "mv": ["transformers", "overhead_lines", "cables"],
            "lv": ["transformers", "cables"],
        }

        # if config is not provided set default path and filenames
        if config is None:
            equipment_dir = "equipment"
            config = {}
            for voltage_level, eq_list in equipment.items():
                for i in eq_list:
                    config[
                        "equipment_{}_parameters_{}".format(voltage_level, i)
                    ] = "equipment-parameters_{}_{}.csv".format(
                        voltage_level.upper(), i
                    )
        else:
            equipment_dir = config["system_dirs"]["equipment_dir"]
            config = config["equipment"]

        package_path = edisgo.__path__[0]
        data = {}

        for voltage_level, eq_list in equipment.items():
            for i in eq_list:
                equipment_parameters = config[
                    "equipment_{}_parameters_{}".format(voltage_level, i)
                ]
                data["{}_{}".format(voltage_level, i)] = pd.read_csv(
                    os.path.join(
                        package_path, equipment_dir, equipment_parameters
                    ),
                    comment="#",
                    index_col="name",
                    delimiter=",",
                    decimal=".",
                )
                # calculate electrical values of transformer from standard
                # values (so far only for LV transformers, not necessary for
                # MV as MV impedances are not used)
                if voltage_level == "lv" and i == "transformers":
                    data["{}_{}".format(voltage_level, i)]["r_pu"] = data[
                        "{}_{}".format(voltage_level, i)
                    ]["P_k"] / (
                        data["{}_{}".format(voltage_level, i)]["S_nom"]
                    )
                    data["{}_{}".format(voltage_level, i)]["x_pu"] = np.sqrt(
                        (data["{}_{}".format(voltage_level, i)]["u_kr"] / 100)
                        ** 2
                        - data["{}_{}".format(voltage_level, i)]["r_pu"] ** 2
                    )
        return data

    @property
    def equipment_data(self):
        """
        Technical data of electrical equipment such as lines and transformers.

        Returns
        --------
        :obj:`dict` of :pandas:`pandas.DataFrame<DataFrame>`
            Data of electrical equipment.

        """
        return self._equipment_data

    @property
    def buses_df(self):
        """
        Dataframe with all buses in MV network and underlying LV grids.

        Parameters
        ----------
        buses_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in MV network and underlying LV grids.
            Index of the dataframe are bus names. Columns of the dataframe are:
            v_nom
            x
            y
            mv_grid_id
            lv_grid_id
            in_building

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in MV network and underlying LV grids.

        """
        return self._buses_df

    @buses_df.setter
    def buses_df(self, buses_df):
        self._buses_df = buses_df

    @property
    def slack_df(self):
        slack_bus = self.transformers_hvmv_df.bus1.iloc[0]
        return pd.DataFrame(
            {
                "bus": [slack_bus],
                "control": ["Slack"],
                "p_nom": [0],
                "name": ["Generator_slack"],
            }
        ).set_index("name")

    @property
    def generators_df(self):
        """
        Dataframe with all generators in MV network and underlying LV grids.

        Parameters
        ----------
        generators_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all generators in MV network and underlying LV grids.
            Index of the dataframe are generator names. Columns of the
            dataframe are:
            bus
            control
            p_nom
            type
            weather_cell_id
            subtype

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all generators in MV network and underlying LV
            grids.

        """
        return self._generators_df

    @generators_df.setter
    def generators_df(self, generators_df):
        self._generators_df = generators_df

    @property
    def loads_df(self):
        """
        Dataframe with all loads in MV network and underlying LV grids.

        Parameters
        ----------
        loads_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all loads in MV network and underlying LV grids.
            Index of the dataframe are load names. Columns of the
            dataframe are:
            bus
            peak_load
            sector
            annual_consumption

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all loads in MV network and underlying LV grids.

        """
        return self._loads_df

    @loads_df.setter
    def loads_df(self, loads_df):
        self._loads_df = loads_df

    @property
    def transformers_df(self):
        """
        Dataframe with all transformers.

        Parameters
        ----------
        transformers_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all transformers.
            Index of the dataframe are transformer names. Columns of the
            dataframe are:
            bus0 - primary side
            bus1 - secondary side
            x_pu
            r_pu
            s_nom
            type_info

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all transformers.

        """
        return self._transformers_df

    @transformers_df.setter
    def transformers_df(self, transformers_df):
        self._transformers_df = transformers_df

    @property
    def transformers_hvmv_df(self):
        """
        Dataframe with all HVMV transformers.

        Parameters
        ----------
        transformers_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all transformers.
            Index of the dataframe are transformer names. Columns of the
            dataframe are:
            bus0
            bus1
            x_pu
            r_pu
            s_nom
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all HVMV transformers.

        """
        return self._transformers_hvmv_df

    @transformers_hvmv_df.setter
    def transformers_hvmv_df(self, transformers_hvmv_df):
        self._transformers_hvmv_df = transformers_hvmv_df

    @property
    def lines_df(self):
        """
        Dataframe with all lines in MV network and underlying LV grids.

        Parameters
        ----------
        lines_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all lines in MV network and underlying LV grids.
            Index of the dataframe are line names. Columns of the
            dataframe are:

            * bus0:
              name of first bus line is attached to
            * bus1:
              name of second bus line is attached to
            * length:
              line length in m
            * x:
              reactance of line (or in case of multiple parallel lines
              total reactance of lines) in Ohm
            * r:
              resistance of line (or in case of multiple parallel lines
              total resistance of lines) in Ohm
            * s_nom:
              apparent power which can pass through the
              line (or in case of multiple parallel lines total apparent
              power which can pass through the lines) in MVA
            * num_parallel:
              number of parallel lines
            * type_info:
              contains type of line as e.g. given in equipment data
            * kind:
              specifies whether line is a cable or overhead line

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all lines in MV network and underlying LV grids.

        """
        return self._lines_df

    @lines_df.setter
    def lines_df(self, lines_df):
        self._lines_df = lines_df

    @property
    def switches_df(self):
        """
        Dataframe with all switches in MV network and underlying LV grids.

        Parameters
        ----------
        switches_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all switches in MV network and underlying LV grids.
            Index of the dataframe are switch names. Columns of the
            dataframe are:
            bus_open
            bus_closed
            branch
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all switches in MV network and underlying LV grids.

        """
        return self._switches_df

    @switches_df.setter
    def switches_df(self, switches_df):
        self._switches_df = switches_df

    @property
    def storage_units_df(self):
        """
        Dataframe with all storage units in MV grid and underlying LV grids.

        Parameters
        ----------
        storage_units_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all storage units in MV grid and underlying LV
            grids.
            Index of the dataframe are storage names. Columns of the
            dataframe are:
            bus
            control
            p_nom
            capacity
            efficiency_store
            efficiency_dispatch

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all storage units in MV network and underlying LV
            grids.

        """
        return self._storage_units_df

    @storage_units_df.setter
    def storage_units_df(self, storage_units_df):
        self._storage_units_df = storage_units_df

    @property
    def charging_points_df(self):
        """
        Dataframe with all charging points in MV grid and underlying LV grids.

        Parameters
        ----------
        charging_points_df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all charging points in MV grid and underlying LV
            grids.
            Index of the dataframe are charging point names. Columns of the
            dataframe are:

            * bus (str)

              Bus name of bus, charging point is connected to.

            * p_nom (float)

              Maximum charging power in MW.

            * use_case (str), optional

              Specifies if charging point is e.g. for charging at
              home, at work, in public, or public fast charging.

            * polygon (shapely Polygon), optional

              Polygon of the area, the charging point is located in, as
              provided by 'simbev' tool.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all charging points in MV network and underlying LV
            grids.

        """
        return self._charging_points_df

    @charging_points_df.setter
    def charging_points_df(self, charging_points_df):
        self._charging_points_df = charging_points_df

    # TODO: fix instantiation of Generator and Load objects (they require
    # edisgo_obj that topology does have as a parameter)
    # @property
    # def generators(self):
    #     """
    #     Connected generators within the network.
    #
    #     Returns
    #     -------
    #     list(:class:`~.network.components.Generator`)
    #         List of generators within the network.
    #
    #     """
    #     for gen in self.generators_df.drop(labels=['Generator_slack']).index:
    #         yield Generator(id=gen)
    #
    # @property
    # def loads(self):
    #     """
    #     Connected loads within the network.
    #
    #     Returns
    #     -------
    #     list(:class:`~.network.components.Load`)
    #         List of loads within the network.
    #
    #     """
    #     for l in self.loads_df.index:
    #         yield Load(id=l)

    @property
    def id(self):
        """
        MV network ID

        Returns
        --------
        :obj:`str`
            MV network ID

        """

        return self.mv_grid.id

    @property
    def generator_scenario(self):
        """
        Defines which scenario of future generator park to use.

        Parameters
        ----------
        generator_scenario_name : :obj:`str`
            Name of scenario of future generator park

        Returns
        --------
        :obj:`str`
            Name of scenario of future generator park

        """
        return self._generator_scenario

    @generator_scenario.setter
    def generator_scenario(self, generator_scenario_name):
        self._generator_scenario = generator_scenario_name

    @property
    def mv_grid(self):
        """
        Medium voltage (MV) network

        Parameters
        ----------
        mv_grid : :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        Returns
        --------
        :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def grid_district(self):
        """
        Dictionary with MV grid district information.

        The dictionary contains the following information:

        * 'population'
          Number of inhabitants in grid district as integer.
        * 'geom'
          Geometry of MV grid district as (Multi)Polygon.
        * 'srid'
          SRID of grid district geometry.

        Parameters
        ----------
        grid_district : dict
            Dictionary with MV grid district information.

        Returns
        --------
        dict
            Dictionary with MV grid district information.

        """
        return self._grid_district

    @grid_district.setter
    def grid_district(self, grid_district):
        self._grid_district = grid_district

    def get_connected_lines_from_bus(self, bus_name):
        """
        Returns all lines connected to bus of name bus_name.

        Parameters
        ----------
        bus_name : str
            name of bus

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe of connected lines

        """
        return self.lines_df.loc[self.lines_df.bus0 == bus_name].append(
            self.lines_df.loc[self.lines_df.bus1 == bus_name]
        )

    def get_connected_components_from_bus(self, bus_name):
        """
        Returns dict of connected elements to bus of provided bus_name.

        Parameters
        ----------
        bus_name: str
            representative of bus

        Returns
        -------
         dict of :pandas:`pandas.DataFrame<DataFrame>`
            dictionary of connected elements with keys 'Generator', 'Line',
            'Load', 'Transformer', 'Transformer_HVMV', 'StorageUnit', 'Switch'
        """
        components = {}
        components["Generator"] = self.generators_df.loc[
            self.generators_df.bus == bus_name
        ]
        components["Line"] = self.get_connected_lines_from_bus(bus_name)
        components["Load"] = self.loads_df.loc[self.loads_df.bus == bus_name]
        components["Transformer"] = self.transformers_df.loc[
            self.transformers_df.bus0 == bus_name
        ].append(
            self.transformers_df.loc[self.transformers_df.bus1 == bus_name]
        )
        components["Transformer_HVMV"] = self.transformers_hvmv_df.loc[
            self.transformers_hvmv_df.bus0 == bus_name
        ].append(
            self.transformers_hvmv_df.loc[
                self.transformers_hvmv_df.bus1 == bus_name
            ]
        )
        components["StorageUnit"] = self.storage_units_df.loc[
            self.storage_units_df.bus == bus_name
        ]
        components["Switch"] = self.switches_df.loc[
            self.switches_df.bus_open == bus_name
        ]
        return components

    def get_neighbours(self, bus_name):
        """
        Returns all neighbour buses of bus with bus_name.

        Parameters
        ----------
        bus_name : str
            name of bus

        Returns
        --------
        list(str)

        """
        lines = self.get_connected_lines_from_bus(bus_name)
        buses = list(lines.bus0)
        buses.extend(list(lines.bus1))
        neighbours = set(buses)
        neighbours.remove(bus_name)
        return neighbours

    def remove_bus(self, name):
        """
        Removes bus with given name from topology.

        Parameters
        ----------
        name : str
            Name of bus as specified in index of `buses_df`.

        Notes
        -------
        Only isolated buses can be deleted from topology. Use respective
        functions first to delete all connected lines, transformers,
        loads, generators and storage_units.
        """

        # check if bus is isolated
        if (
            name in self.lines_df.bus0.values
            or name in self.lines_df.bus1.values
            or name in self.storage_units_df.bus.values
            or name in self.generators_df.bus.values
            or name in self.loads_df.bus.values
            or name in self.transformers_hvmv_df.bus0.values
            or name in self.transformers_hvmv_df.bus1.values
            or name in self.transformers_df.bus0.values
            or name in self.transformers_df.bus1.values
        ):
            raise AssertionError(
                "Bus {} is not isolated. Remove all connected "
                "elements first to remove bus.".format(name)
            )
        else:
            self._buses_df.drop(name, inplace=True)

    def remove_generator(self, name):
        """
        Removes generator with given name from topology.

        Parameters
        ----------
        name : str
            Name of generator as specified in index of `generators_df`.

        """

        # get bus to check if other elements are connected to bus
        bus = self.generators_df.at[name, "bus"]
        # remove generator
        self._generators_df.drop(name, inplace=True)
        # ToDo drop timeseries
        # if no other elements are connected to same bus, remove line and bus
        if check_bus_for_removal(self, bus_name=bus):
            line_name = self.get_connected_lines_from_bus(bus).index[0]
            self.remove_line(line_name)
            logger.debug(
                "Line {} removed together with generator {}.".format(
                    line_name, name
                )
            )

    def remove_load(self, name):
        """
        Removes load with given name from topology.

        Parameters
        ----------
        name : str
            Name of load as specified in index of `loads_df`.

        """

        # get bus to check if other elements are connected to bus
        bus = self.loads_df.at[name, "bus"]
        # remove load
        self._loads_df.drop(name, inplace=True)
        # if no other elements are connected, remove line and bus as well
        if check_bus_for_removal(self, bus_name=bus):
            line_name = self.get_connected_lines_from_bus(bus).index[0]
            self.remove_line(line_name)
            logger.debug(
                "Line {} removed together with load {}.".format(
                    line_name, name
                )
            )

    def remove_storage(self, name):
        """
        Removes storage with given name from topology.

        Parameters
        ----------
        name : str
            Name of storage as specified in index of `storage_units_df`.

        """
        # get bus to check if other elements are connected to bus
        bus = self.storage_units_df.at[name, "bus"]
        # remove storage unit
        self._storage_units_df.drop(name, inplace=True)
        # if no other elements are connected, remove line and bus as well
        if check_bus_for_removal(self, bus_name=bus):
            line_name = self.get_connected_lines_from_bus(bus).index[0]
            self.remove_line(line_name)
            logger.debug(
                "Line {} removed together with storage unit {}.".format(
                    line_name, name
                )
            )

    def remove_charging_point(self, name):
        """
        Removes charging point with given name from topology.

        Parameters
        ----------
        name : str
            Name of charging point as specified in index of `storage_units_df`.

        """
        # get bus to check if other elements are connected to it
        bus = self.charging_points_df.at[name, "bus"]
        # remove charging point
        self._charging_points_df.drop(name, inplace=True)
        # if no other elements are connected, remove line and bus as well
        if check_bus_for_removal(self, bus_name=bus):
            line_name = self.get_connected_lines_from_bus(bus).index[0]
            self.remove_line(line_name)
            logger.debug(
                "Line {} removed together with charging point {}.".format(
                    line_name, name
                )
            )

    def remove_line(self, name):
        """
        Removes line with given name from topology.

        Parameters
        ----------
        name : str
            Name of line as specified in index of `lines_df`.

        """
        if not check_line_for_removal(self, line_name=name):
            raise AssertionError(
                "Removal of line {} would create isolated "
                "node.".format(name)
            )

        # backup buses of line and check if buses can be removed as well
        bus0 = self.lines_df.at[name, "bus0"]
        remove_bus0 = check_bus_for_removal(self, bus0)
        bus1 = self.lines_df.at[name, "bus1"]
        remove_bus1 = check_bus_for_removal(self, bus1)

        # drop line
        self._lines_df.drop(name, inplace=True)

        # drop buses if no other elements are connected
        if remove_bus0:
            self.remove_bus(bus0)
            logger.debug(
                "Bus {} removed together with line {}".format(bus0, name)
            )
        if remove_bus1:
            self.remove_bus(bus1)
            logger.debug(
                "Bus {} removed together with line {}".format(bus1, name)
            )

    def add_generator(
        self, generator_id, bus, p_nom, generator_type, **kwargs
    ):
        """
        Adds generator to topology.

        Generator name is generated automatically.

        Parameters
        ----------
        generator_id : str
            Unique identifier of generator.
        bus: str
            Identifier of connected bus
        control: str
            Control type of generator. Defaults to 'PQ'.
        p_nom: float
            Nominal power [MW]
        generator_type: str
            Type of generator, e.g. 'solar' or 'gas'
        weather_cell_id: int
            ID of weather cell, required for fluctuating generators import from
            oedb.
        subtype: str
            Further specification of type, e.g. 'solar_roof_mounted'

        """
        # ToDo add test
        # check if bus exists
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        if not np.isnan(bus_df.lv_grid_id) and bus_df.lv_grid_id is not None:
            grid_name = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            grid_name = "MVGrid_" + str(int(bus_df.mv_grid_id))

        # generate generator name and check uniqueness
        generator_name = "Generator_{}_{}_{}".format(
            generator_type, grid_name, generator_id
        )
        while generator_name in self.generators_df.index:
            random.seed(a=generator_name)
            generator_name = "Generator_{}_{}".format(
                generator_type, random.randint(10 ** 8, 10 ** 9)
            )

        # unpack optional parameters
        weather_cell_id = kwargs.get("weather_cell_id", None)
        subtype = kwargs.get("subtype", None)
        control = kwargs.get("control", None)

        # create new generator dataframe
        new_gen_df = pd.DataFrame(
            data={
                "bus": bus,
                "p_nom": p_nom,
                "control": control if control is not None else "PQ",
                "type": generator_type,
                "weather_cell_id": weather_cell_id,
                "subtype": subtype,
            },
            index=[generator_name],
        )
        self.generators_df = self._generators_df.append(new_gen_df)
        return generator_name

    def add_load(self, load_id, bus, peak_load, annual_consumption, sector):
        """
        Adds load to topology.

        Load name is generated automatically.

        Parameters
        ----------
        load_id : str
            Unique identifier of generator.
        bus: str
            identifier of connected bus
        peak_load: float
            peak load in [MVA]
        annual_consumption: float
            annual consumption in Todo: specify unit?
        sector: str
            can be 'agricultural', 'industrial', 'residential' or 'retail'
        """
        # Todo: overthink load_id as input parameter, only allow auto created
        #  names?
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        # generate load name and check uniqueness
        if not np.isnan(bus_df.lv_grid_id) and bus_df.lv_grid_id is not None:
            grid_name = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            grid_name = "MVGrid_" + str(int(bus_df.mv_grid_id))
        load_name = "Load_{}_{}_{}".format(sector, grid_name, load_id)
        if load_name in self.loads_df.index:
            nr_loads = len(self._grids[grid_name].loads_df)
            load_name = "Load_{}_{}_{}".format(sector, grid_name, nr_loads + 1)
            while load_name in self.loads_df.index:
                load_name = "Load_{}_{}_{}".format(
                    sector, grid_name, random.randint(10 ** 8, 10 ** 9)
                )

        new_load_df = pd.DataFrame(
            data={
                "bus": bus,
                "peak_load": peak_load,
                "annual_consumption": annual_consumption,
                "sector": sector,
            },
            index=[load_name],
        )
        self.loads_df = self._loads_df.append(new_load_df)
        return load_name

    def add_storage_unit(self, bus, p_nom, control=None):
        """
        Adds storage unit to topology.

        Storage unit name is generated automatically.

        Parameters
        ----------
        bus: str
            Identifier of connected bus.
        p_nom: float
            Nominal power in MW
        control: str
            Control type, defaults to 'PQ'

        """
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        # generate storage name and check uniqueness
        if not np.isnan(bus_df.lv_grid_id) and bus_df.lv_grid_id is not None:
            grid_name = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            grid_name = "MVGrid_" + str(int(bus_df.mv_grid_id))
        storage_id = len(self._grids[grid_name].storage_units_df)
        storage_name = "StorageUnit_{}_{}".format(grid_name, storage_id)
        if storage_name in self.storage_units_df.index:
            storage_name = "StorageUnit_{}_{}".format(
                grid_name, storage_id + 1
            )
            while storage_name in self.storage_units_df.index:
                storage_name = "StorageUnit_{}_{}".format(
                    grid_name, random.randint(10 ** 8, 10 ** 9)
                )

        new_storage_df = pd.DataFrame(
            data={
                "bus": bus,
                "p_nom": p_nom,
                "control": control if control is not None else "PQ",
            },
            index=[storage_name],
        )
        self.storage_units_df = self._storage_units_df.append(new_storage_df)
        return storage_name

    def add_charging_point(self, bus, p_nom, use_case=None, polygon=None):
        """
        Adds charging point to topology.

        Charging point identifier is generated automatically.

        Parameters
        ----------
        bus : str
            Identifier of bus charging point is connected to.
        p_nom : float
            Nominal power in MW
        use_case : str (optional)
            Specifies if charging point is e.g. used for charging at
            home, at work, in public, or public fast charging.
        polygon : shapely Polygon (optional)
            Polygon of the area, the charging point is located in, as
            provided by 'simbev' tool.

        """
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        # generate charging point identifier and check uniqueness
        if not np.isnan(bus_df.lv_grid_id) and bus_df.lv_grid_id is not None:
            grid_name = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            grid_name = "MVGrid_" + str(int(bus_df.mv_grid_id))
        id = len(self._grids[grid_name].charging_points_df)
        name = "ChargingPoint_{}_{}".format(grid_name, id)
        if name in self.charging_points_df.index:
            name = "ChargingPoint_{}_{}".format(
                grid_name, id + 1
            )
            while name in self.charging_points_df.index:
                name = "ChargingPoint_{}_{}".format(
                    grid_name, random.randint(10 ** 8, 10 ** 9)
                )

        new_df = pd.DataFrame(
            data={
                "bus": bus,
                "p_nom": p_nom,
                "use_case": use_case,
                "polygon": polygon
            },
            index=[name],
        )
        self.charging_points_df = self._charging_points_df.append(new_df)
        return name

    def add_bus(self, bus_name, v_nom, **kwargs):
        """
        Adds new bus to topology.

        Parameters
        ----------
        bus_name : str
            representative of bus
        v_nom: float
            nominal voltage at bus [kV]
        x: float
            position (e.g. longitude); the Spatial Reference System Identifier
            (SRID) is saved in the network dataframe
        y: float
            position (e.g. longitude); the Spatial Reference System Identifier
            (SRID) is saved in the network dataframe
        lv_grid_id: int
            identifier of LVGrid, None if bus is MV component
        in_building: bool
            indicator if bus is inside a building

        """
        # ToDo make sure bus_name is unique!
        x = kwargs.get("x", None)
        y = kwargs.get("y", None)
        lv_grid_id = kwargs.get("lv_grid_id", None)
        in_building = kwargs.get("in_building", False)
        # check lv_grid_id
        if v_nom < 1 and lv_grid_id is None:
            raise ValueError(
                "You need to specify an lv_grid_id for low-voltage buses."
            )
        new_bus_df = pd.DataFrame(
            data={
                "v_nom": v_nom,
                "x": x,
                "y": y,
                "mv_grid_id": self.mv_grid.id,
                "lv_grid_id": lv_grid_id,
                "in_building": in_building,
            },
            index=[bus_name],
        )
        self._buses_df = self._buses_df.append(new_bus_df)

    def add_line(self, bus0, bus1, length, **kwargs):
        """
        Adds new line to topology.

        Line name is generated automatically.
        If type_info is provided, x, r and s_nom are calculated.

        Parameters
        ----------
        bus0: str
            identifier of connected bus
        bus1: str
            identifier of connected bus
        length: float
            length of line in [km]
        x: float
            reactance of line [Ohm]
        r: float
            resistance of line [Ohm]
        s_nom: float
            nominal power of line [MVA]
        num_parallel: int
            number of parallel lines
        type_info : str
            Type of line as specified in `equipment_data`.
        kind: str
            either 'cable' or 'line'

        """

        def _get_line_data():
            """
            Gets line data for line type specified in `line_type` from
            equipment data.

            Returns
            --------
            pd.Series
                Line data from equipment_data

            """
            if self.buses_df.loc[bus0, "v_nom"] < 1:
                voltage_level = "lv"
            else:
                voltage_level = "mv"

            # try to get cable data
            try:
                line_data = self.equipment_data[
                    "{}_cables".format(voltage_level)
                ].loc[type_info, :]
            except KeyError:
                try:
                    line_data = self.equipment_data[
                        "{}_overhead_lines".format(voltage_level)
                    ].loc[type_info, :]
                except:
                    raise ValueError("Specified line type is not valid.")
            except:
                raise
            return line_data

        # check if buses exist
        if bus0 not in self.buses_df.index:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus0)
            )
        if bus1 not in self.buses_df.index:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus1)
            )

        # check if line between given buses already exists
        bus0_bus1 = self.lines_df[
            (self.lines_df.bus0 == bus0) & (self.lines_df.bus1 == bus1)
        ]
        bus1_bus0 = self.lines_df[
            (self.lines_df.bus1 == bus0) & (self.lines_df.bus0 == bus1)
        ]
        if not bus0_bus1.empty and bus1_bus0.empty:
            logging.debug("Line between bus0 {} and bus1 {} already exists.")
            return bus1_bus0.append(bus0_bus1).index[0]

        # unpack optional parameters
        x = kwargs.get("x", None)
        r = kwargs.get("r", None)
        s_nom = kwargs.get("s_nom", None)
        num_parallel = kwargs.get("num_parallel", 1)
        type_info = kwargs.get("type_info", None)
        kind = kwargs.get("kind", None)

        # if type of line is specified calculate x, r and s_nom
        if type_info is not None:
            if x is not None or r is not None or s_nom is not None:
                warnings.warn(
                    "When line 'type_info' is provided when creating a new "
                    "line, x, r and s_nom are calculated and provided "
                    "parameters are overwritten."
                )
            line_data = _get_line_data()
            if isinstance(line_data, pd.DataFrame) and len(line_data) > 1:
                line_data = (
                    line_data[
                        line_data.U_n == self.buses_df.loc[bus0, "v_nom"]
                    ]
                ).iloc[0, :]
            x = calculate_line_resistance(line_data.L_per_km, length)
            r = calculate_line_reactance(line_data.R_per_km, length)
            s_nom = calculate_apparent_power(line_data.U_n, line_data.I_max_th)

        # generate line name and check uniqueness
        line_name = "Line_{}_{}".format(bus0, bus1)
        while line_name in self.lines_df.index:
            line_name = "Line_{}_{}_{}".format(
                bus0, bus1, random.randint(10 ** 8, 10 ** 9)
            )

        # ToDo
        # # calculate r if not provided
        # if x is None and type_info:
        new_line_df = pd.DataFrame(
            data={
                "bus0": bus0,
                "bus1": bus1,
                "x": x,
                "r": r,
                "length": length,
                "type_info": type_info,
                "num_parallel": num_parallel,
                "kind": kind,
                "s_nom": s_nom,
            },
            index=[line_name],
        )
        self._lines_df = self._lines_df.append(new_line_df)
        return line_name

    def update_number_of_parallel_lines(self, lines_num_parallel):
        """
        Changes number of parallel lines and updates line attributes.

        When number of parallel lines changes, attributes x, r, and s_nom have
        to be adapted, which is done in this function.

        Parameters
        ------------
        lines_num_parallel : :pandas:`pandas.Series<series>`
            index contains names of lines to update and values of series
            contain corresponding new number of parallel lines.

        """
        # update x, r and s_nom
        self._lines_df.loc[lines_num_parallel.index, "x"] = (
                self._lines_df.loc[lines_num_parallel.index, "x"]
                * self._lines_df.loc[lines_num_parallel.index, "num_parallel"]
                / lines_num_parallel
        )
        self._lines_df.loc[lines_num_parallel.index, "r"] = (
                self._lines_df.loc[lines_num_parallel.index, "r"]
                * self._lines_df.loc[lines_num_parallel.index, "num_parallel"]
                / lines_num_parallel
        )
        self._lines_df.loc[lines_num_parallel.index, "s_nom"] = (
                self._lines_df.loc[lines_num_parallel.index, "s_nom"]
                / self._lines_df.loc[lines_num_parallel.index, "num_parallel"]
                * lines_num_parallel
        )

        # update number parallel lines
        self._lines_df.loc[
            lines_num_parallel.index, "num_parallel"
        ] = lines_num_parallel

    def change_line_type(self, lines, new_line_type):
        """
        Changes line type of specified lines to given new line type.

        Be aware that this function replaces the lines by one line of the
        given line type.
        Lines must all be in the same voltage level and the new line type
        must be a cable with technical parameters given in equipment
        parameters.

        Parameters
        ----------
        lines : list(str)
            List of line names of lines to be changed to new line type.
        new_line_type : str
            Specifies new line type of lines. Line type must be a cable with
            technical parameters given in "mv_cables" or "lv_cables" equipment
            data.

        """

        try:
            data_new_line = self.equipment_data[
                "lv_cables"
            ].loc[new_line_type]
        except KeyError:
            try:
                data_new_line = self.equipment_data[
                    "mv_cables"
                ].loc[new_line_type]
                # in case of MV cable adapt nominal voltage to MV voltage
                grid_voltage = self.buses_df.at[
                    self.lines_df.at[lines[0], "bus0"], "v_nom"]
                if grid_voltage != data_new_line.U_n:
                    logging.debug(
                        "The line type of lines {} is changed to a type with "
                        "a different nominal voltage (nominal voltage of new "
                        "line type is {} kV while nominal voltage of the "
                        "medium voltage grid is {} kV). The nominal voltage "
                        "of the new line type is therefore set to the grids "
                        "nominal voltage.".format(
                            lines, data_new_line.U_n, grid_voltage))
                    data_new_line.U_n = grid_voltage
            except KeyError:
                raise KeyError(
                    "Given new line type is not in equipment data. Please "
                    "make sure to use line type with technical data provided "
                    "in equipment_data 'mv_cables' or 'lv_cables'.")

        self._lines_df.loc[lines, "type_info"] = data_new_line.name
        self._lines_df.loc[lines, "num_parallel"] = 1
        self._lines_df.loc[lines, "kind"] = "cable"

        self._lines_df.loc[lines, "r"] = (
            data_new_line.R_per_km * self.lines_df.loc[lines, "length"]
        )
        self._lines_df.loc[lines, "x"] = (
            data_new_line.L_per_km * 2 * np.pi * 50 / 1e3
            * self.lines_df.loc[lines, "length"]
        )
        self._lines_df.loc[lines, "s_nom"] = (
            np.sqrt(3) * data_new_line.U_n * data_new_line.I_max_th
        )

    def to_csv(self, topology_dir):
        """
        Exports topology to csv files with names buses, generators, lines,
        loads, switches, transformers, transformers_hvmv, network. Files are
        designed in a way that they can be directly imported to pypsa. A sub-
        folder named "topology" is added to the provided directory.

        Parameters
        ----------
        topology_dir: str
            path to save topology to

        """
        os.makedirs(topology_dir, exist_ok=True)
        self._buses_df.to_csv(os.path.join(topology_dir, "buses.csv"))
        self._generators_df.append(self.slack_df).to_csv(
            os.path.join(topology_dir, "generators.csv")
        )
        self._lines_df.to_csv(os.path.join(topology_dir, "lines.csv"))
        self._loads_df.to_csv(os.path.join(topology_dir, "loads.csv"))
        self._charging_points_df.to_csv(
            os.path.join(topology_dir, "charging_points.csv"))
        self._storage_units_df.to_csv(
            os.path.join(topology_dir, "storage_units.csv")
        )
        self._switches_df.to_csv(os.path.join(topology_dir, "switches.csv"))
        self._transformers_df.rename(
            {"x_pu": "x", "r_pu": "r"}, axis=1
        ).to_csv(os.path.join(topology_dir, "transformers.csv"))
        self._transformers_hvmv_df.rename(
            {"x_pu": "x", "r_pu": "r"}, axis=1
        ).to_csv(os.path.join(topology_dir, "transformers_hvmv.csv"))
        network = {"name": self.mv_grid.id}
        network.update(self._grid_district)
        pd.DataFrame([network]).set_index("name").rename(
            {
                "geom": "mv_grid_district_geom",
                "population": "mv_grid_district_population",
            },
            axis=1,
        ).to_csv(os.path.join(topology_dir, "network.csv"))
        logger.debug("Topology exported.")

    def from_csv(self, topology_dir, edisgo_obj):
        """
        # Todo: when done with project, reset_index in save function and remove renaming here
        # Todo: should this
        Exports topology to csv files with names buses, generators, lines,
        loads, switches, transformers, transformers_hvmv, network. Files are
        designed in a way that they can be directly imported to pypsa. A sub-
        folder named "topology" is added to the provided directory.

        Parameters
        ----------
        topology_dir: str
            path to save topology to

        """
        # import buses and lines
        self.buses_df = pd.read_csv(os.path.join(topology_dir, "buses.csv")).\
            rename(columns={'Unnamed: 0': 'name'}).set_index('name')
        self.lines_df = pd.read_csv(os.path.join(topology_dir, "lines.csv")).\
            rename(columns={'Unnamed: 0': 'name'}).set_index('name')
        # import mvlv transformers
        if os.path.exists(os.path.join(topology_dir, "transformers.csv")):
            self.transformers_df = \
                pd.read_csv(os.path.join(topology_dir, "transformers.csv")). \
                    rename(columns={'Unnamed: 0': 'name', "x": "x_pu",
                                    "r": "r_pu"}).set_index('name')
        # import hvmv transformers
        if os.path.exists(os.path.join(topology_dir, "transformers_hvmv.csv")):
            self.transformers_hvmv_df = \
                pd.read_csv(
                    os.path.join(topology_dir, "transformers_hvmv.csv")). \
                    rename(columns={'Unnamed: 0': 'name', "x": "x_pu",
                                    "r": "r_pu"}).set_index('name')
        # import generators
        if os.path.exists(os.path.join(topology_dir, "generators.csv")):
            all_generators_df =\
                pd.read_csv(os.path.join(topology_dir, "generators.csv")).\
                rename(columns={'Unnamed: 0': 'name'}).set_index('name')
            if hasattr(self, '_transformers_hvmv_df'):
                # remove slack
                self.generators_df = all_generators_df.drop(
                    all_generators_df.loc[all_generators_df.control == 'Slack'].
                        index)
            else:
                self.generators_df = all_generators_df
            # self._slack_df = all_generators_df.loc[
            #     all_generators_df.index.str.contains('slack')]
            # if len(self.slack_df) != 1:
            #     logging.warning('Slack could not be imported. '
            #                     'Please set one manually.')
            self.generators_df = all_generators_df.drop(self.slack_df.index)
        # import loads
        if os.path.exists(os.path.join(topology_dir, "loads.csv")):
            self.loads_df = \
                pd.read_csv(os.path.join(topology_dir, "loads.csv")).\
                rename(columns={'Unnamed: 0': 'name'}).set_index('name')
        # import charging points
        if os.path.exists(os.path.join(topology_dir, "charging_points.csv")):
            self.charging_points_df = \
                pd.read_csv(os.path.join(topology_dir, "charging_points.csv")). \
                rename(columns={'Unnamed: 0': 'name'}).set_index('name')
        # import storage units
        if os.path.exists(os.path.join(topology_dir, "storage_units.csv")):
            self.storage_units_df = \
                pd.read_csv(os.path.join(topology_dir, "storage_units.csv")). \
                rename(columns={'Unnamed: 0': 'name'}).set_index('name')
        # import switches
        if os.path.exists(os.path.join(topology_dir, "switches.csv")):
            self.switches_df = \
                pd.read_csv(os.path.join(topology_dir, "switches.csv")). \
                rename(columns={'Unnamed: 0': 'name'}).set_index('name')

        # import network data
        network = pd.read_csv(os.path.join(topology_dir, "network.csv")).\
            rename(columns={
                "mv_grid_district_geom": "geom",
                "mv_grid_district_population": "population",
            })
        self.mv_grid = MVGrid(edisgo_obj=edisgo_obj, id=network['name'].values[0])
        # set up medium voltage grid
        self._grids = {}
        self._grids[
            str(self.mv_grid)
        ] = self.mv_grid

        # set up low voltage grids
        lv_grid_ids = set(self.buses_df.lv_grid_id.dropna())
        for lv_grid_id in lv_grid_ids:
            lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_obj)
            self.mv_grid._lv_grids.append(lv_grid)
            self._grids[str(lv_grid)] = lv_grid

        # Check data integrity
        _validate_ding0_grid_import(edisgo_obj.topology)
        # set grid district

        self.grid_district = network.loc[0].drop('name').to_dict()
        logger.debug("Topology imported.")

    def __repr__(self):
        return "Network topology " + str(self.id)
