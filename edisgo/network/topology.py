import logging
import random
import pandas as pd
import numpy as np
import os
import warnings
import networkx as nx

import edisgo
from edisgo.network.grids import MVGrid, LVGrid
from edisgo.network.components import Switch
from edisgo.tools.tools import (
    calculate_line_resistance,
    calculate_line_reactance,
    calculate_apparent_power,
    select_cable
)
from edisgo.tools import networkx_helper
from edisgo.tools import geo
from edisgo.io.ding0_import import _validate_ding0_grid_import

if "READTHEDOCS" not in os.environ:
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import Point, LineString
    from shapely.ops import transform

logger = logging.getLogger("edisgo")

COLUMNS = {
    "loads_df": [
        "bus", "peak_load", "annual_consumption", "sector"],
    "generators_df": [
            "bus", "p_nom", "type", "control", "weather_cell_id", "subtype"],
    "charging_points_df": ["bus", "p_nom", "use_case"],
    "storage_units_df": ["bus", "control", "p_nom"],
    "transformers_df": ["bus0", "bus1", "x_pu", "r_pu", "s_nom", "type_info"],
    "lines_df": [
        "bus0", "bus1", "length", "x", "r", "s_nom", "num_parallel",
        "type_info", "kind",

    ],
    "buses_df": ["v_nom", "x", "y", "mv_grid_id", "lv_grid_id", "in_building"],
    "switches_df": ["bus_open", "bus_closed", "branch", "type_info"]
}


class Topology:
    """
    Container for all grid topology data of a single MV grid.

    Data may as well include grid topology data of underlying LV grids.

    Other Parameters
    -----------------
    config : None or :class:`~.tools.config.Config`
        Provide your configurations if you want to load self-provided equipment
        data. Path to csv files containing the technical data is set in
        `config_system.cfg` in sections `system_dirs` and `equipment`.
        The default is None in which case the equipment data provided by
        eDisGo is used.

    Attributes
    -----------
    _grids : dict
        Dictionary containing all grids (keys are grid representatives and
        values the grid objects)

    """

    def __init__(self, **kwargs):

        # load technical data of equipment
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
        dict
            Dictionary with :pandas:`pandas.DataFrame<DataFrame>` containing
            equipment data. Keys of the dictionary are 'mv_transformers',
            'mv_overhead_lines', 'mv_cables', 'lv_transformers', and
            'lv_cables'.

        Notes
        ------
        This function calculates electrical values of transformers from
        standard values (so far only for MV/LV transformers, not necessary for
        HV/MV transformers as MV impedances are not used).

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
    def loads_df(self):
        """
        Dataframe with all loads in MV network and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all loads in MV network and underlying LV grids.
            Index of the dataframe are load names as string. Columns of the
            dataframe are:

            bus : str
                Identifier of bus load is connected to.

            peak_load : float
                Peak load in MW.

            annual_consumption : float
                Annual consumption in MWh.

            sector : str
                Specifies type of load. If demandlib is used to generate
                sector-specific time series, the sector needs to either be
                'agricultural', 'industrial', 'residential' or 'retail'.
                Otherwise sector can be chosen freely.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all loads in MV network and underlying LV grids.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._loads_df
        except:
            return pd.DataFrame(columns=COLUMNS["loads_df"])

    @loads_df.setter
    def loads_df(self, df):
        self._loads_df = df

    @property
    def generators_df(self):
        """
        Dataframe with all generators in MV network and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all generators in MV network and underlying LV
            grids. Index of the dataframe are generator names as string.
            Columns of the dataframe are:

            bus : str
                Identifier of bus generator is connected to.

            p_nom : float
                Nominal power in MW.

            type : str
                Type of generator, e.g. 'solar', 'run_of_river', etc. Is used
                in case generator type specific time series are provided.

            control : str
                Control type of generator used for power flow analysis. In MV
                and LV grids usually 'PQ'.

            weather_cell_id : int
                ID of weather cell, that identifies the weather data cell from
                the weather data set used in the research project
                `open_eGo <https://openegoproject.wordpress.com/>`_ to
                determine feed-in profiles of wind and solar generators.
                Only required when time series of  wind and solar generators
                are assigned using precalculated time series from the
                OpenEnergy DataBase.

            subtype : str
                Further specification of type, e.g. 'solar_roof_mounted'.
                Currently not required for any functionality.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all generators in MV network and underlying LV
            grids. For more information on the dataframe see input parameter
            `df`.

        """
        try:
            return self._generators_df
        except:
            return pd.DataFrame(columns=COLUMNS["generators_df"])

    @generators_df.setter
    def generators_df(self, df):
        self._generators_df = df

    @property
    def charging_points_df(self):
        """
        Dataframe with all charging points in MV grid and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all charging points in MV grid and underlying LV
            grids. Index of the dataframe are charging point names as string.
            Columns of the dataframe are:

            bus : str
              Identifier of bus charging point is connected to.

            p_nom : float
                Maximum charging power in MW.

            use_case : str
              Specifies if charging point is e.g. for charging at
              home, at work, in public, or public fast charging. Used in
              charging point integration (:attr:`~.EDisGo.integrate_component`)
              to determine possible grid connection points, in which case use
              cases 'home', 'work', 'public', and 'fast' are distinguished.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all charging points in MV network and underlying LV
            grids. For more information on the dataframe see input parameter
            `df`.

        """
        try:
            return self._charging_points_df
        except:
            return pd.DataFrame(columns=COLUMNS["charging_points_df"])

    @charging_points_df.setter
    def charging_points_df(self, df):
        self._charging_points_df = df

    @property
    def storage_units_df(self):
        """
        Dataframe with all storage units in MV grid and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all storage units in MV grid and underlying LV
            grids. Index of the dataframe are storage names as string. Columns
            of the dataframe are:

            bus : str
                Identifier of bus storage unit is connected to.

            control : str
                Control type of storage unit used for power flow analysis,
                usually 'PQ'.

            p_nom : float
                Nominal power in MW.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all storage units in MV network and underlying LV
            grids. For more information on the dataframe see input parameter
            `df`.

        """
        try:
            return self._storage_units_df
        except:
            return pd.DataFrame(columns=COLUMNS["storage_units_df"])

    @storage_units_df.setter
    def storage_units_df(self, df):
        self._storage_units_df = df

    @property
    def transformers_df(self):
        """
        Dataframe with all MV/LV transformers.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all MV/LV transformers. Index of the dataframe are
            transformer names as string. Columns of the dataframe are:

            bus0 : str
                Identifier of bus at the transformer's primary (MV) side.

            bus1 : str
                Identifier of bus at the transformer's secondary (LV) side.

            x_pu : float
                Per unit series reactance.

            r_pu : float
                Per unit series resistance.

            s_nom : float
                Nominal apparent power in MW.

            type_info : str
                Type of transformer.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all MV/LV transformers. For more information on the
            dataframe see input parameter `df`.

        """
        try:
            return self._transformers_df
        except:
            return pd.DataFrame(columns=COLUMNS["transformers_df"])

    @transformers_df.setter
    def transformers_df(self, df):
        self._transformers_df = df

    @property
    def transformers_hvmv_df(self):
        """
        Dataframe with all HV/MV transformers.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all HV/MV transformers, with the same format as
            :py:attr:`~transformers_df`.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all HV/MV transformers. For more information on
            format see :py:attr:`~transformers_df`.

        """
        try:
            return self._transformers_hvmv_df
        except:
            return pd.DataFrame(columns=COLUMNS["transformers_df"])

    @transformers_hvmv_df.setter
    def transformers_hvmv_df(self, df):
        self._transformers_hvmv_df = df

    @property
    def lines_df(self):
        """
        Dataframe with all lines in MV network and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all lines in MV network and underlying LV grids.
            Index of the dataframe are line names as string. Columns of the
            dataframe are:

            bus0 : str
                Identifier of first bus to which line is attached.

            bus1 : str
                Identifier of second bus to which line is attached.

            length : float
                Line length in km.

            x : float
                Reactance of line (or in case of multiple parallel lines
                total reactance of lines) in Ohm.

            r : float
                Resistance of line (or in case of multiple parallel lines
                total resistance of lines) in Ohm.

            s_nom : float
                Apparent power which can pass through the line (or in case of
                multiple parallel lines total apparent power which can pass
                through the lines) in MVA.

            num_parallel : int
                Number of parallel lines.

            type_info : str
                Type of line as e.g. given in `equipment_data`.

            kind : str
                Specifies whether line is a cable ('cable') or overhead line
                ('line').

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all lines in MV network and underlying LV grids.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._lines_df
        except:
            return pd.DataFrame(columns=COLUMNS["lines_df"])

    @lines_df.setter
    def lines_df(self, df):
        self._lines_df = df

    @property
    def buses_df(self):
        """
        Dataframe with all buses in MV network and underlying LV grids.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in MV network and underlying LV grids.
            Index of the dataframe are bus names as strings. Columns of the
            dataframe are:

            v_nom : float
                Nominal voltage in kV.

            x : float
                x-coordinate (longitude) of geolocation.

            y : float
                y-coordinate (latitude) of geolocation.

            mv_grid_id : int
                ID of MV grid the bus is in.

            lv_grid_id : int
                ID of LV grid the bus is in. In case of MV buses this is NaN.

            in_building : bool
                Signifies whether a bus is inside a building, in which case
                only components belonging to this house connection can be
                connected to it.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all buses in MV network and underlying LV grids.

        """
        try:
            return self._buses_df
        except:
            return pd.DataFrame(columns=COLUMNS["buses_df"])

    @buses_df.setter
    def buses_df(self, df):
        # make sure in_building takes on only True or False (not numpy bools)
        # needs to be tested using `== True`, not `is True`
        buses_in_building = df[df.in_building == True].index
        df.loc[buses_in_building, "in_building"] = True
        df.loc[
            ~df.index.isin(buses_in_building), "in_building"] = False
        self._buses_df = df

    @property
    def switches_df(self):
        """
        Dataframe with all switches in MV network and underlying LV grids.

        Switches are implemented as branches that, when they are closed, are
        connected to a bus (`bus_closed`) such that there is a closed ring,
        and when they are open, connected to a virtual bus (`bus_open`), such
        that there is no closed ring. Once the ring is closed, the virtual
        is a single bus that is not connected to the rest of the grid.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all switches in MV network and underlying LV grids.
            Index of the dataframe are switch names as string. Columns of the
            dataframe are:

            bus_open : str
                Identifier of bus the switch branch is connected to when the
                switch is open.

            bus_closed : str
                Identifier of bus the switch branch is connected to when the
                switch is closed.

            branch : str
                Identifier of branch that represents the switch.

            type : str
                Type of switch, e.g. switch disconnector.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with all switches in MV network and underlying LV grids.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._switches_df
        except:
            return pd.DataFrame(columns=COLUMNS["switches_df"])

    @switches_df.setter
    def switches_df(self, df):
        self._switches_df = df

    @property
    def id(self):
        """
        MV network ID.

        Returns
        --------
        int
            MV network ID.

        """

        return self.mv_grid.id

    @property
    def mv_grid(self):
        """
        Medium voltage network.

        The medium voltage network object only contains components (lines,
        generators, etc.) that are in or connected to the MV grid and does
        not include any components of the underlying LV grids (also not
        MV/LV transformers).

        Parameters
        ----------
        mv_grid : :class:`~.network.grids.MVGrid`
            Medium voltage network.

        Returns
        --------
        :class:`~.network.grids.MVGrid`
            Medium voltage network.

        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def grid_district(self):
        """
        Dictionary with MV grid district information.

        Parameters
        ----------
        grid_district : dict
            Dictionary with the following MV grid district information:

            'population' : int
                Number of inhabitants in grid district.
            'geom' : :shapely:`shapely.MultiPolygon<MultiPolygon>`
                Geometry of MV grid district as (Multi)Polygon.
            'srid' : int
                SRID (spatial reference ID) of grid district geometry.

        Returns
        --------
        dict
            Dictionary with MV grid district information. For more information
            on the dictionary see input parameter `grid_district`.

        """
        return self._grid_district

    @grid_district.setter
    def grid_district(self, grid_district):
        self._grid_district = grid_district

    @property
    def rings(self):
        """
        List of rings in the grid topology.

        A ring is represented by the names of buses within that ring.

        Returns
        --------
        list(list)
            List of rings, where each ring is again represented by a list of
            buses within that ring.

        """
        if hasattr(self, '_rings'):
            return self._rings
        else:
            # close switches
            switches = [Switch(id=_, topology=self)
                        for _ in self.switches_df.index]
            switch_status = {}
            for switch in switches:
                switch_status[switch] = switch.state
                switch.close()
            # find rings in topology
            graph = self.to_graph()
            self.rings = nx.cycle_basis(graph)
            # reopen switches
            for switch in switches:
                if switch_status[switch] == 'open':
                    switch.open()
            return self.rings

    @rings.setter
    def rings(self, rings):
        self._rings = rings

    @property
    def equipment_data(self):
        """
        Technical data of electrical equipment such as lines and transformers.

        Returns
        --------
        dict
            Dictionary with :pandas:`pandas.DataFrame<DataFrame>` containing
            equipment data. Keys of the dictionary are 'mv_transformers',
            'mv_overhead_lines', 'mv_cables', 'lv_transformers', and
            'lv_cables'.

        """
        return self._equipment_data

    def get_connected_lines_from_bus(self, bus_name):
        """
        Returns all lines connected to specified bus.

        Parameters
        ----------
        bus_name : str
            Name of bus to get connected lines for.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with connected lines with the same format as
            :attr:`~.network.topology.Topology.lines_df`.

        """
        return self.lines_df.loc[self.lines_df.bus0 == bus_name].append(
            self.lines_df.loc[self.lines_df.bus1 == bus_name]
        )

    def get_line_connecting_buses(self, bus_1, bus_2):
        """
        Returns information of line connecting bus_1 and bus_2.

        Parameters
        ----------
        bus_1 : str
            Name of first bus.
        bus_2 : str
            Name of second bus.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information of line connecting bus_1 and bus_2
            in the same format as
            :attr:`~.network.topology.Topology.lines_df`.

        """
        lines_bus_1 = self.get_connected_lines_from_bus(bus_1)
        lines_bus_2 = self.get_connected_lines_from_bus(bus_2)
        line = [_ for _ in lines_bus_1.index if _ in lines_bus_2.index]
        if len(line) > 0:
            return self.lines_df.loc[line, :]
        else:
            return None

    def get_connected_components_from_bus(self, bus_name):
        """
        Returns dictionary of components connected to specified bus.

        Parameters
        ----------
        bus_name : str
            Identifier of bus to get connected components for.

        Returns
        -------
         dict of :pandas:`pandas.DataFrame<DataFrame>`
            Dictionary of connected components with keys 'generators', 'loads',
            'charging_points', 'storage_units', 'lines', 'transformers',
            'transformers_hvmv', 'switches'. Corresponding values are component
            dataframes containing only components that are connected to the
            given bus.

        """
        components = {}
        components["generators"] = self.generators_df.loc[
            self.generators_df.bus == bus_name
        ]
        components["loads"] = self.loads_df.loc[self.loads_df.bus == bus_name]
        components["charging_points"] = self.charging_points_df.loc[
            self.charging_points_df.bus == bus_name
        ]
        components["storage_units"] = self.storage_units_df.loc[
            self.storage_units_df.bus == bus_name
            ]
        components["lines"] = self.get_connected_lines_from_bus(bus_name)
        components["transformers"] = self.transformers_df.loc[
            self.transformers_df.bus0 == bus_name
        ].append(
            self.transformers_df.loc[self.transformers_df.bus1 == bus_name]
        )
        components["transformers_hvmv"] = self.transformers_hvmv_df.loc[
            self.transformers_hvmv_df.bus0 == bus_name
        ].append(
            self.transformers_hvmv_df.loc[
                self.transformers_hvmv_df.bus1 == bus_name
            ]
        )
        components["switches"] = self.switches_df.loc[
            self.switches_df.bus_closed == bus_name
        ]
        return components

    def get_neighbours(self, bus_name):
        """
        Returns a set of neighbour buses of specified bus.

        Parameters
        ----------
        bus_name : str
            Identifier of bus to get neighbouring buses for.

        Returns
        --------
        set(str)
            Set of identifiers of neighbouring buses.

        """
        lines = self.get_connected_lines_from_bus(bus_name)
        buses = list(lines.bus0)
        buses.extend(list(lines.bus1))
        neighbours = set(buses)
        neighbours.remove(bus_name)
        return neighbours

    def _check_bus_for_removal(self, bus_name):
        """
        Checks whether the specified bus can be safely removed from topology.

        Returns False if there is more than one line or any other component,
        such as generator, transformer, etc. connected to the given bus, as in
        that case removing the bus will lead to an invalid grid topology.

        Parameters
        ----------
        bus_name : str
            Identifier of bus for which save removal is checked.

        Returns
        -------
        bool
            True if bus can be safely removed from topology, False if removal
            of bus will lead to an invalid grid topology.

        """
        # check if bus is part of topology
        if bus_name not in self.buses_df.index:
            warnings.warn(
                "Bus of name {} not in Topology. Cannot be "
                "removed.".format(bus_name)
            )
            return False

        conn_comp = self.get_connected_components_from_bus(bus_name)
        lines = conn_comp.pop("lines")
        # if more than one line is connected, return false
        if len(lines) > 1:
            return False
        conn_comp_types = [k for k, v in conn_comp.items() if not v.empty]
        # if any other component is connected, return false
        if len(conn_comp_types) > 0:
            return False
        else:
            return True

    def _check_line_for_removal(self, line_name):
        """
        Checks whether the specified line can be safely removed from topology.

        Returns True if one of the buses the line is connected to can be
        safely removed (see
        :attr:`~.network.results.Results._check_bus_for_removal`) or if the
        line is part of a closed ring and thus removing it would not lead to
        isolated parts. In any other case, the line cannot be safely removed
        and False is returned.

        Parameters
        ----------
        line_name : str
            Identifier of line for which save removal is checked.

        Returns
        -------
        bool
            True if line can be safely removed from topology, False if removal
            of line will lead to an invalid grid topology.

        """
        # check if line is part of topology
        if line_name not in self.lines_df.index:
            warnings.warn(
                "Line of name {} not in Topology. Cannot be "
                "removed.".format(line_name)
            )
            return False

        bus0 = self.lines_df.loc[line_name, "bus0"]
        bus1 = self.lines_df.loc[line_name, "bus1"]
        # if one of the buses can be removed as well, line can be removed
        # safely
        if self._check_bus_for_removal(bus0) or \
                self._check_bus_for_removal(bus1):
            return True
        # otherwise both buses have to be in the same ring
        # find rings in topology
        graph = self.to_graph()
        rings = nx.cycle_basis(graph)
        for ring in rings:
            if bus0 in ring and bus1 in ring:
                return True
        return False

    def add_load(self, bus, peak_load, annual_consumption, **kwargs):
        """
        Adds load to topology.

        Load name is generated automatically.

        Parameters
        ----------
        bus : str
            See :py:attr:`~loads_df` for more information.
        peak_load : float
            See :py:attr:`~loads_df` for more information.
        annual_consumption : float
            See :py:attr:`~loads_df` for more information.

        Other Parameters
        -----------------
        kwargs :
            Kwargs may contain any further attributes you want to specify.
            See :py:attr:`~loads_df` for more information on additional
            attributes used for some functionalities in edisgo. Kwargs may
            also contain a load ID (provided through keyword argument
            `load_id` as string) used to generate a unique identifier
            for the newly added load.

        Returns
        --------
        str
            Unique identifier of added load.

        """
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        # generate load name and check uniqueness
        if bus_df.lv_grid_id is not None and not np.isnan(bus_df.lv_grid_id):
            grid_name = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            grid_name = "MVGrid_" + str(int(bus_df.mv_grid_id))
        tmp = grid_name
        sector = kwargs.get("sector", None)
        if sector is not None:
            tmp = tmp + "_" + sector
        load_id = kwargs.pop("load_id", None)
        if load_id is not None:
            tmp = tmp + "_" + str(load_id)
        load_name = "Load_{}".format(tmp)
        if load_name in self.loads_df.index:
            nr_loads = len(self._grids[grid_name].loads_df)
            load_name = "Load_{}_{}".format(tmp, nr_loads)
            while load_name in self.loads_df.index:
                random.seed(a=load_name)
                load_name = "Load_{}_{}".format(
                    tmp, random.randint(10 ** 8, 10 ** 9)
                )

        # create new load dataframe
        data = {
            "bus": bus,
            "peak_load": peak_load,
            "annual_consumption": annual_consumption,
        }
        data.update(kwargs)
        new_df = pd.Series(
            data,
            name=load_name,
        ).to_frame().T
        self._loads_df = self.loads_df.append(new_df)
        return load_name

    def add_generator(
        self, bus, p_nom, generator_type, control="PQ", **kwargs
    ):
        """
        Adds generator to topology.

        Generator name is generated automatically.

        Parameters
        ----------
        bus : str
            See :py:attr:`~generators_df` for more information.
        p_nom : float
            See :py:attr:`~generators_df` for more information.
        generator_type : str
            Type of generator, e.g. 'solar' or 'gas'. See 'type' in
            :py:attr:`~generators_df` for more information.
        control : str
            See :py:attr:`~generators_df` for more information. Defaults
            to 'PQ'.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain any further attributes you want to specify.
            See :py:attr:`~generators_df` for more information on additional
            attributes used for some functionalities in edisgo. Kwargs may
            also contain a generator ID (provided through keyword argument
            `generator_id` as string) used to generate a unique identifier
            for the newly added generator.

        Returns
        -------
        str
            Unique identifier of added generator.

        """
        # check if bus exists
        try:
            bus_df = self.buses_df.loc[bus]
        except KeyError:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus)
            )

        # generate generator name and check uniqueness
        if not np.isnan(bus_df.lv_grid_id) and bus_df.lv_grid_id is not None:
            tmp = "LVGrid_" + str(int(bus_df.lv_grid_id))
        else:
            tmp = "MVGrid_" + str(int(bus_df.mv_grid_id))
        tmp = tmp + "_" + generator_type
        generator_id = kwargs.pop("generator_id", None)
        if generator_id is not None:
            tmp = tmp + "_" + str(generator_id)
        generator_name = "Generator_{}".format(tmp)
        while generator_name in self.generators_df.index:
            random.seed(a=generator_name)
            generator_name = "Generator_{}_{}".format(
                tmp, random.randint(10 ** 8, 10 ** 9)
            )

        # create new generator dataframe
        data = {
            "bus": bus,
            "p_nom": p_nom,
            "type": generator_type,
            "control": control
        }
        data.update(kwargs)
        new_df = pd.Series(
            data,
            name=generator_name,
        ).to_frame().T

        self.generators_df = self.generators_df.append(new_df)
        return generator_name

    def add_charging_point(self, bus, p_nom, use_case, **kwargs):
        """
        Adds charging point to topology.

        Charging point identifier is generated automatically.

        Parameters
        ----------
        bus : str
            See :py:attr:`~charging_points_df` for more information.
        p_nom : float
            See :py:attr:`~charging_points_df` for more information.
        use_case : str
            See :py:attr:`~charging_points_df` for more information.

        Other Parameters
        -----------------
        kwargs :
            Kwargs may contain any further attributes you want to specify.

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
                random.seed(a=name)
                name = "ChargingPoint_{}_{}".format(
                    grid_name, random.randint(10 ** 8, 10 ** 9)
                )

        data = {
                "bus": bus,
                "p_nom": p_nom,
                "use_case": use_case
            }
        data.update(kwargs)
        new_df = pd.Series(
            data,
            name=name,
        ).to_frame().T
        self.charging_points_df = self.charging_points_df.append(new_df)
        return name

    def add_storage_unit(self, bus, p_nom, control="PQ", **kwargs):
        """
        Adds storage unit to topology.

        Storage unit name is generated automatically.

        Parameters
        ----------
        bus : str
            See :py:attr:`~storage_units_df` for more information.
        p_nom : float
            See :py:attr:`~storage_units_df` for more information.
        control : str, optional
            See :py:attr:`~storage_units_df` for more information. Defaults
            to 'PQ'.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain any further attributes you want to specify.

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
                random.seed(a=storage_name)
                storage_name = "StorageUnit_{}_{}".format(
                    grid_name, random.randint(10 ** 8, 10 ** 9)
                )

        # create new storage unit dataframe
        data = {
            "bus": bus,
            "p_nom": p_nom,
            "control": control
        }
        data.update(kwargs)
        new_df = pd.Series(
            data,
            name=storage_name,
        ).to_frame().T
        self.storage_units_df = self.storage_units_df.append(new_df)
        return storage_name

    def add_line(self, bus0, bus1, length, **kwargs):
        """
        Adds line to topology.

        Line name is generated automatically.
        If `type_info` is provided, `x`, `r` and `s_nom` are calculated.

        Parameters
        ----------
        bus0 : str
            Identifier of connected bus.
        bus1 : str
            Identifier of connected bus.
        length : float
            See :py:attr:`~lines_df` for more information.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain any further attributes in :py:attr:`~lines_df`.
            It is necessary to either provide `type_info` to determine `x`, `r`
            and `s_nom` of the line, or to provide `x`, `r` and `s_nom`
            directly.

        """

        def _get_line_data():
            """
            Gets line data for line type specified in `line_type` from
            equipment data.

            Returns
            --------
            :pandas:`pandas.Series<Series>`
                Line data from equipment_data.

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
            x = calculate_line_reactance(
                line_data.L_per_km, length, num_parallel)
            r = calculate_line_resistance(
                line_data.R_per_km, length, num_parallel)
            s_nom = calculate_apparent_power(
                line_data.U_n, line_data.I_max_th, num_parallel)

        # generate line name and check uniqueness
        line_name = "Line_{}_{}".format(bus0, bus1)
        while line_name in self.lines_df.index:
            random.seed(a=line_name)
            line_name = "Line_{}_{}_{}".format(
                bus0, bus1, random.randint(10 ** 8, 10 ** 9)
            )

        # check if all necessary data is now available
        if x is None or r is None:
            raise AttributeError(
                "Newly added line has no line resistance and/or reactance."
            )
        if s_nom is None:
            warnings.warn(
                "Newly added line has no nominal power."
            )

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
        self.lines_df = self.lines_df.append(new_line_df)
        return line_name

    def add_bus(self, bus_name, v_nom, **kwargs):
        """
        Adds bus to topology.

        If provided bus name already exists, a unique name is created.

        Parameters
        ----------
        bus_name : str
            Name of new bus.
        v_nom : float
            See :py:attr:`~buses_df` for more information.

        Other Parameters
        ----------------
        x : float
            See :py:attr:`~buses_df` for more information.
        y : float
            See :py:attr:`~buses_df` for more information.
        lv_grid_id : int
            See :py:attr:`~buses_df` for more information.
        in_building : bool
            See :py:attr:`~buses_df` for more information.

        Returns
        -------
        str
            Name of bus. If provided bus name already exists, a unique name
            is created.

        """
        # check uniqueness of provided bus name and otherwise change bus name
        while bus_name in self.buses_df.index:
            random.seed(a=bus_name)
            bus_name = "Bus_{}".format(
                random.randint(10 ** 8, 10 ** 9)
            )

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
        self.buses_df = self.buses_df.append(new_bus_df)
        return bus_name

    def remove_load(self, name):
        """
        Removes load with given name from topology.

        Parameters
        ----------
        name : str
            Identifier of load as specified in index of :py:attr:`~loads_df`.

        """
        if name in self.loads_df.index:
            bus = self.loads_df.at[name, "bus"]
            self._loads_df.drop(name, inplace=True)

            # if no other elements are connected, remove line and bus as well
            if self._check_bus_for_removal(bus):
                line_name = self.get_connected_lines_from_bus(bus).index[0]
                self.remove_line(line_name)
                logger.debug(
                    "Line {} removed together with load {}.".format(
                        line_name, name
                    )
                )

    def remove_generator(self, name):
        """
        Removes generator with given name from topology.

        Parameters
        ----------
        name : str
            Identifier of generator as specified in index of
            :py:attr:`~generators_df`.

        """
        if name in self.generators_df.index:
            bus = self.generators_df.at[name, "bus"]
            self._generators_df.drop(name, inplace=True)

            # if no other elements are connected to same bus, remove line
            # and bus
            if self._check_bus_for_removal(bus):
                line_name = self.get_connected_lines_from_bus(bus).index[0]
                self.remove_line(line_name)
                logger.debug(
                    "Line {} removed together with generator {}.".format(
                        line_name, name
                    )
                )

    def remove_charging_point(self, name):
        """
        Removes charging point from topology.

        Parameters
        ----------
        name : str
            Identifier of charging point as specified in index of
            :py:attr:`~charging_points_df`.

        """
        if name in self.charging_points_df.index:
            bus = self.charging_points_df.at[name, "bus"]
            self._charging_points_df.drop(name, inplace=True)

            # if no other elements are connected, remove line and bus as well
            if self._check_bus_for_removal(bus):
                line_name = self.get_connected_lines_from_bus(bus).index[0]
                self.remove_line(line_name)
                logger.debug(
                    "Line {} removed together with charging point {}.".format(
                        line_name, name
                    )
                )

    def remove_storage_unit(self, name):
        """
        Removes storage with given name from topology.

        Parameters
        ----------
        name : str
            Identifier of storage as specified in index of
            :py:attr:`~storage_units_df`.

        """
        # remove storage unit and time series
        if name in self.storage_units_df.index:
            bus = self.storage_units_df.at[name, "bus"]
            self._storage_units_df.drop(name, inplace=True)

            # if no other elements are connected, remove line and bus as well
            if self._check_bus_for_removal(bus):
                line_name = self.get_connected_lines_from_bus(bus).index[0]
                self.remove_line(line_name)
                logger.debug(
                    "Line {} removed together with storage unit {}.".format(
                        line_name, name
                    )
                )

    def remove_line(self, name):
        """
        Removes line with given name from topology.

        Parameters
        ----------
        name : str
            Identifier of line as specified in index of :py:attr:`~lines_df`.

        """
        if not self._check_line_for_removal(name):
            raise AssertionError(
                "Removal of line {} would create isolated "
                "node.".format(name)
            )

        # backup buses of line and check if buses can be removed as well
        bus0 = self.lines_df.at[name, "bus0"]
        remove_bus0 = self._check_bus_for_removal(bus0)
        bus1 = self.lines_df.at[name, "bus1"]
        remove_bus1 = self._check_bus_for_removal(bus1)

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

    def remove_bus(self, name):
        """
        Removes bus with given name from topology.

        Parameters
        ----------
        name : str
            Identifier of bus as specified in index of :py:attr:`~buses_df`.

        Notes
        -------
        Only isolated buses can be deleted from topology. Use respective
        functions first to delete all connected components (e.g. lines,
        transformers, loads, etc.). Use function
        :func:`~.network.topology.Topology.get_connected_components_from_bus`
        to get all connected components.

        """
        conn_comp = self.get_connected_components_from_bus(name)
        conn_comp_types = [k for k, v in conn_comp.items() if not v.empty]
        if len(conn_comp_types) > 0:
            warnings.warn(
                "Bus {} is not isolated and therefore not removed. Remove all "
                "connected elements ({}) first to remove bus.".format(
                    name, conn_comp_types)
            )
        else:
            self._buses_df.drop(name, inplace=True)

    def update_number_of_parallel_lines(self, lines_num_parallel):
        """
        Changes number of parallel lines and updates line attributes.

        When number of parallel lines changes, attributes x, r, and s_nom have
        to be adapted, which is done in this function.

        Parameters
        ------------
        lines_num_parallel : :pandas:`pandas.Series<series>`
            Index contains identifiers of lines to update as in index of
            :py:attr:`~lines_df` and values of series contain corresponding
            new number of parallel lines.

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
            technical parameters given in "mv_cables" or "lv_cables" of
            equipment data.

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

    def connect_to_mv(self, edisgo_object, comp_data, comp_type="Generator"):
        """
        Add and connect new generator or charging point to MV grid topology.

        This function creates a new bus the new component is connected to. The
        new bus is then connected to the grid depending on the specified
        voltage level (given in `comp_data` parameter).
        Components of voltage level 4 are connected to the HV/MV station.
        Components of voltage level 5 are connected to the nearest
        MV bus or line. In case the component is connected to a line, the line
        is split at the point closest to the new component (using perpendicular
        projection) and a new branch tee is added to connect the new
        component to.

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
            parameters of those methods. Additionally, the dictionary must
            contain the voltage level to connect in in key 'voltage_level' and
            the geolocation in key 'geom'. The
            voltage level must be provided as integer, with possible options
            being 4 (component is connected directly to the HV/MV station)
            or 5 (component is connected somewhere in the MV grid). The
            geolocation must be provided as
            :shapely:`Shapely Point object<points>`.
        comp_type : str
            Type of added component. Can be 'Generator' or 'ChargingPoint'.
            Default: 'Generator'.

        Returns
        -------
        str
            The identifier of the newly connected component.

        """
        # ToDo connect charging points via transformer?

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
                    len(self.generators_df))
        else:
            bus = "Bus_ChargingPoint_{}".format(
                len(self.charging_points_df))

        self.add_bus(
            bus_name=bus,
            v_nom=self.mv_grid.nominal_voltage,
            x=geom.x,
            y=geom.y,
        )

        # add component to newly created bus
        if comp_type == "Generator":
            comp_name = self.add_generator(
                bus=bus,
                **comp_data
            )
        else:
            comp_name = self.add_charging_point(
                bus=bus,
                **comp_data
            )

        # ===== voltage level 4: component is connected to MV station =====
        if comp_data["voltage_level"] == 4:

            # add line
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=bus,
                bus_target=self.mv_grid.station.index[0],
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001

            line_type, num_parallel = select_cable(
                edisgo_object, "mv", comp_data["p_nom"])

            line_name = self.add_line(
                bus0=self.mv_grid.station.index[0],
                bus1=bus,
                length=line_length,
                kind="cable",
                type_info=line_type.name,
                num_parallel=num_parallel
            )

            # add line to equipment changes to track costs
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[line_name],
            )

        # == voltage level 5: component is connected to MV grid
        # (next-neighbor) ==
        elif comp_data["voltage_level"] == 5:

            # get branches within the predefined `connection_buffer_radius`
            lines = geo.calc_geo_lines_in_buffer(
                grid_topology=self,
                bus=self.buses_df.loc[bus, :],
                grid=self.mv_grid,
                buffer_radius=int(
                    edisgo_object.config["grid_connection"][
                        "conn_buffer_radius"]),
                buffer_radius_inc=int(
                    edisgo_object.config["grid_connection"][
                        "conn_buffer_radius_inc"])
            )

            # calc distance between component and grid's lines -> find nearest
            # line
            conn_objects_min_stack = geo.find_nearest_conn_objects(
                grid_topology=self,
                bus=self.buses_df.loc[bus, :],
                lines=lines,
                conn_diff_tolerance=edisgo_object.config[
                    "grid_connection"]["conn_diff_tolerance"]
            )

            # connect
            # go through the stack (from nearest to farthest connection target
            # object)
            comp_connected = False
            for dist_min_obj in conn_objects_min_stack:
                # do not allow connection to virtual busses
                if "virtual" not in dist_min_obj["repr"]:
                    line_type, num_parallel = select_cable(
                        edisgo_object, "mv", comp_data["p_nom"])
                    target_obj_result = self._connect_mv_bus_to_target_object(
                        edisgo_object=edisgo_object,
                        bus=self.buses_df.loc[bus, :],
                        target_obj=dist_min_obj,
                        line_type=line_type.name,
                        number_parallel_lines=num_parallel
                    )

                    if target_obj_result is not None:
                        comp_connected = True
                        break

            if not comp_connected:
                logger.error(
                    "Component {} could not be connected. Try to "
                    "increase the parameter `conn_buffer_radius` in "
                    "config file `config_grid.cfg` to gain more possible "
                    "connection points.".format(comp_name)
                )
        return comp_name

    def connect_to_lv(self, edisgo_object, comp_data, comp_type="Generator",
                      allowed_number_of_comp_per_bus=2):
        """
        Add and connect new generator or charging point to LV grid topology.

        This function connects the new component depending on the voltage
        level, and information on the MV/LV substation ID and geometry, all
        provided in the `comp_data` parameter.
        It connects

            * Components with specified voltage level 6
                * to MV/LV substation (a new bus is created for
                  the new component, unless no geometry data is available, in
                  which case the new component is connected directly to the
                  substation)

            * Generators with specified voltage level 7
                * with a nominal capacity of <=30 kW to LV loads of type
                  residential, if available
                * with a nominal capacity of >30 kW to LV loads of type
                  retail, industrial or agricultural, if available
                * to random bus in the LV grid as fallback if no
                  appropriate load is available

            * Charging Points with specified voltage level 7
                * with use case 'home' to LV loads of type
                  residential, if available
                * with use case 'work' to LV loads of type
                  retail, industrial or agricultural, if available, otherwise
                * with use case 'public' or 'fast' to some bus in the grid that
                  is not a house connection
                * to random bus in the LV grid that
                  is not a house connection if no appropriate load is available
                  (fallback)

        In case no MV/LV substation ID is provided a random LV grid is chosen.
        In case the provided MV/LV substation ID does not exist (i.e. in case
        of components in an aggregated load area), the new component is
        directly connected to the HV/MV station (will be changed once
        generators in aggregated areas are treated differently in
        ding0).

        The number of generators or charging points connected at
        one load is restricted by the parameter
        `allowed_number_of_comp_per_bus`. If every possible load
        already has more than the allowed number then the new component
        is directly connected to the MV/LV substation.

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
            Additionally, the dictionary must contain the voltage level to
            connect in in key 'voltage_level' and may contain the geolocation
            in key 'geom' and the LV grid ID to connect the component in in key
            'mvlv_subst_id'. The voltage level must be provided as integer,
            with possible options being 6 (component is connected directly to
            the MV/LV substation) or 7 (component is connected somewhere in the
            LV grid). The geolocation must be provided as
            :shapely:`Shapely Point object<points>` and the LV grid ID as
            integer.
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

        global add_func

        def _connect_to_station():
            """
            Connects new component to substation via an own bus.

            """

            # add bus for new component
            if comp_type == "Generator":
                if comp_data["generator_id"] is not None:
                    b = "Bus_Generator_{}".format(comp_data["generator_id"])
                else:
                    b = "Bus_Generator_{}".format(
                        len(self.generators_df))
            else:
                b = "Bus_ChargingPoint_{}".format(
                    len(self.charging_points_df))

            if not type(comp_data["geom"]) is Point:
                geom = wkt_loads(comp_data["geom"])
            else:
                geom = comp_data["geom"]

            self.add_bus(
                bus_name=b,
                v_nom=lv_grid.nominal_voltage,
                x=geom.x,
                y=geom.y,
                lv_grid_id=lv_grid.id,
            )

            # add line to connect new component
            station_bus = lv_grid.station.index[0]
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=b,
                bus_target=station_bus,
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001
            # get suitable line type
            line_type, num_parallel = select_cable(
                edisgo_object, "lv", comp_data["p_nom"])
            line_name = self.add_line(
                bus0=station_bus,
                bus1=b,
                length=line_length,
                kind="cable",
                type_info=line_type.name,
                num_parallel=num_parallel
            )

            # add line to equipment changes to track costs
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[line_name],
            )

            # add new component
            return add_func(
                bus=b, **comp_data
            )

        def _choose_random_substation_id():
            """
            Returns a random LV grid to connect component in in case no
            substation ID is provided or it does not exist.

            """
            if comp_type == "Generator":
                random.seed(a=comp_data["generator_id"])
            else:
                # ToDo: Seed shouldn't depend on number of charging points, but
                #  there is currently no better solution
                random.seed(a=len(self.charging_points_df))
            lv_grid_id = random.choice(lv_grid_ids)
            return LVGrid(id=lv_grid_id, edisgo_obj=edisgo_object)

        # get list of LV grid IDs
        lv_grid_ids = [_.id for _ in self.mv_grid.lv_grids]

        if comp_type == "Generator":
            add_func = self.add_generator
        elif comp_type == "ChargingPoint":
            add_func = self.add_charging_point
        else:
            logger.error(
                "Component type {} is not a valid option.".format(comp_type)
            )

        if comp_data["mvlv_subst_id"]:

            # if substation ID (= LV grid ID) is given and it matches an
            # existing LV grid ID (i.e. it is no aggregated LV grid), set grid
            # to connect component to to specified grid (in case the component
            # has no geometry it is connected to the grid's station)
            if comp_data["mvlv_subst_id"] in lv_grid_ids:

                # get LV grid
                lv_grid = self._grids[
                    "LVGrid_{}".format(int(comp_data["mvlv_subst_id"]))
                ]

            # if substation ID (= LV grid ID) is given but it does not match an
            # existing LV grid ID a random LV grid to connect in is chosen
            else:
                # ToDo
                # lv_grid = _choose_random_substation_id()
                # warnings.warn(
                #     "Given mvlv_subst_id does not exist, wherefore a random "
                #     "LV Grid ({}) to connect in is chosen.".format(
                #         lv_grid.id
                #     )
                # )
                comp_name = add_func(
                    bus=self.mv_grid.station.index[0],
                    **comp_data
                )
                return comp_name

        # if no MV/LV substation ID is given, choose random LV grid
        else:
            lv_grid = _choose_random_substation_id()
            warnings.warn(
                "Component has no mvlv_subst_id. It is therefore allocated "
                "to a random LV Grid ({}).".format(
                    lv_grid.id
                )
            )

        # v_level 6 -> connect to grid's LV station
        if comp_data["voltage_level"] == 6:
            # if no geom is given, connect directly to LV grid's station, as
            # connecting via separate bus will otherwise throw an error (see
            # _connect_to_station function)
            if ("geom" not in comp_data.keys()) or \
                    ("geom" in comp_data.keys() and not comp_data["geom"]):
                comp_name = add_func(
                    bus=lv_grid.station.index[0], **comp_data
                )
                logger.debug(
                    "Component {} has no geom entry and will be connected "
                    "to grid's LV station.".format(comp_name)
                )
            else:
                comp_name = _connect_to_station()
            return comp_name

        # v_level 7 -> connect in LV grid
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
                        ~lv_grid.buses_df.in_building.astype(bool)].index

            # generate random list (unique elements) of possible target buses
            # to connect components to
            if comp_type == "Generator":
                random.seed(a=comp_data["generator_id"])
            else:
                random.seed(
                    a="{}_{}_{}".format(
                        comp_data["use_case"],
                        comp_data["p_nom"],
                        len(lv_grid.charging_points_df)
                    )
                )

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
                    lv_grid.buses_df[
                        ~lv_grid.buses_df.in_building.astype(bool)].index
                )
                comp_name = add_func(
                    bus=bus, **comp_data
                )
                return comp_name

            # search through list of target buses for bus with less
            # than or equal the allowed number of components of the same type
            # already connected to it
            lv_conn_target = None

            while len(lv_buses_rnd) > 0 and lv_conn_target is None:

                lv_bus = lv_buses_rnd.pop()

                # determine number of components of the same type at LV bus
                if comp_type == "Generator":
                    comps_at_bus = self.generators_df[
                        self.generators_df.bus == lv_bus
                    ]
                else:
                    comps_at_bus = self.charging_points_df[
                        self.charging_points_df.bus == lv_bus
                    ]

                # ToDo: Increase number of generators/charging points
                #  allowed at one load in case all loads already have one
                #  generator/charging point
                if len(comps_at_bus) <= allowed_number_of_comp_per_bus:
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

    def _connect_mv_bus_to_target_object(self, edisgo_object, bus, target_obj,
                                         line_type, number_parallel_lines):
        """
        Connects given MV bus to given target object (MV line or bus).

        If the target object is a bus, a new line between the two buses is
        created.
        If the target object is a line, the bus is connected to a newly
        created bus (using perpendicular projection) on this line.
        New lines are created using the line type specified through parameter
        `line_type` and using the number of parallel lines specified through
        parameter `number_parallel_lines`.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        bus : :pandas:`pandas.Series<Series>`
            Data of bus to connect.
            Series has same rows as columns of
            :attr:`~.network.topology.Topology.buses_df`.
        target_obj : dict
            Dictionary containing the following necessary target object
            information:

                * repr : str
                    Name of line or bus to connect to.
                * shp : :shapely:`Shapely Point object<points>` or \
                :shapely:`Shapely Line object<lines>`
                    Geometry of line or bus to connect to.

        line_type : str
            Line type to use to connect new component with.
        number_parallel_lines : int
            Number of parallel lines to connect new component with.

        Returns
        -------
        str
            Name of the bus the given bus was connected to.

        """

        srid = self.grid_district["srid"]
        bus_shp = transform(geo.proj2equidistant(srid), Point(bus.x, bus.y))

        # MV line is nearest connection point => split old line into 2 segments
        # (delete old line and create 2 new ones)
        if isinstance(target_obj["shp"], LineString):

            line_data = self.lines_df.loc[
                            target_obj["repr"], :
                        ]

            # if line that is split is connected to switch, the line name needs
            # to be adapted in the switch information
            if line_data.name in self.switches_df.branch.values:
                # get switch
                switch_data = self.switches_df[
                    self.switches_df.branch ==
                    line_data.name].iloc[0]
                # get bus to which the new line will be connected
                switch_bus = (switch_data.bus_open
                              if switch_data.bus_open
                                 in line_data.loc[["bus0", "bus1"]].values
                              else switch_data.bus_closed
                              )
            else:
                switch_bus = None

            # find nearest point on MV line
            conn_point_shp = target_obj["shp"].interpolate(
                target_obj["shp"].project(bus_shp)
            )
            conn_point_shp = transform(
                geo.proj2equidistant_reverse(srid), conn_point_shp
            )

            # create new branch tee bus
            branch_tee_repr = "BranchTee_{}".format(target_obj["repr"])
            self.add_bus(
                bus_name=branch_tee_repr,
                v_nom=self.mv_grid.nominal_voltage,
                x=conn_point_shp.x,
                y=conn_point_shp.y,
            )

            # add new line between newly created branch tee and line's bus0
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=line_data.bus0,
                bus_target=branch_tee_repr,
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001
            line_name_bus0 = self.add_line(
                bus0=branch_tee_repr,
                bus1=line_data.bus0,
                length=line_length,
                kind=line_data.kind,
                type_info=line_data.type_info,
            )
            # if line connected to switch was split, write new line name to
            # switch data
            if switch_bus and switch_bus == line_data.bus0:
                self.switches_df.loc[
                    switch_data.name, "branch"] = line_name_bus0
            # add line to equipment changes
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[line_name_bus0, :],
            )

            # add new line between newly created branch tee and line's bus0
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=line_data.bus1,
                bus_target=branch_tee_repr,
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001
            line_name_bus1 = self.add_line(
                bus0=branch_tee_repr,
                bus1=line_data.bus1,
                length=line_length,
                kind=line_data.kind,
                type_info=line_data.type_info,
            )
            # if line connected to switch was split, write new line name to
            # switch data
            if switch_bus and switch_bus == line_data.bus1:
                self.switches_df.loc[
                    switch_data.name, "branch"] = line_name_bus1
            # add line to equipment changes
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[line_name_bus1, :],
            )

            # add new line for new bus
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=bus.name,
                bus_target=branch_tee_repr,
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001
            new_line_name = self.add_line(
                bus0=branch_tee_repr,
                bus1=bus.name,
                length=line_length,
                kind="cable",
                type_info=line_type,
                num_parallel=number_parallel_lines
            )
            # add line to equipment changes
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[new_line_name, :],
            )

            # remove old line from topology and equipment changes
            self.remove_line(line_data.name)
            edisgo_object.results._del_line_from_equipment_changes(
                line_repr=line_data.name
            )

            return branch_tee_repr

        # bus ist nearest connection point
        else:

            # add new branch for satellite (station to station)
            line_length = geo.calc_geo_dist_vincenty(
                grid_topology=self,
                bus_source=bus.name,
                bus_target=target_obj["repr"],
                branch_detour_factor=edisgo_object.config["grid_connection"][
                    "branch_detour_factor"
                ]
            )
            # avoid very short lines by limiting line length to at least 1m
            if line_length < 0.001:
                line_length = 0.001

            new_line_name = self.add_line(
                bus0=target_obj["repr"],
                bus1=bus.name,
                length=line_length,
                kind="cable",
                type_info=line_type,
                num_parallel=number_parallel_lines
            )

            # add line to equipment changes
            edisgo_object.results._add_line_to_equipment_changes(
                line=self.lines_df.loc[new_line_name, :],
            )

            return target_obj["repr"]

    def to_graph(self):
        """
        Returns graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """
        graph = networkx_helper.translate_df_to_graph(
            self.buses_df,
            self.lines_df,
            self.transformers_df,
        )
        return graph

    def to_csv(self, directory):
        """
        Exports topology to csv files.

        The following attributes are exported:

        * 'loads_df' : Attribute :py:attr:`~loads_df` is saved to
          `loads.csv`.
        * 'generators_df' : Attribute :py:attr:`~generators_df` is saved to
          `generators.csv`.
        * 'charging_points_df' : Attribute :py:attr:`~charging_points_df` is
          saved to `charging_points.csv`.
        * 'storage_units_df' : Attribute :py:attr:`~storage_units_df` is
          saved to `storage_units.csv`.
        * 'transformers_df' : Attribute :py:attr:`~transformers_df` is saved to
          `transformers.csv`.
        * 'transformers_hvmv_df' : Attribute :py:attr:`~transformers_df` is
          saved to `transformers.csv`.
        * 'lines_df' : Attribute :py:attr:`~lines_df` is saved to
          `lines.csv`.
        * 'buses_df' : Attribute :py:attr:`~buses_df` is saved to
          `buses.csv`.
        * 'switches_df' : Attribute :py:attr:`~switches_df` is saved to
          `switches.csv`.
        * 'grid_district' : Attribute :py:attr:`~grid_district` is saved to
          `network.csv`.

        Attributes are exported in a way that they can be directly imported to
        pypsa.

        Parameters
        ----------
        directory : str
            Path to save topology to.

        """
        os.makedirs(directory, exist_ok=True)
        if not self.loads_df.empty:
            self.loads_df.to_csv(os.path.join(directory, "loads.csv"))
        if not self.generators_df.empty:
            self.generators_df.to_csv(
                os.path.join(directory, "generators.csv")
            )
        if not self.charging_points_df.empty:
            self.charging_points_df.to_csv(
                os.path.join(directory, "charging_points.csv")
            )
        if not self.storage_units_df.empty:
            self.storage_units_df.to_csv(
                os.path.join(directory, "storage_units.csv")
            )
        if not self.transformers_df.empty:
            self.transformers_df.rename(
                {"x_pu": "x", "r_pu": "r"}, axis=1
            ).to_csv(os.path.join(directory, "transformers.csv"))
        if not self.transformers_hvmv_df.empty:
            self.transformers_hvmv_df.rename(
                {"x_pu": "x", "r_pu": "r"}, axis=1
            ).to_csv(os.path.join(directory, "transformers_hvmv.csv"))
        self.lines_df.to_csv(os.path.join(directory, "lines.csv"))
        self.buses_df.to_csv(os.path.join(directory, "buses.csv"))
        if not self.switches_df.empty:
            self.switches_df.to_csv(os.path.join(directory, "switches.csv"))

        network = {"name": self.mv_grid.id}
        network.update(self._grid_district)
        pd.DataFrame([network]).set_index("name").rename(
            {
                "geom": "mv_grid_district_geom",
                "population": "mv_grid_district_population",
            },
            axis=1,
        ).to_csv(os.path.join(directory, "network.csv"))

    def from_csv(self, directory, edisgo_obj):
        """
        Restores topology from csv files.

        Parameters
        ----------
        directory : str
            Path to topology csv files.

        """
        self.buses_df = pd.read_csv(
            os.path.join(directory, "buses.csv"),
            index_col=0
        )
        self.lines_df = pd.read_csv(
            os.path.join(directory, "lines.csv"),
            index_col=0
        )
        if os.path.exists(os.path.join(directory, "loads.csv")):
            self.loads_df = pd.read_csv(
                os.path.join(directory, "loads.csv"),
                index_col=0
            )
        if os.path.exists(os.path.join(directory, "generators.csv")):
            generators_df = pd.read_csv(
                os.path.join(directory, "generators.csv"),
                index_col=0
            )
            # delete slack if it was included
            slack = generators_df.loc[
                generators_df.control == "Slack"].index
            self.generators_df = generators_df.drop(slack)
        if os.path.exists(os.path.join(directory, "charging_points.csv")):
            self.charging_points_df = pd.read_csv(
                os.path.join(directory, "charging_points.csv"),
                index_col=0
            )
        if os.path.exists(os.path.join(directory, "storage_units.csv")):
            self.storage_units_df = pd.read_csv(
                os.path.join(directory, "storage_units.csv"),
                index_col=0
            )
        if os.path.exists(os.path.join(directory, "transformers.csv")):
            self.transformers_df = pd.read_csv(
                os.path.join(directory, "transformers.csv"),
                index_col=0
            ).rename(
                columns={"x": "x_pu",
                         "r": "r_pu"}
            )
        if os.path.exists(os.path.join(directory, "transformers_hvmv.csv")):
            self.transformers_hvmv_df = pd.read_csv(
                os.path.join(directory, "transformers_hvmv.csv"),
                index_col=0
            ).rename(
                columns={"x": "x_pu",
                         "r": "r_pu"}
            )
        if os.path.exists(os.path.join(directory, "switches.csv")):
            self.switches_df = pd.read_csv(
                os.path.join(directory, "switches.csv"),
                index_col=0
            )

        # import network data
        network = pd.read_csv(os.path.join(directory, "network.csv")).\
            rename(columns={
                "mv_grid_district_geom": "geom",
                "mv_grid_district_population": "population",
            })
        self.grid_district = {
            "population": network.population[0],
            "geom": wkt_loads(network.geom[0]),
            "srid": network.srid[0],
        }
        # set up medium voltage grid
        self.mv_grid = MVGrid(
            edisgo_obj=edisgo_obj,
            id=network['name'].values[0]
        )
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

    def __repr__(self):
        return "Network topology " + str(self.id)
