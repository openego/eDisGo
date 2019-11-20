import logging
import random
import pandas as pd
import numpy as np
import os

import edisgo
from edisgo.network.components import Generator, Load


logger = logging.getLogger('edisgo')


class Topology:
    """
    Used as container for all data related to a single
    :class:`~.network.grids.MVGrid`.

    Parameters
    -----------
    config_data : :class:`~.tools.config.Config`
        Config object with configuration data from config files.

    Attributes
    -----------
    _grid_district : :obj:`dict`
        Contains the following information about the supplied
        region (network district) of the network:
        'geom': Shape of network district as MultiPolygon.
        'population': Number of inhabitants.
    _grids : dict

    """
    #ToDo Implement update (and add) functions for component dataframes to
    # avoid using protected variables in other classes and modules

    def __init__(self, **kwargs):

        # load configuration and equipment data
        self._equipment_data = self._load_equipment_data(
            kwargs.get('config', None))


    def _load_equipment_data(self, config=None):
        """
        Load equipment data for transformers, cables etc.

        Parameters
        -----------
        config : :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        Returns
        -------
        :obj:`dict`
            Dictionary with :pandas:`pandas.DataFrame<dataframe>` containing
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

        equipment = {'mv': ['transformers', 'overhead_lines', 'cables'],
                     'lv': ['transformers', 'cables']}

        # if config is not provided set default path and filenames
        if config is None:
            equipment_dir = 'equipment'
            config = {}
            for voltage_level, eq_list in equipment.items():
                for i in eq_list:
                    config['equipment_{}_parameters_{}'.format(
                        voltage_level, i)] = \
                        'equipment-parameters_{}_{}.csv'.format(
                            voltage_level.upper(), i)
        else:
            equipment_dir = config['system_dirs']['equipment_dir']
            config = config['equipment']

        package_path = edisgo.__path__[0]
        data = {}

        for voltage_level, eq_list in equipment.items():
            for i in eq_list:
                equipment_parameters = config[
                    'equipment_{}_parameters_{}'.format(voltage_level, i)]
                data['{}_{}'.format(voltage_level, i)] = pd.read_csv(
                    os.path.join(package_path, equipment_dir,
                                 equipment_parameters),
                    comment='#', index_col='name',
                    delimiter=',', decimal='.')
                # calculate electrical values of transformer from standard
                # values (so far only for LV transformers, not necessary for
                # MV as MV impedances are not used)
                if voltage_level == 'lv' and i == 'transformers':
                    data['{}_{}'.format(voltage_level, i)]['r_pu'] = \
                        data['{}_{}'.format(voltage_level, i)]['P_k'] / \
                        (data['{}_{}'.format(voltage_level, i)][
                             'S_nom'] )
                    data['{}_{}'.format(voltage_level, i)][
                        'x_pu'] = np.sqrt(
                        (data['{}_{}'.format(voltage_level, i)][
                             'u_kr'] / 100) ** 2 \
                        - data['{}_{}'.format(voltage_level, i)][
                            'r_pu'] ** 2)
        return data

    @property
    def equipment_data(self):
        """
        Technical data of electrical equipment such as lines and transformers.

        Returns
        --------
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
            Data of electrical equipment.

        """
        return self._equipment_data

    @property
    def buses_df(self):
        """
        Dataframe with all buses in MV network and underlying LV grids.

        Parameters
        ----------
        buses_df : :pandas:`pandas.DataFrame<dataframe>`
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
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in MV network and underlying LV grids.

        """
        return self._buses_df

    @buses_df.setter
    def buses_df(self, buses_df):
        self._buses_df = buses_df

    @property
    def generators_df(self):
        """
        Dataframe with all generators in MV network and underlying LV grids.

        Parameters
        ----------
        generators_df : :pandas:`pandas.DataFrame<dataframe>`
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
        :pandas:`pandas.DataFrame<dataframe>`
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
        loads_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all loads in MV network and underlying LV grids.
            Index of the dataframe are load names. Columns of the
            dataframe are:
            bus
            peak_load
            sector
            annual_consumption

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
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
        transformers_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all transformers.
            Index of the dataframe are transformer names. Columns of the
            dataframe are:
            bus0
            bus1
            x_pu
            r_pu
            s_nom
            type_info

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
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
        transformers_df : :pandas:`pandas.DataFrame<dataframe>`
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
        :pandas:`pandas.DataFrame<dataframe>`
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
        lines_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all lines in MV network and underlying LV grids.
            Index of the dataframe are line names. Columns of the
            dataframe are:
            bus0
            bus1
            length
            x
            r
            s_nom
            num_parallel
            type_info
            kind

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
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
        switches_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all switches in MV network and underlying LV grids.
            Index of the dataframe are switch names. Columns of the
            dataframe are:
            bus_open
            bus_closed
            branch
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all switches in MV network and underlying LV grids.

        """
        return self._switches_df

    @switches_df.setter
    def switches_df(self, switches_df):
        self._switches_df = switches_df

    @property
    def storages_df(self):
        """
        Dataframe with all storages in MV network and underlying LV grids.

        Parameters
        ----------
        storages_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all storages in MV network and underlying LV grids.
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
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all storages in MV network and underlying LV grids.

        """
        return self._storages_df

    @storages_df.setter
    def storages_df(self, storages_df):
        self._storages_df = storages_df

    @property
    def generators(self):
        """
        Connected generators within the network.

        Returns
        -------
        list(:class:`~.network.components.Generator`)
            List of generators within the network.

        """
        for gen in self.generators_df.drop(labels=['Generator_slack']).index:
            yield Generator(id=gen)

    @property
    def loads(self):
        """
        Connected loads within the network.

        Returns
        -------
        list(:class:`~.network.components.Load`)
            List of loads within the network.

        """
        for l in self.loads_df.index:
            yield Load(id=l)

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

    def remove_generator(self, generator_name):
        """
        Removes generator with given name from topology.

        Parameters
        ----------
        generator_name : str
            Name of generator as specified in index of `generators_df`.

        """
        # ToDo add test
        self._generators_df.drop(generator_name)

    def add_generator(self, generator_id, bus, p_nom, generator_type,
                      weather_cell_id=None, subtype=None, control=None):
        """
        Adds generator to topology.

        Generator name is generated automatically.

        Parameters
        ----------
        generator_id : str
            Unique identifier of generator.
        bus
        control
        p_nom
        generator_type
        weather_cell_id
        subtype

        """
        #ToDo add test
        # check if bus exists
        if bus not in self.buses_df.index:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus))

        # generate generator name and check uniqueness
        generator_name = 'Generator_{}_{}'.format(generator_type, generator_id)
        while generator_name in self.generators_df.index:
            generator_name = 'Generator_{}_{}'.format(
                generator_type, random.randint(10**8, 10**9), generator_id)

        new_gen_df = pd.DataFrame(
            data={'bus': bus,
                  'p_nom': p_nom,
                  'control': control if not None else 'PQ',
                  'type': generator_type,
                  'weather_cell_id': weather_cell_id,
                  'subtype': subtype},
            index=[generator_name])
        self.generators_df = self._generators_df.append(new_gen_df)
        return generator_name

    def add_bus(self, bus_name, v_nom, x=None, y=None, lv_grid_id=None,
                in_building=False):
        """
        Adds new bus to topology.

        Parameters
        ----------
        bus_name : str
        v_nom
        x
        y
        lv_grid_id
        in_building

        """
        #ToDo add test
        # check lv_grid_id
        if v_nom < 1 and lv_grid_id is None:
            raise ValueError(
                "You need to specify an lv_grid_id for low-voltage buses.")
        new_bus_df = pd.DataFrame(
            data={'v_nom': v_nom,
                  'x': x,
                  'y': y,
                  'mv_grid_id': self.mv_grid.id,
                  'lv_grid_id': lv_grid_id,
                  'in_building': in_building},
            index=[bus_name])
        self._buses_df = self._buses_df.append(new_bus_df)

    def add_line(self, bus0, bus1, length, x=None, r=None,
                 s_nom=None, num_parallel=1, type_info=None, kind=None):
        """
        Adds new line to topology.

        Line name is generated automatically.

        Parameters
        ----------
        bus0
        bus1
        length
        x
        r
        s_nom
        num_parallel
        type_info
        kind

        """
        #ToDo add test
        # check if buses exist
        if bus0 not in self.buses_df.index:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus0))
        if bus1 not in self.buses_df.index:
            raise ValueError(
                "Specified bus {} is not valid as it is not defined in "
                "buses_df.".format(bus1))

        # check if line between given buses already exists
        bus0_bus1 = self.lines_df[
            (self.lines_df.bus0 == bus0) & (self.lines_df.bus1 == bus1)]
        bus1_bus0 = self.lines_df[
            (self.lines_df.bus1 == bus0) & (self.lines_df.bus0 == bus1)]
        if not bus0_bus1.empty and bus1_bus0.empty:
            logging.debug("Line between bus0 {} and bus1 {} already exists.")
            return bus1_bus0.append(bus0_bus1).index[0]

        # generate line name and check uniqueness
        line_name = 'Line_{}_{}'.format(bus0, bus1)
        while line_name in self.lines_df.index:
            line_name = 'Line_{}_{}_{}'.format(
                bus0, bus1, random.randint(10**8, 10**9))

        #ToDo
        # # calculate r if not provided
        # if x is None and type_info:
        new_line_df = pd.DataFrame(
            data={'bus0': bus0,
                  'bus1': bus1,
                  'x': x,
                  'r': r,
                  'length': length,
                  'type_info': type_info,
                  'num_parallel': num_parallel,
                  'kind': kind,
                  's_nom': s_nom},
            index=[line_name])
        self._lines_df = self._lines_df.append(new_line_df)
        return line_name

    def __repr__(self):
        return 'Network topology ' + str(self.id)
