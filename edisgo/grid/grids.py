from abc import ABC, abstractmethod

from edisgo.grid.components import Generator, Load, Switch
from edisgo.grid import network
from edisgo.tools import pypsa_io


class Grid(ABC):
    """
    Defines a basic grid in eDisGo

    Parameters
    -----------
    _id : str or int
        Identifier
    _network : :class:`~.grid.network.Network`
        Network container.

    # ToDo add annual_consumption property?

    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        if isinstance(self._id, float):
            self._id = int(self._id)
        self._network = network.NETWORK

        self._nominal_voltage = None

        # # ToDo Implement if necessary
        # self._station = None
        # ToDo maybe add lines_df and lines property if needed

    @property
    def id(self):
        return self._id

    @property
    def network(self):
        return self._network

    @property
    def nominal_voltage(self):
        """
        Nominal voltage of grid in V.

        Parameters
        ----------
        nominal_voltage : float

        Returns
        -------
        float
            Nominal voltage of grid in V.

        """
        if self._nominal_voltage is None:
            self._nominal_voltage = self.buses_df.v_nom.max()
        return self._nominal_voltage

    @nominal_voltage.setter
    def nominal_voltage(self, nominal_voltage):
        self._nominal_voltage = nominal_voltage

    @property
    def station(self):
        """
        Bus that represents station of grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe that contains station bus. Format is the same as for
            buses_df.
        """
        return self.network.buses_df.loc[
            self.transformers_df.bus1.unique()
        ]

    @property
    def transformers_df(self):
        """
        Transformers to overlaying grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all transformers to overlaying grid. For more
            information on the dataframe see
            :attr:`~.grid.network.Network.transformers_df`.
        """
        return self.network.transformers_df[
            self.network.transformers_df.bus1.isin(self.buses_df.index)]

    @property
    def generators_df(self):
        """
        Connected generators within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all generators in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.generators_df`.

        """
        return self.network.generators_df[
            self.network.generators_df.bus.isin(self.buses_df.index)]

    @property
    def generators(self):
        """
        Connected generators within the grid.

        Returns
        -------
        list(:class:`~.grid.components.Generator`)
            List of generators within the grid.

        """
        for gen in self.generators_df.index:
            yield Generator(id=gen)

    @property
    def loads_df(self):
        """
        Connected loads within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all loads in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.loads_df`.

        """
        return self.network.loads_df[
            self.network.loads_df.bus.isin(self.buses_df.index)]

    @property
    def loads(self):
        """
        Connected loads within the grid.

        Returns
        -------
        list(:class:`~.grid.components.Load`)
            List of loads within the grid.

        """
        for l in self.loads_df.index:
            yield Load(id=l)

    @property
    def storages_df(self):
        """
        Connected storage units within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all storages in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.storages_df`.
        """
        return self.network.storages_df[
            self.network.storages_df.bus.isin(self.buses_df.index)]

    @property
    def switch_disconnectors_df(self):
        """
        Switch disconnectors in grid.

        Switch disconnectors are points where rings are split under normal
        operating conditions.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all switch disconnectors in grid. For more
            information on the dataframe see
            :attr:`~.grid.network.Network.switches_df`.

        """
        return self.network.switches_df[
            self.network.switches_df.bus_closed.isin(self.buses_df.index)][
            self.network.switches_df.type_info=='Switch Disconnector']

    @property
    def switch_disconnectors(self):
        """
        Switch disconnectors within the grid.

        Returns
        -------
        list(:class:`~.grid.components.Switch`)
            List of switch disconnectory within the grid.

        """
        for s in self.switch_disconnectors_df.index:
            yield Switch(id=s)

    @property
    def lines_df(self):
        """
        Lines within the grid.

         Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.lines_df`.
        """
        return self.network.lines_df[
            self.network.lines_df.bus0.isin(self.buses_df.index) &
            self.network.lines_df.bus1.isin(self.buses_df.index)]

    @property
    @abstractmethod
    def buses_df(self):
        """
        Buses within the grid.

         Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.buses_df`.

        """

    @property
    def weather_cells(self):
        """
        Weather cells in grid.

        Returns
        -------
        list(int)
            List of weather cell IDs in grid.

        """
        return self.generators_df.weather_cell_id.dropna().unique()

    @property
    def peak_generation_capacity(self):
        """
        Cumulative peak generation capacity of generators in the grid in MW.

        Returns
        -------
        float
            Cumulative peak generation capacity of generators in the grid
            in MW.

        """
        return self.generators_df.p_nom.sum()

    @property
    def peak_generation_capacity_per_technology(self):
        """
        Cumulative peak generation capacity of generators in the grid per
        technology type in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Cumulative peak generation capacity of generators in the grid per
            technology type in MW.

        """
        return self.generators_df.groupby(['type']).sum()['p_nom']

    @property
    def peak_load(self):
        """
        Cumulative peak load of loads in the grid in MW.

        Returns
        -------
        float
            Cumulative peak load of loads in the grid in MW.

        """
        return self.loads_df.peak_load.sum()

    @property
    def peak_load_per_sector(self):
        """
        Cumulative peak load of loads in the grid per sector in MW.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Cumulative peak load of loads in the grid per sector in MW.

        """
        return self.loads_df.groupby(['sector']).sum()['peak_load']

    def __repr__(self):
        return '_'.join([self.__class__.__name__, str(self.id)])

    def connect_generators(self, generators):
        """
        Connects generators to grid.

        Parameters
        ----------
        generators : :pandas:`pandas.DataFrame<dataframe>`
            Generators to be connected.

        """
        # ToDo: Should we implement this or move function from tools here?
        raise NotImplementedError

    def to_pypsa(self, mode = 'mv', timesteps=None):
        """
        PyPSA grid representation

        A grid topology representation based on
        :pandas:`pandas.DataFrame<dataframe>`. The overall container object of
        this data model, the :pypsa:`pypsa.Network<network>`,
        is assigned to this attribute.
        Todo: Docstring
        :param mode:
        :param timesteps:

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA grid representation. The attribute `edisgo_mode` is added
            to specify if pypsa representation of the edisgo network
            was created for the whole grid topology (MV + LV), only MV or only
            LV. See parameter `mode` in
            :meth:`~.grid.network.EDisGo.analyze` for more information.
        """
        if timesteps is None:
            timesteps = self.network.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]
        return pypsa_io.to_pypsa(self, mode=mode, timesteps=timesteps)


class MVGrid(Grid):
    """
    Defines a medium voltage grid in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._lv_grids = kwargs.get('lv_grids', [])

    @property
    def lv_grids(self):
        """
        Underlying LV grids.

        Parameters
        ----------
        lv_grids : list(:class:`~.grid.grids.LVGrid`)

        Returns
        -------
        list generator
            Generator object of underlying LV grids of type
            :class:`~.grid.grids.LVGrid`.

        """
        for lv_grid in self._lv_grids:
            yield lv_grid

    @lv_grids.setter
    def lv_grids(self, lv_grids):
        self._lv_grids = lv_grids

    @property
    def buses_df(self):
        """
        Buses within the grid.

         Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.buses_df`.

        """
        return self.network.buses_df.drop(
            self.network.buses_df.lv_grid_id.dropna().index)

    def draw(self):
        """
        Draw MV grid.

        """
        # ToDo call EDisGoReimport.plot_mv_grid_topology
        raise NotImplementedError


class LVGrid(Grid):
    """
    Defines a low voltage grid in eDisGo.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def buses_df(self):
        """
        Buses within the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in grid. For more information on the
            dataframe see :attr:`~.grid.network.Network.buses_df`.

        """
        return self.network.buses_df.loc[
            self.network.buses_df.lv_grid_id == self.id]

    def draw(self):
        """
        Draw LV grid.

        """
        # ToDo: implement networkx graph plot
        raise NotImplementedError
