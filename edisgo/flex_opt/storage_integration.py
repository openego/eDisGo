from math import sqrt
import logging
import pandas as pd

from edisgo.network.grids import MVGrid
from edisgo.network.tools import select_cable
from edisgo.tools import pypsa_io
from edisgo.flex_opt.storage_operation import fifty_fifty
from edisgo.flex_opt.storage_positioning import one_storage_per_feeder
from edisgo.network.components import Load, Generator, Storage

logger = logging.getLogger('edisgo')

def storage_at_hvmv_substation(mv_grid, parameters, mode=None):
    """
    Place storage at HV/MV substation bus bar.

    Parameters
    ----------
    mv_grid : :class:`~.network.grids.MVGrid`
        MV network instance
    parameters : :obj:`dict`
        Dictionary with storage parameters. Must at least contain
        'nominal_power'. See :class:`~.network.network.StorageControl` for more
        information.
    mode : :obj:`str`, optional
        Operational mode. See :class:`~.network.network.StorageControl` for
        possible options and more information. Default: None.

    Returns
    -------
    :class:`~.network.components.Storage`, :class:`~.network.components.Line`
        Created storage instance and newly added line to connect storage.

    """
    storage = set_up_storage(node=mv_grid.station, parameters=parameters,
                             operational_mode=mode)
    line = connect_storage(storage, mv_grid.station)
    return storage, line


def set_up_storage(node, parameters,
                   voltage_level=None, operational_mode=None):
    """
    Sets up a storage instance.

    Parameters
    ----------
    node : :class:`~.network.components.Station` or :class:`~.network.components.BranchTee`
        Node the storage will be connected to.
    parameters : :obj:`dict`, optional
        Dictionary with storage parameters. Must at least contain
        'nominal_power'. See :class:`~.network.network.StorageControl` for more
        information.
    voltage_level : :obj:`str`, optional
        This parameter only needs to be provided if `node` is of type
        :class:`~.network.components.LVStation`. In that case `voltage_level`
        defines which side of the LV station the storage is connected to. Valid
        options are 'lv' and 'mv'. Default: None.
    operational_mode : :obj:`str`, optional
        Operational mode. See :class:`~.network.network.StorageControl` for
        possible options and more information. Default: None.

    """

    # if node the storage is connected to is an LVStation voltage_level
    # defines which side the storage is connected to
    if isinstance(node, LVStation):
        if voltage_level == 'lv':
            grid = node.grid
        elif voltage_level == 'mv':
            grid = node.mv_grid
        else:
            raise ValueError(
                "{} is not a valid option for voltage_level.".format(
                    voltage_level))
    else:
        grid = node.grid

    return Storage(operation=operational_mode,
                   id='{}_storage_{}'.format(grid,
                                             len(grid.graph.nodes_by_attribute(
                                                 'storage')) + 1),
                   grid=grid,
                   geom=node.geom,
                   **parameters)


def connect_storage(storage, node):
    """
    Connects storage to the given node.

    The storage is connected by a cable
    The cable the storage is connected with is selected to be able to carry
    the storages nominal power and equal amount of reactive power.
    No load factor is considered.

    Parameters
    ----------
    storage : :class:`~.network.components.Storage`
        Storage instance to be integrated into the network.
    node : :class:`~.network.components.Station` or :class:`~.network.components.BranchTee`
        Node the storage will be connected to.

    Returns
    -------
    :class:`~.network.components.Line`
        Newly added line to connect storage.

    """

    # add storage itself to graph
    storage.grid.graph.add_node(storage, type='storage')

    # add 1m connecting line to node the storage is connected to
    if isinstance(storage.grid, MVGrid):
        voltage_level = 'mv'
    else:
        voltage_level = 'lv'

    # necessary apparent power the line must be able to carry is set to be
    # the storages nominal power and equal amount of reactive power devided by
    # the minimum load factor
    lf_dict = storage.grid.network.config['grid_expansion_load_factors']
    lf = min(lf_dict['{}_feedin_case_line'.format(voltage_level)],
             lf_dict['{}_load_case_line'.format(voltage_level)])
    apparent_power_line = sqrt(2) * storage.nominal_power / lf
    line_type, line_count = select_cable(storage.grid.network, voltage_level,
                                         apparent_power_line)
    line = Line(
        id=storage.id,
        type=line_type,
        kind='cable',
        length=1e-3,
        grid=storage.grid,
        quantity=line_count)

    storage.grid.graph.add_edge(node, storage, line=line, type='line')

    return line


class StorageControl:
    """
    Integrates storages into the network.

    Parameters
    ----------
    edisgo : :class:`~.network.network.EDisGo`
    timeseries : :obj:`str` or :pandas:`pandas.Series<series>` or :obj:`dict`
        Parameter used to obtain time series of active power the
        storage(s) is/are charged (negative) or discharged (positive) with. Can
        either be a given time series or an operation strategy.
        Possible options are:

        * :pandas:`pandas.Series<series>`
          Time series the storage will be charged and discharged with can be
          set directly by providing a :pandas:`pandas.Series<series>` with
          time series of active charge (negative) and discharge (positive)
          power in kW. Index needs to be a
          :pandas:`pandas.DatetimeIndex<datetimeindex>`.
          If no nominal power for the storage is provided in
          `parameters` parameter, the maximum of the time series is
          used as nominal power.
          In case of more than one storage provide a :obj:`dict` where each
          entry represents a storage. Keys of the dictionary have to match
          the keys of the `parameters dictionary`, values must
          contain the corresponding time series as
          :pandas:`pandas.Series<series>`.
        * 'fifty-fifty'
          Storage operation depends on actual power of generators. If
          cumulative generation exceeds 50% of the nominal power, the storage
          will charge. Otherwise, the storage will discharge.
          If you choose this option you have to provide a nominal power for
          the storage. See `parameters` for more information.

        Default: None.
    position : None or :obj:`str` or :class:`~.network.components.Station` or :class:`~.network.components.BranchTee`  or :class:`~.network.components.Generator` or :class:`~.network.components.Load` or :obj:`dict`
        To position the storage a positioning strategy can be used or a
        node in the network can be directly specified. Possible options are:

        * 'hvmv_substation_busbar'
          Places a storage unit directly at the HV/MV station's bus bar.
        * :class:`~.network.components.Station` or :class:`~.network.components.BranchTee` or :class:`~.network.components.Generator` or :class:`~.network.components.Load`
          Specifies a node the storage should be connected to. In the case
          this parameter is of type :class:`~.network.components.LVStation` an
          additional parameter, `voltage_level`, has to be provided to define
          which side of the LV station the storage is connected to.
        * 'distribute_storages_mv'
          Places one storage in each MV feeder if it reduces network expansion
          costs. This method needs a given time series of active power.
          ToDo: Elaborate

        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` and `parameters`
        dictionaries, values must contain the corresponding positioning
        strategy or node to connect the storage to.
    parameters : :obj:`dict`, optional
        Dictionary with the following optional storage parameters:

        .. code-block:: python

            {
                'nominal_power': <float>, # in kW
                'max_hours': <float>, # in h
                'soc_initial': <float>, # in kWh
                'efficiency_in': <float>, # in per unit 0..1
                'efficiency_out': <float>, # in per unit 0..1
                'standing_loss': <float> # in per unit 0..1
            }

        See :class:`~.network.components.Storage` for more information on storage
        parameters.
        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` dictionary, values must
        contain the corresponding parameters dictionary specified above.
        Note: As edisgo currently only provides a power flow analysis storage
        parameters don't have any effect on the calculations, except of the
        nominal power of the storage.
        Default: {}.
    voltage_level : :obj:`str` or :obj:`dict`, optional
        This parameter only needs to be provided if any entry in `position` is
        of type :class:`~.network.components.LVStation`. In that case
        `voltage_level` defines which side of the LV station the storage is
        connected to. Valid options are 'lv' and 'mv'.
        In case of more than one storage provide a :obj:`dict` specifying the
        voltage level for each storage that is to be connected to an LV
        station. Keys of the dictionary have to match the keys of the
        `timeseries` dictionary, values must contain the corresponding
        voltage level.
        Default: None.
    timeseries_reactive_power : :pandas:`pandas.Series<series>` or :obj:`dict`
        By default reactive power is set through the config file
        `config_timeseries` in sections `reactive_power_factor` specifying
        the power factor and `reactive_power_mode` specifying if inductive
        or capacitive reactive power is provided.
        If you want to over-write this behavior you can provide a reactive
        power time series in kvar here. Be aware that eDisGo uses the generator
        sign convention for storages (see `Definitions and units` section of
        the documentation for more information). Index of the series needs to
        be a  :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` dictionary, values must contain the
        corresponding time series as :pandas:`pandas.Series<series>`.

    """

    def __init__(self, edisgo, timeseries, position, **kwargs):

        self.edisgo = edisgo
        voltage_level = kwargs.pop('voltage_level', None)
        parameters = kwargs.pop('parameters', {})
        timeseries_reactive_power = kwargs.pop(
            'timeseries_reactive_power', None)
        if isinstance(timeseries, dict):
            # check if other parameters are dicts as well if provided
            if voltage_level is not None:
                if not isinstance(voltage_level, dict):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              '`voltage_level` has to be provided as a ' \
                              'dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            if parameters is not None:
                if not all(isinstance(value, dict) == True
                           for value in parameters.values()):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              'storage parameters of each storage have to ' \
                              'be provided as a dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            if timeseries_reactive_power is not None:
                if not isinstance(timeseries_reactive_power, dict):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              '`timeseries_reactive_power` has to be ' \
                              'provided as a dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            for storage, ts in timeseries.items():
                try:
                    pos = position[storage]
                except KeyError:
                    message = 'Please provide position for storage {}.'.format(
                        storage)
                    logging.error(message)
                    raise KeyError(message)
                try:
                    voltage_lev = voltage_level[storage]
                except:
                    voltage_lev = None
                try:
                    params = parameters[storage]
                except:
                    params = {}
                try:
                    reactive_power = timeseries_reactive_power[storage]
                except:
                    reactive_power = None
                self._integrate_storage(ts, pos, params,
                                        voltage_lev, reactive_power, **kwargs)
        else:
            self._integrate_storage(timeseries, position, parameters,
                                    voltage_level, timeseries_reactive_power,
                                    **kwargs)

        # add measure to Results object
        self.edisgo.results.measures = 'storage_integration'

    def _integrate_storage(self, timeseries, position, params, voltage_level,
                           reactive_power_timeseries, **kwargs):
        """
        Integrate storage units in the network.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            Parameter used to obtain time series of active power the storage
            storage is charged (negative) or discharged (positive) with. Can
            either be a given time series or an operation strategy. See class
            definition for more information
        position : :obj:`str` or :class:`~.network.components.Station` or :class:`~.network.components.BranchTee` or :class:`~.network.components.Generator` or :class:`~.network.components.Load`
            Parameter used to place the storage. See class definition for more
            information.
        params : :obj:`dict`
            Dictionary with storage parameters for one storage. See class
            definition for more information on what parameters must be
            provided.
        voltage_level : :obj:`str` or None
            `voltage_level` defines which side of the LV station the storage is
            connected to. Valid options are 'lv' and 'mv'. Default: None. See
            class definition for more information.
        reactive_power_timeseries : :pandas:`pandas.Series<series>` or None
            Reactive power time series in kvar (generator sign convention).
            Index of the series needs to be a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        """
        # place storage
        params = self._check_nominal_power(params, timeseries)
        if isinstance(position, Station) or isinstance(position, BranchTee) \
                or isinstance(position, Generator) \
                or isinstance(position, Load):
            storage = set_up_storage(
                node=position, parameters=params, voltage_level=voltage_level)
            line = connect_storage(storage, position)
        elif isinstance(position, str) \
                and position == 'hvmv_substation_busbar':
            storage, line = storage_at_hvmv_substation(
                self.edisgo.network.mv_grid, params)
        elif isinstance(position, str) \
                and position == 'distribute_storages_mv':
            # check active power time series
            if not isinstance(timeseries, pd.Series):
                raise ValueError(
                    "Storage time series needs to be a pandas Series if "
                    "`position` is 'distribute_storages_mv'.")
            else:
                timeseries = pd.DataFrame(data={'p': timeseries},
                                          index=timeseries.index)
                self._check_timeindex(timeseries)
            # check reactive power time series
            if reactive_power_timeseries is not None:
                self._check_timeindex(reactive_power_timeseries)
                timeseries['q'] = reactive_power_timeseries.loc[
                    timeseries.index]
            else:
                timeseries['q'] = 0
            # start storage positioning method
            one_storage_per_feeder(
                edisgo=self.edisgo, storage_timeseries=timeseries,
                storage_nominal_power=params['nominal_power'], **kwargs)
            return
        else:
            message = 'Provided storage position option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

        # implement operation strategy (active power)
        if isinstance(timeseries, pd.Series):
            timeseries = pd.DataFrame(data={'p': timeseries},
                                      index=timeseries.index)
            self._check_timeindex(timeseries)
            storage.timeseries = timeseries
        elif isinstance(timeseries, str) and timeseries == 'fifty-fifty':
            fifty_fifty(self.edisgo.network, storage)
        else:
            message = 'Provided storage timeseries option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

        # reactive power
        if reactive_power_timeseries is not None:
            self._check_timeindex(reactive_power_timeseries)
            storage.timeseries = pd.DataFrame(
                {'p': storage.timeseries.p,
                 'q': reactive_power_timeseries.loc[storage.timeseries.index]},
                index=storage.timeseries.index)

        # update pypsa representation
        if self.edisgo.network.pypsa is not None:
            pypsa_io.update_pypsa_storage(
                self.edisgo.network.pypsa,
                storages=[storage], storages_lines=[line])

    def _check_nominal_power(self, storage_parameters, timeseries):
        """
        Tries to assign a nominal power to the storage.

        Checks if nominal power is provided through `storage_parameters`,
        otherwise tries to return the absolute maximum of `timeseries`. Raises
        an error if it cannot assign a nominal power.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            See parameter `timeseries` in class definition for more
            information.
        storage_parameters : :obj:`dict`
            See parameter `parameters` in class definition for more
            information.

        Returns
        --------
        :obj:`dict`
            The given `storage_parameters` is returned extended by an entry for
            'nominal_power', if it didn't already have that key.

        """
        if storage_parameters.get('nominal_power', None) is None:
            try:
                storage_parameters['nominal_power'] = max(abs(timeseries))
            except:
                raise ValueError("Could not assign a nominal power to the "
                                 "storage. Please provide either a nominal "
                                 "power or an active power time series.")
        return storage_parameters

    def _check_timeindex(self, timeseries):
        """
        Raises an error if time index of storage time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        timeseries : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power the storage is charged (negative)
            and discharged (positive) with in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        try:
            timeseries.loc[self.edisgo.network.timeseries.timeindex]
        except:
            message = 'Time index of storage time series does not match ' \
                      'with load and feed-in time series.'
            logging.error(message)
            raise KeyError(message)
