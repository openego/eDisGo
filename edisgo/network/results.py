import os
import logging
import csv
import pandas as pd
import numpy as np

logger = logging.getLogger("edisgo")


def _get_matching_dict_of_attributes_and_file_names():
    """
    Helper function that matches attribute names to file names.

    Is used in functions :attr:`~.network.results.Results.to_csv`
    and :attr:`~.network.results.Results.from_csv` to set which attribute
    of :class:`~.network.results.Results` is saved under which file name.

    Returns
    -------
    tuple(dict, dict)
        Dictionaries matching attribute names and file names with attribute
        names as keys and corresponding file names as values. First dictionary
        matches power flow result attributes and second dictionary grid
        expansion result attributes.

    """
    powerflow_results_dict = {
        "v_res": "voltages_pu",
        "i_res": "currents",
        "pfa_p": "active_powers",
        "pfa_q": "reactive_powers",
        "s_res": "apparent_powers",
        "grid_losses": "grid_losses",
        "pfa_slack": "slack_results",
        "pfa_v_mag_pu_seed": "pfa_v_mag_pu_seed",
        "pfa_v_ang_seed": "pfa_v_ang_seed",
    }
    grid_expansion_results_dict = {
        "grid_expansion_costs": "grid_expansion_costs",
        "unresolved_issues": "unresolved_issues",
        "equipment_changes": "equipment_changes"
    }
    return powerflow_results_dict, grid_expansion_results_dict


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the network's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    edisgo_object : :class:`~.EDisGo`

    """

    def __init__(self, edisgo_object):

        self.edisgo_object = edisgo_object
        self._measures = ["original"]

    @property
    def measures(self):
        """
        List with measures conducted to increase network's hosting capacity.

        Parameters
        ----------
        measure : str
            Measure to increase network's hosting capacity. Possible options
            so far are 'grid_expansion', 'storage_integration', 'curtailment'.

        Returns
        -------
        list
            A stack that details the history of measures to increase network's
            hosting capacity. The last item refers to the latest measure. The
            key `original` refers to the state of the network topology as it
            was initially imported.

        """
        return self._measures

    @measures.setter
    def measures(self, measure):
        self._measures.append(measure)

    @property
    def pfa_p(self):
        """
        Active power over components in MW from last power flow analysis.

        The given active power for each line / transformer is the
        active power at the line ending / transformer side with the higher
        apparent power determined from active powers :math:`p_0` and
        :math:`p_1` and reactive powers :math:`q_0` and :math:`q_0` at the
        line endings / transformer sides:

        .. math::

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Results for active power over lines and transformers in MW from
            last power flow analysis. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            indicating the time period the power flow analysis was conducted
            for; columns of the dataframe are the representatives of the lines
            and stations included in the power flow analysis.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Results for active power over lines and transformers in MW from
            last power flow analysis. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._pfa_p
        except:
            return pd.DataFrame()

    @pfa_p.setter
    def pfa_p(self, df):
        self._pfa_p = df

    @property
    def pfa_q(self):
        """
        Active power over components in Mvar from last power flow analysis.

        The given reactive power over each line / transformer is the
        reactive power at the line ending / transformer side with the higher
        apparent power determined from active powers :math:`p_0` and
        :math:`p_1` and reactive powers :math:`q_0` and :math:`q_1` at the
        line endings / transformer sides:

        .. math::

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Results for reactive power over lines and transformers in Mvar from
            last power flow analysis. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            period the power flow analysis was conducted
            for; columns of the dataframe are the representatives of the lines
            and stations included in the power flow analysis.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Results for reactive power over lines and transformers in Mvar from
            last power flow analysis. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._pfa_q
        except:
            return pd.DataFrame()

    @pfa_q.setter
    def pfa_q(self, df):
        self._pfa_q = df

    @property
    def v_res(self):
        """
        Voltages at buses in p.u. from last power flow analysis.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with voltages at buses in p.u. from last power flow
            analysis. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the bus names of all buses in the analyzed grids.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with voltages at buses in p.u. from last power flow
            analysis. For more information on the dataframe see input
            parameter `df`.

        """
        try:
            return self._v_res
        except:
            return pd.DataFrame()

    @v_res.setter
    def v_res(self, df):
        self._v_res = df

    @property
    def i_res(self):
        """
        Current over components in kA from last power flow analysis.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Results for currents over lines and transformers in kA from last
            power flow analysis. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the representatives of the lines and stations
            included in the power flow analysis.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Results for current over lines and transformers in kA from
            last power flow analysis. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._i_res
        except:
            return pd.DataFrame()

    @i_res.setter
    def i_res(self, df):
        self._i_res = df

    @property
    def s_res(self):
        """
        Apparent power over components in MVA from last power flow analysis.

        The given apparent power over each line / transformer is the
        apparent power at the line ending / transformer side with the higher
        apparent power determined from active powers :math:`p_0` and
        :math:`p_1` and reactive powers :math:`q_0` and :math:`q_1` at the
        line endings / transformer sides:

        .. math::

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Apparent power in MVA over lines and transformers.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the representatives of the lines
            and stations included in the power flow analysis.

        """
        if self.pfa_p.empty or self.pfa_q.empty:
            return pd.DataFrame()

        return np.hypot(self.pfa_p, self.pfa_q)

    @property
    def equipment_changes(self):
        """
        Tracks changes to the grid topology.

        When the grid is reinforced using :attr:`~.EDisGo.reinforce` or new
        generators added using :attr:`~.EDisGo.import_generators`, new lines
        and/or transformers are added, lines split, etc. This is tracked in
        this attribute.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe holding information on added, changed and removed
            lines and transformers. Index of the dataframe is in case of
            lines the name of the line, and in case of transformers the name
            of the grid the station is in (in case of MV/LV transformers the
            name of the LV grid and in case of HV/MV transformers the name of
            the MV grid). Columns are the following:

            equipment : str
                Type of new line or transformer as in
                :attr:`~.network.topology.Topology.equipment_data`.

            change : str
                Specifies if something was added, changed or removed.

            iteration_step : int
                Grid reinforcement iteration step the change was conducted in.
                For changes conducted during grid integration of new generators
                the iteration step is set to 0.

            quantity : int
                Number of components added or removed. Only relevant for
                calculation of network expansion costs to keep track of how
                many new standard lines were added.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding information on added, changed and removed
            lines and transformers. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._equipment_changes
        except:
            return pd.DataFrame()

    @equipment_changes.setter
    def equipment_changes(self, df):
        self._equipment_changes = df

    @property
    def grid_expansion_costs(self):
        """
        Costs per expanded component in kEUR.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Costs per expanded line and transformer in kEUR.
            Index of the dataframe is the name of the expanded component as
            string. Columns are the following:

            type : str
                Type of new line or transformer as in
                :attr:`~.network.topology.Topology.equipment_data`.

            total_costs : float
                Costs of equipment in kEUR. For lines the line length and
                number of parallel lines is already included in the total
                costs.

            quantity : int
                For transformers quantity is always one, for lines it specifies
                the number of parallel lines.

            length : float
                Length of line or in case of parallel lines all lines in km.

            voltage_level : str
                Specifies voltage level the equipment is in ('lv', 'mv' or
                'mv/lv').

            Provide this if you want to set grid expansion costs. For retrieval
            of costs do not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Costs per expanded line and transformer in kEUR. For more
            information on the dataframe see input parameter `df`.

        Notes
        -------
        Network expansion measures are tracked in
        :attr:`~.network.results.Results.equipment_changes`. Resulting costs
        are calculated using :func:`~.flex_opt.costs.grid_expansion_costs`.
        Total network expansion costs can be obtained through
        grid_expansion_costs.total_costs.sum().

        """
        try:
            return self._grid_expansion_costs
        except:
            return pd.DataFrame()

    @grid_expansion_costs.setter
    def grid_expansion_costs(self, df):
        self._grid_expansion_costs = df

    @property
    def grid_losses(self):
        """
        Active and reactive network losses in MW and Mvar, respectively.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Results for active and reactive network losses in columns 'p'
            and 'q' and in MW and Mvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Results for active and reactive network losses MW and
            Mvar, respectively. For more information on the dataframe see
            input parameter `df`.

        Notes
        ------
        Grid losses are calculated as follows:

        .. math::

            P_{loss} = \lvert \sum{infeed} - \sum{load} + P_{slack} \lvert

        .. math::

            Q_{loss} = \lvert \sum{infeed} - \sum{load} + Q_{slack} \lvert

        As the slack is placed at the station's secondary side (if MV is
        included, it's positioned at the HV/MV station's secondary side and if
        a single LV grid is analysed it's positioned at the LV station's
        secondary side) losses do not include losses over the respective
        station's transformers.

        """
        try:
            return self._grid_losses
        except:
            return pd.DataFrame()

    @grid_losses.setter
    def grid_losses(self, df):
        self._grid_losses = df

    @property
    def pfa_slack(self):
        """
        Active and reactive power from slack in MW and Mvar, respectively.

        In case the MV level is included in the power flow analysis, the slack
        is placed at the secondary side of the HV/MV station and gives the
        energy transferred to and taken from the HV network. In case a single
        LV network is analysed, the slack is positioned at the respective
        station's secondary, in which case this gives the energy transferred
        to and taken from the overlying MV network.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Results for active and reactive power from the slack in MW and
            Mvar, respectively. Dataframe has the columns 'p', holding the
            active power results, and 'q', holding the reactive power results.
            Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Results for active and reactive power from the slack in MW and
            Mvar, respectively. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._pfa_slack
        except:
            return pd.DataFrame()

    @pfa_slack.setter
    def pfa_slack(self, df):
        self._pfa_slack = df

    @property
    def pfa_v_mag_pu_seed(self):
        """
        Voltages in p.u. from previous power flow analyses to be used as seed.

        See :func:`~.io.pypsa_io.set_seed` for more information.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Voltages at buses in p.u. from previous power flow analyses
            including the MV level. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating
            the time steps previous power flow analyses were conducted
            for; columns of the dataframe are the representatives of the buses
            included in the power flow analyses.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Voltages at buses in p.u. from previous power flow analyses to be
            opionally used as seed in following power flow analyses. For more
            information on the dataframe see input parameter `df`.

        """
        try:
            return self._pfa_v_mag_pu_seed
        except:
            return pd.DataFrame()

    @pfa_v_mag_pu_seed.setter
    def pfa_v_mag_pu_seed(self, df):
        self._pfa_v_mag_pu_seed = df

    @property
    def pfa_v_ang_seed(self):
        """
        Voltages in p.u. from previous power flow analyses to be used as seed.

        See :func:`~.io.pypsa_io.set_seed` for more information.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Voltage angles at buses in radians from previous power flow
            analyses including the MV level. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating
            the time steps previous power flow analyses were conducted
            for; columns of the dataframe are the representatives of the buses
            included in the power flow analyses.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Voltage angles at buses in radians from previous power flow
            analyses to be opionally used as seed in following power flow
            analyses. For more information on the dataframe see input
            parameter `df`.

        """
        try:
            return self._pfa_v_ang_seed
        except:
            return pd.DataFrame()

    @pfa_v_ang_seed.setter
    def pfa_v_ang_seed(self, df):
        self._pfa_v_ang_seed = df

    # @property
    # def curtailment(self):
    #     """
    #     Holds curtailment assigned to each generator per curtailment target.
    #
    #     ToDo: adapt to refactored code!
    #
    #     Returns
    #     -------
    #     :obj:`dict` with :pandas:`pandas.DataFrame<dataframe>`
    #         Keys of the dictionary are generator types (and weather cell ID)
    #         curtailment targets were given for. E.g. if curtailment is provided
    #         as a :pandas:`pandas.DataFrame<dataframe>` with
    #         :pandas.`pandas.MultiIndex` columns with levels 'type' and
    #         'weather cell ID' the dictionary key is a tuple of
    #         ('type','weather_cell_id').
    #         Values of the dictionary are dataframes with the curtailed power in
    #         kW per generator and time step. Index of the dataframe is a
    #         :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
    #         generators of type
    #         :class:`edisgo.network.components.GeneratorFluctuating`.
    #
    #     """
    #     raise NotImplementedError
    #     if self._curtailment is not None:
    #         result_dict = {}
    #         for key, gen_list in self._curtailment.items():
    #             curtailment_df = pd.DataFrame()
    #             for gen in gen_list:
    #                 curtailment_df[gen] = gen.curtailment
    #             result_dict[key] = curtailment_df
    #         return result_dict
    #     else:
    #         return None
    #
    # @property
    # def storage_units(self):
    #     """
    #     Gathers relevant storage results.
    #
    #     ToDo: adapt to refactored code!
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.DataFrame<dataframe>`
    #
    #         Dataframe containing all storage units installed in the MV and
    #         LV grids. Index of the dataframe are the storage representatives,
    #         columns are the following:
    #
    #         nominal_power : :obj:`float`
    #             Nominal power of the storage in kW.
    #
    #         voltage_level : :obj:`str`
    #             Voltage level the storage is connected to. Can either be 'mv'
    #             or 'lv'.
    #
    #     """
    #     raise NotImplementedError
    #
    #     grids = [self.edisgo_object.topology.mv_grid] + list(
    #         self.edisgo_object.topology.mv_grid.lv_grids
    #     )
    #     storage_results = {}
    #     storage_results["storage_id"] = []
    #     storage_results["nominal_power"] = []
    #     storage_results["voltage_level"] = []
    #     storage_results["grid_connection_point"] = []
    #     for grid in grids:
    #         for storage in grid.graph.nodes_by_attribute("storage"):
    #             storage_results["storage_id"].append(repr(storage))
    #             storage_results["nominal_power"].append(storage.nominal_power)
    #             storage_results["voltage_level"].append(
    #                 "mv" if isinstance(grid, MVGrid) else "lv"
    #             )
    #             storage_results["grid_connection_point"].append(
    #                 list(grid.graph.neighbors(storage))[0]
    #             )
    #
    #     return pd.DataFrame(storage_results).set_index("storage_id")
    #
    #
    # @property
    # def storage_units_costs_reduction(self):
    #     """
    #     Contains costs reduction due to storage integration.
    #
    #     ToDo: adapt to refactored code!
    #
    #     Parameters
    #     ----------
    #     costs_df : :pandas:`pandas.DataFrame<dataframe>`
    #         Dataframe containing network expansion costs in kEUR before and after
    #         storage integration in columns 'grid_expansion_costs_initial' and
    #         'grid_expansion_costs_with_storage_units', respectively. Index of
    #         the dataframe is the MV network id.
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.DataFrame<dataframe>`
    #
    #         Dataframe containing network expansion costs in kEUR before and after
    #         storage integration in columns 'grid_expansion_costs_initial' and
    #         'grid_expansion_costs_with_storage_units', respectively. Index of
    #         the dataframe is the MV network id.
    #
    #     """
    #     raise NotImplementedError
    #     return self._storage_units_costs_reduction
    #
    # @storage_units_costs_reduction.setter
    # def storage_units_costs_reduction(self, costs_df):
    #     self._storage_units_costs_reduction = costs_df

    @property
    def unresolved_issues(self):
        """
        Lines and buses with remaining grid issues after network reinforcement.

        In case overloading or voltage issues could not be solved
        after maximum number of iterations, network reinforcement is not
        aborted but network expansion costs are still calculated and unresolved
        issues listed here.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`

            Dataframe containing remaining grid issues. Names of remaining
            critical lines, stations and buses are in the index of the
            dataframe. Columns depend on the equipment type. See
            :func:`~.flex_opt.check_tech_constraints.mv_line_load` for format
            of remaining overloading issues of lines,
            :func:`~.flex_opt.check_tech_constraints.hv_mv_station_load`
            for format of remaining overloading issues of transformers, and
            :func:`~.flex_opt.check_tech_constraints.mv_voltage_deviation`
            for format of remaining voltage issues.

            Provide this if you want to set unresolved_issues. For retrieval
            of data do not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with remaining grid issues. For more information on the
            dataframe see input parameter `df`.

        """
        try:
            return self._unresolved_issues
        except:
            return pd.DataFrame()

    @unresolved_issues.setter
    def unresolved_issues(self, df):
        self._unresolved_issues = df

    def _add_line_to_equipment_changes(self, line):
        """
        Adds new line to equipment changes.

        All changes of equipment are stored in
        :attr:`~.network.results.Results.equipment_changes`
        which is used later on to determine network expansion costs.

        Parameters
        -----------
        line : :pandas:`pandas.Series<Series>`
            Series with data of line to add. Series has same rows as columns
            of :attr:`~.network.topology.Topology.lines_df`, but must at least
            contain `type_info`. Line representative is the series name.

        """
        self.equipment_changes = \
            self.equipment_changes.append(
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

    def _del_line_from_equipment_changes(self, line_repr):
        """
        Deletes line from equipment changes if it exists.

        This is needed when a line was already added to
        :attr:`~.network.results.Results.equipment_changes` but another
        component is later connected to this line. Therefore, the line needs
        to be split which changes the representative of the line and the line
        data.

        Parameters
        -----------
        line_repr : str
            Line representative as in index of
            :attr:`~.network.topology.Topology.lines_df`.

        """
        if line_repr in self.equipment_changes.index:
            self.equipment_changes = \
                self.equipment_changes.drop(
                    line_repr
                )

    def reduce_memory(self, attr_to_reduce=None, to_type="float32"):
        """
        Reduces size of dataframes containing time series to save memory.

        See :attr:`~.EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Attributes need to be
            dataframes containing only time series. Possible options are:
            'pfa_p', 'pfa_q', 'v_res', 'i_res', and 'grid_losses'.
            Per default, all these attributes are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        Notes
        ------
        Reducing the data type of the seeds for the power flow analysis,
        :py:attr:`~pfa_v_mag_pu_seed` and :py:attr:`~pfa_v_ang_seed`, can lead
        to non-convergence of the power flow analysis, wherefore memory
        reduction is not provided for those attributes.

        """
        if attr_to_reduce is None:
            attr_to_reduce = [
                "pfa_p", "pfa_q",
                "v_res", "i_res",
                "grid_losses"
            ]
        for attr in attr_to_reduce:
            setattr(
                self,
                attr,
                getattr(self, attr).apply(
                    lambda _: _.astype(to_type)
                )
            )

    def to_csv(self, directory, parameters=None, reduce_memory=False,
               save_seed=False, **kwargs):
        """
        Saves results to csv.

        Saves power flow results and grid expansion results to separate
        directories. Which results are saved depends on what is specified in
        `parameters`. Per default, all attributes are saved.

        Power flow results are saved to directory 'powerflow_results' and
        comprise the following, if not otherwise specified:

        * 'v_res' : Attribute :py:attr:`~v_res` is saved to
          `voltages_pu.csv`.
        * 'i_res' : Attribute :py:attr:`~i_res` is saved to
          `currents.csv`.
        * 'pfa_p' : Attribute :py:attr:`~pfa_p` is saved to
          `active_powers.csv`.
        * 'pfa_q' : Attribute :py:attr:`~pfa_q` is saved to
          `reactive_powers.csv`.
        * 's_res' : Attribute :py:attr:`~s_res` is saved to
          `apparent_powers.csv`.
        * 'grid_losses' : Attribute :py:attr:`~grid_losses` is saved to
          `grid_losses.csv`.
        * 'pfa_slack' : Attribute :py:attr:`~pfa_slack` is saved to
          `pfa_slack.csv`.
        * 'pfa_v_mag_pu_seed' : Attribute :py:attr:`~pfa_v_mag_pu_seed` is
          saved to `pfa_v_mag_pu_seed.csv`, if `save_seed` is set to True.
        * 'pfa_v_ang_seed' : Attribute :py:attr:`~pfa_v_ang_seed` is
          saved to `pfa_v_ang_seed.csv`, if `save_seed` is set to True.

        Grid expansion results are saved to directory 'grid_expansion_results'
        and comprise the following, if not otherwise specified:

        * grid_expansion_costs : Attribute :py:attr:`~grid_expansion_costs`
          is saved to `grid_expansion_costs.csv`.
        * equipment_changes : Attribute :py:attr:`~equipment_changes`
          is saved to `equipment_changes.csv`.
        * unresolved_issues : Attribute :py:attr:`~unresolved_issues`
          is saved to `unresolved_issues.csv`.

        Parameters
        ----------
        directory : str
            Main directory to save the results in.
        parameters : None or dict, optional
            Specifies which results to save. By default this is set to None,
            in which case all results are saved.
            To only save certain results provide a dictionary. Possible keys
            are 'powerflow_results' and 'grid_expansion_results'. Corresponding
            values must be lists with attributes to save or None to save all
            attributes. For example, with the first input only the power flow
            results `i_res` and `v_res` are saved, and with the second input
            all power flow results are saved.

            .. code-block:: python

                {'powerflow_results': ['i_res', 'v_res']}

            .. code-block:: python

                {'powerflow_results': None}

            See function docstring for possible power flow and grid expansion
            results to save and under which file name they are saved.

        reduce_memory : bool, optional
            If True, size of dataframes containing time series to save memory
            is reduced using :attr:`~.network.results.Results.reduce_memory`.
            Optional parameters of
            :attr:`~.network.results.Results.reduce_memory` can be passed as
            kwargs to this function. Default: False.
        save_seed : bool, optional
            If True, :py:attr:`~pfa_v_mag_pu_seed` and
            :py:attr:`~pfa_v_ang_seed` are as well saved as csv. As these are
            only relevant if calculations are not final, the default is False,
            in which case they are not saved.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain optional arguments of
            :attr:`~.network.results.Results.reduce_memory`.

        """

        def _save_power_flow_results(target_dir, save_attributes):
            # create directory
            os.makedirs(target_dir, exist_ok=True)

            if save_attributes is None:
                save_attributes = list(power_flow_results_dict.keys())

            for attr in save_attributes:
                if not getattr(self, attr).empty:
                    getattr(self, attr).to_csv(
                        os.path.join(target_dir, "{}.csv".format(
                            power_flow_results_dict[attr]))
                    )

        def _save_grid_expansion_results(target_dir, save_attributes):
            # create directory
            os.makedirs(target_dir, exist_ok=True)

            if save_attributes is None:
                save_attributes = list(grid_expansion_results_dict.keys())

            for attr in save_attributes:
                if not getattr(self, attr).empty:
                    getattr(self, attr).to_csv(
                        os.path.join(target_dir, "{}.csv".format(
                            grid_expansion_results_dict[attr]
                        ))
                    )

        def _save_curtailment_results(target_dir):
            pass
            # if self.curtailment is not None:
            #     # create directory
            #     os.makedirs(target_dir, exist_ok=True)
            #
            #     for key, curtailment_df in self.curtailment.items():
            #         if type(key) == tuple:
            #             type_prefix = '-'.join([key[0], str(key[1])])
            #         elif type(key) == str:
            #             type_prefix = key
            #         else:
            #             raise KeyError("Unknown key type {} for key {}".format(
            #                 type(key), key))
            #
            #         filename = os.path.join(
            #             target_dir, '{}.csv'.format(type_prefix))
            #
            #         curtailment_df.to_csv(filename, index_label=type_prefix)

        def _save_storage_integration_results(target_dir):
            pass
            # storages = self.storages
            # if not storages.empty:
            #     # create directory
            #     os.makedirs(target_dir, exist_ok=True)
            #
            #     # general storage information
            #     storages.to_csv(os.path.join(target_dir, 'storages.csv'))
            #
            #     # storages time series
            #     ts_p, ts_q = self.storages_timeseries()
            #     ts_p.to_csv(os.path.join(
            #         target_dir, 'storages_active_power.csv'))
            #     ts_q.to_csv(os.path.join(
            #         target_dir, 'storages_reactive_power.csv'))
            #
            #     if not self.storages_costs_reduction is None:
            #         self.storages_costs_reduction.to_csv(
            #             os.path.join(target_dir,
            #                          'storages_costs_reduction.csv'))

        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        # dictionary with function to call to save each parameter
        func_dict = {
            "powerflow_results": _save_power_flow_results,
            "grid_expansion_results": _save_grid_expansion_results
        }

        # get dictionaries matching attribute names and file names
        power_flow_results_dict, grid_expansion_results_dict = \
            _get_matching_dict_of_attributes_and_file_names()

        # if None, set to save all attributes
        if parameters is None:
            parameters = {
                "powerflow_results": list(power_flow_results_dict.keys()),
                "grid_expansion_results": list(
                    grid_expansion_results_dict.keys())
            }
            if not save_seed:
                parameters["powerflow_results"] = [
                    _ for _ in parameters["powerflow_results"]
                    if not "seed" in _
                ]

        if not isinstance(parameters, dict):
            raise ValueError(
                "Invalid input for `parameters` when saving "
                "results to csv. `parameters` must be a dictionary. "
                "See docstring for more information.")

        # iterate over dictionary to save power flow results, etc. to csv
        # depending on what is specified in parameters
        for k, v in parameters.items():
            try:
                func_dict[k](os.path.join(directory, k), v)
            except KeyError:
                message = (
                    "Invalid input for `parameters` when saving "
                    "results to csv. See docstring for possible options."
                )
                logger.error(message)
                raise KeyError(message)
            except:
                raise

        # save measures
        pd.DataFrame(data={"measure": self.measures}).to_csv(
            os.path.join(directory, "measures.csv")
        )
        # save configs
        with open(os.path.join(directory, "configs.csv"), "w") as f:
            writer = csv.writer(f)
            rows = [
                ["{}".format(key)]
                + [value for item in values.items() for value in item]
                for key, values in self.edisgo_object.config._data.items()
            ]
            writer.writerows(rows)

    def from_csv(self, directory, parameters=None):
        """
        Restores results from csv files.

        See :func:`~to_csv` for more information on which results can be saved
        and under which filename and directory they are stored.

        Parameters
        ----------
        directory : str
            Main directory results are saved in.
        parameters : None or dict, optional
            Specifies which results to restore. By default this is set to None,
            in which case all available results are restored.
            To only restore certain results provide a dictionary. Possible keys
            are 'powerflow_results' and 'grid_expansion_results'. Corresponding
            values must be lists with attributes to restore or None to restore
            all available attributes. See function docstring `parameters`
            parameter in :func:`~to_csv` for more information.

        """
        # restore measures
        if os.path.exists(os.path.join(directory, "measures.csv")):
            measures_df = pd.read_csv(
                os.path.join(directory, 'measures.csv'),
                index_col=0)
            self._measures = list(measures_df.measure.values)

        # get dictionaries matching attribute names and file names
        power_flow_results_dict, grid_expansion_results_dict = \
            _get_matching_dict_of_attributes_and_file_names()

        # if None, set to restore all attributes
        if parameters is None:
            parameters = {
                "powerflow_results": list(power_flow_results_dict.keys()),
                "grid_expansion_results": list(
                    grid_expansion_results_dict.keys())
            }

        if not isinstance(parameters, dict):
            raise ValueError(
                "Invalid input for `parameters` when restoring "
                "results from csv. `parameters` must be a dictionary. "
                "See docstring for more information.")

        # import power flow results
        if 'powerflow_results' in list(parameters.keys()) and \
                os.path.isdir(os.path.join(directory, 'powerflow_results')):
            for attr in parameters["powerflow_results"]:
                path = os.path.join(
                            directory,
                            'powerflow_results',
                            '{}.csv'.format(power_flow_results_dict[attr])
                        )
                if os.path.exists(path):
                    setattr(
                        self,
                        attr,
                        pd.read_csv(path, index_col=0, parse_dates=True)
                    )

        # import grid expansion results
        if 'grid_expansion_results' in list(parameters.keys()) and \
                os.path.isdir(
                    os.path.join(directory, 'grid_expansion_results')):
            for attr in parameters["grid_expansion_results"]:
                path = os.path.join(
                            directory,
                            'grid_expansion_results',
                            '{}.csv'.format(grid_expansion_results_dict[attr])
                        )
                if os.path.exists(path):
                    setattr(
                        self,
                        attr,
                        pd.read_csv(path, index_col=0)
                    )

        # # import curtailment results
        # if 'curtailment_results' in parameters and os.path.isdir(
        #         os.path.join(results_path, 'curtailment_results')):
        #     self.curtailment = {}
        #     for file in os.listdir(os.path.join(
        #             results_path, 'curtailment_results')):
        #         if file.endswith(".csv"):
        #             try:
        #                 key = file[0:-4]
        #                 if '-' in key:
        #                     # make tuple if curtailment was given for generator
        #                     # type and weather cell id
        #                     tmp = key.split('-')
        #                     key = (tmp[0], float(tmp[1]))
        #                 self.curtailment[key] = pd.read_csv(
        #                     os.path.join(
        #                         results_path, 'curtailment_results', file),
        #                     index_col=0, parse_dates=True)
        #             except Exception as e:
        #                 logging.warning(
        #                     'The following error occured when trying to '
        #                     'import curtailment results: {}'.format(e))
        # else:
        #     self.curtailment = None

        # # import storage results
        # if 'storage_integration_results' in parameters and os.path.isdir(
        #         os.path.join(results_path, 'storage_integration_results')):
        #     # storages
        #     self.storages = pd.read_csv(
        #         os.path.join(results_path, 'storage_integration_results',
        #                      'storages.csv'),
        #         index_col=0)
        #     # storages costs reduction
        #     try:
        #         self.storages_costs_reduction = pd.read_csv(
        #             os.path.join(
        #                 results_path, 'storage_integration_results',
        #                 'storages_costs_reduction.csv'),
        #             index_col=0)
        #     except:
        #         pass
        #     # storages time series
        #     self.storages_p = pd.read_csv(
        #         os.path.join(
        #             results_path, 'storage_integration_results',
        #             'storages_active_power.csv'),
        #         index_col=0, parse_dates=True)
        #     # storages time series
        #     self.storages_q = pd.read_csv(
        #         os.path.join(
        #             results_path, 'storage_integration_results',
        #             'storages_reactive_power.csv'),
        #         index_col=0, parse_dates=True)
        #
        # else:
        #     self.storages = None
        #     self.storages_costs_reduction = None
        #     self.storages_p = None
        #     self.storages_q = None
