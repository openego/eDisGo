import logging
import os

from zipfile import ZipFile

import pandas as pd

logger = logging.getLogger(__name__)


class LineVariables:
    def __init__(self):
        self.p = pd.DataFrame()
        self.q = pd.DataFrame()
        self.ccm = pd.DataFrame()

    def _attributes(self):
        return ["p", "q", "ccm"]


class HeatStorage:
    def __init__(self):
        self.p = pd.DataFrame()
        self.e = pd.DataFrame()
        self.p_slack = pd.DataFrame()

    def _attributes(self):
        return ["p", "e", "p_slack"]


class BatteryStorage:
    def __init__(self):
        self.p = pd.DataFrame()
        self.e = pd.DataFrame()

    def _attributes(self):
        return ["p", "e"]


class GridSlacks:
    def __init__(self):
        self.gen_d_crt = pd.DataFrame()
        self.gen_nd_crt = pd.DataFrame()
        self.load_shedding = pd.DataFrame()
        self.cp_load_shedding = pd.DataFrame()
        self.hp_load_shedding = pd.DataFrame()
        self.hp_operation_slack = pd.DataFrame()

    def _attributes(self):
        return [
            "gen_d_crt",
            "gen_nd_crt",
            "load_shedding",
            "cp_load_shedding",
            "hp_load_shedding",
            "hp_operation_slack",
        ]


class OPFResults:
    def __init__(self):
        self.status = None
        self.solution_time = None
        self.solver = None
        self.lines_t = LineVariables()
        self.slack_generator_t = pd.DataFrame()
        self.heat_storage_t = HeatStorage()
        self.hv_requirement_slacks_t = pd.DataFrame()
        self.grid_slacks_t = GridSlacks()
        self.overlying_grid = pd.DataFrame()
        self.battery_storage_t = BatteryStorage()

    def to_csv(self, directory, attributes=None):
        """
        Exports OPF results data to csv files.

        The following attributes can be exported:

        * 'lines_t' : The results of the three variables in attribute
          :py:attr:`~lines_t` are saved to `lines_t_p.csv`, `lines_t_p.csv`, and
          `lines_t_ccm.csv`.
        * 'slack_generator_t' : Attribute :py:attr:`~slack_generator_t` is saved to
          `slack_generator_t.csv`.
        * 'heat_storage_t' : The results of the two variables in attribute
          :py:attr:`~heat_storage_t` are saved to `heat_storage_t_p.csv` and
          `heat_storage_t_e.csv`.
        * 'hv_requirement_slacks_t' : Attribute :py:attr:`~hv_requirement_slacks_t` is
          saved to `hv_requirement_slacks_t.csv`.
        * 'grid_slacks_t' : The results of the five variables in attribute
          :py:attr:`~grid_slacks_t` are saved to `dispatchable_gen_crt.csv`,
          `non_dispatchable_gen_crt.csv`, `load_shedding.csv`, `cp_load_shedding.csv`
          and `hp_load_shedding.csv`.
        * 'overlying_grid' : Attribute :py:attr:`~overlying_grid` is saved to
          `overlying_grid.csv`.

        Parameters
        ----------
        directory : str
            Path to save OPF results data to.
        attributes : list(str) or None
            List of attributes to export. See above for attributes that can be exported.
            If None, all specified attributes are exported. Default: None.

        """
        os.makedirs(directory, exist_ok=True)

        attrs_file_names = _get_matching_dict_of_attributes_and_file_names()

        if attributes is None:
            attributes = list(attrs_file_names.keys())

        for attr in attributes:
            file = attrs_file_names[attr]
            df = getattr(self, attr)
            if attr in [
                "lines_t",
                "heat_storage_t",
                "grid_slacks_t",
                "battery_storage_t",
            ]:
                for variable in file.keys():
                    if variable in df._attributes() and not getattr(df, variable).empty:
                        path = os.path.join(directory, file[variable])
                        getattr(df, variable).to_csv(path)
            else:
                if not df.empty:
                    path = os.path.join(directory, file)
                    df.to_csv(path)

    def from_csv(self, data_path, from_zip_archive=False):
        """
        Restores OPF results from csv files.

        Parameters
        ----------
        data_path : str
            Path to OPF results csv files.
        from_zip_archive : bool, optional
            Set True if data is archived in a zip archive. Default: False.

        """
        attrs = _get_matching_dict_of_attributes_and_file_names()

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {
                k: (
                    f"opf_results/{v}"
                    if isinstance(v, str)
                    else {k2: f"opf_results/{v2}" for k2, v2 in v.items()}
                )
                for k, v in attrs.items()
            }

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

        attrs_to_read = {
            k: v
            for k, v in attrs.items()
            if (isinstance(v, str) and v in files)
            or (isinstance(v, dict) and any([_ in files for _ in v.values()]))
        }

        for attr, file in attrs_to_read.items():
            if attr in [
                "lines_t",
                "heat_storage_t",
                "grid_slacks_t",
                "battery_storage_t",
            ]:
                for variable, file_name in file.items():
                    if file_name in files:
                        if from_zip_archive:
                            # open zip file to make it readable for pandas
                            with zip.open(file_name) as f:
                                setattr(
                                    getattr(self, attr),
                                    variable,
                                    pd.read_csv(f, index_col=0, parse_dates=True),
                                )
                        else:
                            path = os.path.join(data_path, file_name)
                            setattr(
                                getattr(self, attr),
                                variable,
                                pd.read_csv(path, index_col=0, parse_dates=True),
                            )
            else:
                if from_zip_archive:
                    # open zip file to make it readable for pandas
                    with zip.open(file) as f:
                        df = pd.read_csv(f, index_col=0, parse_dates=True)
                else:
                    path = os.path.join(data_path, file)
                    df = pd.read_csv(path, index_col=0, parse_dates=True)

                setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()


def _get_matching_dict_of_attributes_and_file_names():
    """
    Helper function to specify which OPF results attributes to save and
    restore and maps them to the file name.

    Is used in functions
    :attr:`~.opf.results.opf_results_class.from_csv` and
    :attr:`~.opf.results.opf_results_class.to_csv`.

    Returns
    -------
    dict
        Dict of OPF results attributes to save and restore as keys and matching files as
        values.

    """
    opf_results_dict = {
        "slack_generator_t": "slack_generator_t.csv",
        "hv_requirement_slacks_t": "hv_requirement_slacks_t.csv",
        "overlying_grid": "overlying_grid.csv",
        "lines_t": {
            "p": "lines_t_p.csv",
            "q": "lines_t_q.csv",
            "ccm": "lines_t_ccm.csv",
        },
        "heat_storage_t": {
            "p": "heat_storage_t_p.csv",
            "e": "heat_storage_t_e.csv",
            "p_slack": "heat_storage_t_p_slack.csv",
        },
        "battery_storage_t": {
            "p": "battery_storage_t_p.csv",
            "e": "battery_storage_t_e.csv",
        },
        "grid_slacks_t": {
            "gen_d_crt": "dispatchable_gen_crt.csv",
            "gen_nd_crt": "non_dispatchable_gen_crt.csv",
            "load_shedding": "load_shedding.csv",
            "cp_load_shedding": "cp_load_shedding.csv",
            "hp_load_shedding": "hp_load_shedding.csv",
            "hp_operation_slack": "hp_operation_slack.csv",
        },
    }

    return opf_results_dict
