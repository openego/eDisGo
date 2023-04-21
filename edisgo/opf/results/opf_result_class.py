import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LineVariables:
    def __init__(self):
        self.p = pd.DataFrame()
        self.q = pd.DataFrame()
        self.ccm = pd.DataFrame()


class BusVariables:
    def __init__(self):
        self.w = pd.DataFrame()


class HeatStorage:
    def __init__(self):
        self.p = pd.DataFrame()
        self.e = pd.DataFrame()


class GridSlacks:
    def __init__(self):
        self.gen_d_crt = pd.DataFrame()
        self.gen_nd_crt = pd.DataFrame()
        self.load_shedding = pd.DataFrame()
        self.cp_load_shedding = pd.DataFrame()
        self.hp_load_shedding = pd.DataFrame()


class OPFResults:
    def __init__(self):
        self.status = None
        self.solution_time = None
        self.solver = None
        self.buses_t = BusVariables()
        self.lines_t = LineVariables()
        self.slack_generator_t = pd.DataFrame()
        self.heat_storage_t = HeatStorage()
        self.hv_requirement_slacks_t = pd.DataFrame()
        self.grid_slacks_t = GridSlacks()
