import json
import logging
import os
import subprocess

from timeit import default_timer as timer

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pm_optimize(
    edisgo_obj,
    flexible_cps=[],
    flexible_hps=[],
    flexible_loads=[],
    opt_version=1,
    opt_flex=["curt", "storage", "cp", "hp", "dsm"],
    hv_req_p=pd.DataFrame(),
    hv_req_q=pd.DataFrame(),
    method="soc",
):
    """
    Runs OPF for edisgo object in julia subprocess and writes results of OPF to edisgo
    object. Results of OPF are time series of operation schedules of flexibilities.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    opt_version: Int
        Version of optimization models to choose from. For more information see MA.
        Must be one of [1, 2, 3].
    opt_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    method: str
        Optimization method to use. Must be either "soc" (Second Order Cone) or "nc"
        (Non Convex).
        If method is "soc", OPF is run in PowerModels with Gurobi solver with SOC
        relaxation of equality constraint P²+Q² = V²*I². If method is "nc", OPF is run
        with Ipopt solver as a non-convex problem due to quadratic equality constraint
        P²+Q² = V²*I².
    """

    opf_dir = os.path.dirname(os.path.abspath(__file__))
    julia_env_dir = os.path.join(opf_dir, "PowerModels.jl")
    solution_dir = os.path.join(opf_dir, "opf_solutions")

    pm = edisgo_obj.to_powermodels(
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        opt_version=opt_version,
        opt_flex=opt_flex,
        hv_req_p=hv_req_p,
        hv_req_q=hv_req_q,
    )

    def _convert(o):
        """
        Helper function for json dump, as int64 cannot be dumped.

        """
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    json_str = json.dumps(pm, default=_convert)

    logger.info("starting julia process")
    start = timer()
    julia_process = subprocess.run(
        [
            "julia",
            "--project={}".format(julia_env_dir),
            os.path.join(opf_dir, "PowerModels.jl/Main.jl"),
            pm["name"],
            solution_dir,
            method,
        ],
        input=json_str,
        text=True,
        capture_output=True,
    )
    end = timer()
    run_time = end - start
    logger.info("julia terminated after {} s".format(run_time))
    if julia_process.returncode != 0:
        raise RuntimeError("Julia subprocess failed.")

    pm_opf = json.loads(julia_process.stdout.split("\n")[-1])
    # write results to edisgo object
    edisgo_obj.from_powermodels(pm_opf)
