import json
import logging
import os
import subprocess
import sys

import numpy as np

# logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def pm_optimize(
    edisgo_obj,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    opt_version=1,
    opt_flex=None,
    method="soc",
    solver_tol=1e-6,
    silence_moi=False,
    save_heat_storage=False,
    save_gen_slack=False,
    save_hv_slack=False,
    path="",
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
    silence_moi: bool
        If set to True, MathOptInterface's optimizer attribute "MOI.Silent" is set
        to True in julia subprocess. This attribute is for silencing the output of
        an optimizer. When set to True, it requires the solver to produce no output,
        hence there will be no logging coming from julia subprocess in python
        process.
    save_heat_storage: bool
        Indicates whether to save results of heat storage variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        directory.
            Default: False
    save_gen_slack: bool
        Indicates whether to save results of slack generator variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        Default: False
    save_hv_slack: bool
        Indicates whether to save results of slack variables for high voltage
        requirements (sum, minimal and maximal and mean deviation) from the optimization
        to csv file in the current working directory. Set parameter "path" to change the
        directory the file is saved to.
        Default: False
    path : str
        Directory the csv file is saved to. Per default it takes the current
        working directory.
    """
    opf_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(opf_dir, "opf_solutions")

    pm, hv_flex_dict = edisgo_obj.to_powermodels(
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        opt_version=opt_version,
        opt_flex=opt_flex,
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
    julia_process = subprocess.Popen(
        [
            "julia",
            os.path.join(opf_dir, "PowerModels.jl/Main.jl"),
            pm["name"],
            solution_dir,
            method,
            str(silence_moi),
            str(solver_tol),
        ],
        stdin=subprocess.PIPE,
        text=True,
        stdout=subprocess.PIPE,
    )
    julia_process.stdin.write(json_str)
    julia_process.stdin.close()
    while True:
        out = julia_process.stdout.readline()
        if out == "" and julia_process.poll() is not None:
            break
        if out.rstrip().startswith('{"name"'):
            logger.info("Julia process was successful.")
            pm_opf = json.loads(out)
            # write results to edisgo object
            edisgo_obj.from_powermodels(
                pm_opf,
                hv_flex_dict,
                save_heat_storage=save_heat_storage,
                save_gen_slack=save_gen_slack,
                save_hv_slack=save_hv_slack,
                path=path,
            )
        elif out.rstrip().startswith("Set parameter") or out.rstrip().startswith(
            "Academic"
        ):
            continue
        elif out != "":
            sys.stdout.write(out)
            sys.stdout.flush()
