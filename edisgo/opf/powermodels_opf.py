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
    opt_flex=[],
    hv_req_p=pd.DataFrame(),
    hv_req_q=pd.DataFrame(),
    method="soc",
):

    opf_dir = os.path.dirname(os.path.abspath(__file__))
    julia_env_dir = os.path.join(opf_dir, "PowerModels.jl")
    solution_dir = os.path.join(opf_dir, "opf_solutions")

    # edisgo_obj: can be path to edisgo csv files, or edisgo obj
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
            json_str,
            pm["name"],
            solution_dir,
            method,
        ]
    )
    end = timer()
    run_time = end - start
    logger.info("julia terminated after {} s".format(run_time))

    if julia_process.returncode != 0:
        raise RuntimeError("Julia subprocess failed.")

    edisgo_obj.from_powermodels(os.path.join(solution_dir, pm["name"] + ".json"))
