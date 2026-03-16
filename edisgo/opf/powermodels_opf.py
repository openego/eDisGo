import json
import logging
import os
import shutil
import subprocess
import sys

from typing import Optional

import numpy as np

from edisgo.flex_opt import exceptions
from edisgo.io.powermodels_io import from_powermodels
from edisgo.network.topology import Topology

logger = logging.getLogger(__name__)


def find_julia_binary():
    """
    Find suitable Julia binary with version >= 1.6.
    
    Tries in order:
    1. julia1.11 (if available)
    2. julia (if version >= 1.6)
    3. Raises error if no suitable Julia found
    
    Returns
    -------
    str
        Path or name of Julia binary
    """
    # Try julia1.11 first (preferred for eDisGo)
    julia1_11_path = shutil.which("julia1.11")
    if julia1_11_path:
        try:
            result = subprocess.run(
                [julia1_11_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.debug(f"Using Julia: {result.stdout.strip()}")
                return julia1_11_path
        except Exception as e:
            logger.warning(f"julia1.11 found but failed to execute: {e}")
    
    # Try default julia
    julia_path = shutil.which("julia")
    if julia_path:
        try:
            result = subprocess.run(
                [julia_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                # Extract version number (e.g., "julia version 1.8.5")
                version_parts = version_str.split()
                if len(version_parts) >= 3:
                    version = version_parts[2]
                    major_minor = tuple(map(int, version.split('.')[:2]))
                    if major_minor >= (1, 6):
                        logger.debug(f"Using Julia: {version_str}")
                        return julia_path
                    else:
                        logger.warning(
                            f"Found Julia {version} but eDisGo requires >= 1.6"
                        )
        except Exception as e:
            logger.warning(f"julia found but failed to execute: {e}")
    
    # No suitable Julia found
    raise RuntimeError(
        "No suitable Julia installation found. "
        "eDisGo requires Julia >= 1.6. "
        "Please install Julia 1.11 or later from https://julialang.org/downloads/"
    )


def pm_optimize(
    edisgo_obj,
    s_base: int = 1,
    flexible_cps: Optional[np.ndarray] = None,
    flexible_hps: Optional[np.ndarray] = None,
    flexible_loads: Optional[np.ndarray] = None,
    flexible_storage_units: Optional[np.ndarray] = None,
    opf_version: int = 1,
    method: str = "soc",
    warm_start: bool = False,
    silence_moi: bool = False,
) -> None:
    """
    Run OPF for edisgo object in julia subprocess and write results of OPF to edisgo
    object. Results of OPF are time series of operation schedules of flexibilities.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 1 MVA.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all charging points that allow for flexible charging.
        Default: None.
    flexible_hps : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
        Default: None.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all flexible loads that allow for application of demand side
        management strategy.
        Default: None.
    flexible_storage_units : :numpy:`numpy.ndarray<ndarray>` or None
        Array containing all flexible storage units. Non-flexible storage units operate
        to optimize self consumption.
        Default: None
    opf_version : int
        Version of optimization models to choose from. The grid model is a radial branch
        flow model (BFM). Optimization versions differ in lifted or additional
        constraints and the objective function.
        Implemented versions are:

        * 1
            * Lifted constraints: grid restrictions
            * Objective: minimize line losses and maximal line loading
        * 2
            * Objective: minimize line losses and grid related slacks
        * 3
            * Additional constraints: high voltage requirements
            * Lifted constraints: grid restrictions
            * Objective: minimize line losses, maximal line loading and HV slacks
        * 4
            * Additional constraints: high voltage requirements
            * Objective: minimize line losses, HV slacks and grid related slacks

        Must be one of [1, 2, 3, 4].
        Default: 1.
    method : str
        Optimization method to use. Must be either "soc" (Second Order Cone) or "nc"
        (Non Convex).
        If method is "soc", OPF is run in PowerModels with Gurobi solver with SOC
        relaxation of equality constraint P²+Q² = V²*I². If method is "nc", OPF is run
        with Ipopt solver as a non-convex problem due to quadratic equality constraint
        P²+Q² = V²*I².
        Default: "soc".
    warm_start : bool
        If set to True and if method is set to "soc", non-convex IPOPT OPF will be run
        additionally and will be warm started with Gurobi SOC solution. Warm-start will
        only be run if results for Gurobi's SOC relaxation is exact.
        Default: False.
    silence_moi : bool
        If set to True, MathOptInterface's optimizer attribute "MOI.Silent" is set
        to True in julia subprocess. This attribute is for silencing the output of
        an optimizer. When set to True, it requires the solver to produce no output,
        hence there will be no logging coming from julia subprocess in python
        process.
        Default: False.
    save_heat_storage : bool
        Indicates whether to save results of heat storage variables from the
        optimization to eDisGo object.
        Default: True.
    save_slack_gen : bool
        Indicates whether to save results of slack generator variables from the
        optimization to eDisGo object.
        Default: True.
    save_slacks : bool
        Indicates whether to save results of slack variables of OPF. Depending on
        chosen opf_version, different slacks are used. For more information see
        :func:`edisgo.io.powermodels_io.from_powermodels`.
        Default: True.

    """
    Topology.find_meshes(edisgo_obj)
    opf_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(opf_dir, "opf_solutions")
    pm, hv_flex_dict = edisgo_obj.to_powermodels(
        s_base=s_base,
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        flexible_storage_units=flexible_storage_units,
        opf_version=opf_version,
    )

    def _convert(o):
        """Helper function for json dump, as int64 cannot be dumped."""
        for f in [np.int8, np.int16, np.int32, np.int64]:
            if isinstance(o, f):
                return int(o)
        raise TypeError

    json_str = json.dumps(pm, default=_convert)

    # Find suitable Julia binary
    julia_binary = find_julia_binary()
    logger.info(f"starting julia process with {julia_binary}")
    
    julia_process = subprocess.Popen(
        [
            julia_binary,
            os.path.join(opf_dir, "eDisGo_OPF.jl/Main.jl"),
            pm["name"],
            solution_dir,
            method,
            str(silence_moi),
            str(warm_start),
        ],
        stdin=subprocess.PIPE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    julia_process.stdin.write(json_str)
    julia_process.stdin.close()
    while True:
        out = julia_process.stdout.readline()
        if out == "" and julia_process.poll() is not None:
            if julia_process.poll() == 0:
                logger.info("Julia process was successful.")
            else:
                # capture any remaining output and stderr for diagnostics
                try:
                    remaining_out = julia_process.stdout.read() or ""
                except Exception:
                    remaining_out = ""
                try:
                    err_out = julia_process.stderr.read() or ""
                except Exception:
                    err_out = ""

                logger.error(
                    "Julia process failed with return code %s. stdout: %s stderr: %s",
                    julia_process.returncode,
                    remaining_out,
                    err_out,
                )

                raise exceptions.InfeasibleModelError(
                    "Julia process failed! See logger for stdout/stderr details.\n"
                    f"returncode={julia_process.returncode}\n"
                    f"stdout={remaining_out}\n"
                    f"stderr={err_out}"
                )
            break
        if out.rstrip().startswith('{"name"'):
            pm_opf = json.loads(out)
            # write results to edisgo object
            from_powermodels(
                edisgo_obj,
                pm_results=pm_opf,
                hv_flex_dict=hv_flex_dict,
                s_base=s_base,
            )
        elif out.rstrip().startswith("Set parameter") or out.rstrip().startswith(
            "Academic"
        ):
            continue
        elif out != "":
            sys.stdout.write(out)
            sys.stdout.flush()
