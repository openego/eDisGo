import json
import logging
import os
import subprocess
import sys

from typing import Optional

import numpy as np

from edisgo.flex_opt import exceptions
from edisgo.io.powermodels_io import from_powermodels
from edisgo.network.topology import Topology

logger = logging.getLogger(__name__)


def pm_optimize(
    edisgo_obj,
    s_base: int = 1,
    flexible_cps: Optional[np.ndarray] = None,
    flexible_hps: Optional[np.ndarray] = None,
    flexible_loads: Optional[np.ndarray] = None,
    flexible_storage_units: Optional[np.ndarray] = None,
    method: str = "soc",
    warm_start: bool = False,
    silence_moi: bool = False,
    curtailment_14a_heatpumps: bool = False,
    curtailment_14a_charging_points: bool = False,
    curtailment_14a_loads: bool = False,
    minimize_losses: bool = True,
    minimize_line_loading: bool = False,
    minimize_slacks: bool = True,
    minimize_hv_slacks: bool = False,
    minimize_14a_curtailment: bool = True,
    weight_losses: Optional[float] = None,
    weight_line_loading: Optional[float] = None,
    weight_slacks: float = 0.6,
    weight_hv_slacks: Optional[float] = None,
    weight_14a: float = 0.5,
    weight_heat_storage_violation: float = 1e4,
    objective_config: Optional[dict] = None,
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
    curtailment_14a_heatpumps : bool
        If True, enables §14a EnWG curtailment for heat pumps with virtual
        generators. Heat pumps can be curtailed down to 4.2 kW with time budget
        constraints.
        Default: False.
    curtailment_14a_charging_points : bool
        If True, enables §14a EnWG curtailment for charging points with virtual
        generators. Charging points can be curtailed down to 4.2 kW with time budget
        constraints.
        Default: False.
    curtailment_14a_loads : bool
        If True, enables §14a EnWG curtailment for loads with virtual
        generators. Loads can be curtailed down to 4.2 kW with time budget
        constraints.
        Default: False.
    minimize_losses : bool
        Minimize line losses in objective function.
        Default: True.
    minimize_line_loading : bool
        Minimize maximum line loading in objective function.
        Default: False.
    minimize_slacks : bool
        Minimize grid constraint violation slacks in objective function.
        Default: True.
    minimize_hv_slacks : bool
        Minimize high voltage requirement slacks in objective function.
        Default: False.
    minimize_14a_curtailment : bool
        Minimize §14a curtailment in objective function.
        Default: True.
    weight_losses : float or None
        Weight for line losses in objective function. Auto-calculated if None.
        Default: None.
    weight_line_loading : float or None
        Weight for line loading in objective function. Auto-calculated if None.
        Default: None.
    weight_slacks : float
        Weight for slacks in objective function.
        Default: 0.6.
    weight_hv_slacks : float or None
        Weight for HV slacks in objective function. Auto-calculated if None.
        Default: None.
    weight_14a : float
        Weight for §14a curtailment in objective function. Lower values allow more
        curtailment, higher values minimize curtailment use.
        Default: 0.5.
    weight_heat_storage_violation : float
        Large penalty weight for heat storage constraint violations.
        Default: 1e4.
    objective_config : dict or None
        Optional dictionary to override all objective configuration parameters.
        If provided, individual objective parameters are ignored.
        Keys: minimize_losses, minimize_line_loading, minimize_slacks,
              minimize_hv_slacks, minimize_14a_curtailment, weight_losses,
              weight_line_loading, weight_slacks, weight_hv_slacks, weight_14a,
              weight_heat_storage_violation.
        Default: None (uses individual parameters).
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
    
    # Validate: Components cannot be both flexible AND §14a curtailable
    validation_errors = []
    
    if curtailment_14a_heatpumps and flexible_hps is not None and len(flexible_hps) > 0:
        validation_errors.append(
            "Heat pumps cannot be both flexible (flexible_hps) and §14a curtailable "
            "(curtailment_14a_heatpumps). Please choose one approach."
        )
    
    if curtailment_14a_charging_points and flexible_cps is not None and len(flexible_cps) > 0:
        validation_errors.append(
            "Charging points cannot be both flexible (flexible_cps) and §14a curtailable "
            "(curtailment_14a_charging_points). Please choose one approach."
        )
    
    if curtailment_14a_loads and flexible_loads is not None and len(flexible_loads) > 0:
        validation_errors.append(
            "Loads cannot be both flexible (flexible_loads) and §14a curtailable "
            "(curtailment_14a_loads). Please choose one approach."
        )
    
    if validation_errors:
        error_message = "\n".join([
            "❌ Configuration Error: Conflicting flexibility definitions detected!",
            "",
            "The following conflicts were found:",
            ""
        ] + ["  • " + err for err in validation_errors] + [
            "",
            "Note: §14a curtailment and flexible operation are mutually exclusive approaches.",
            "      - §14a: Simple power curtailment with minimum power constraints",
            "      - Flexible: Advanced optimization with storage, scheduling, etc.",
            "",
            "Please set either the flexible_* parameter OR the curtailment_14a_* parameter,",
            "but not both for the same component type."
        ])
        raise ValueError(error_message)
    
    # Build objective_config from individual parameters if not provided
    if objective_config is None:
        objective_config = {
            'minimize_losses': minimize_losses,
            'minimize_line_loading': minimize_line_loading,
            'minimize_slacks': minimize_slacks,
            'minimize_hv_slacks': minimize_hv_slacks,
            'minimize_14a_curtailment': minimize_14a_curtailment,
            'weight_slacks': weight_slacks,
            'weight_14a': weight_14a,
            'weight_heat_storage_violation': weight_heat_storage_violation,
        }
        # Add optional weights only if provided
        if weight_losses is not None:
            objective_config['weight_losses'] = weight_losses
        if weight_line_loading is not None:
            objective_config['weight_line_loading'] = weight_line_loading
        if weight_hv_slacks is not None:
            objective_config['weight_hv_slacks'] = weight_hv_slacks
    
    # Determine curtailment_14a flag for to_powermodels
    curtailment_14a = (
        curtailment_14a_heatpumps or 
        curtailment_14a_charging_points or 
        curtailment_14a_loads
    )
    
    pm, hv_flex_dict = edisgo_obj.to_powermodels(
        s_base=s_base,
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        flexible_storage_units=flexible_storage_units,
        curtailment_14a=curtailment_14a,
        curtailment_14a_heatpumps=curtailment_14a_heatpumps,
        curtailment_14a_charging_points=curtailment_14a_charging_points,
        curtailment_14a_loads=curtailment_14a_loads,
        objective_config=objective_config,
    )

    def _convert(o):
        """Helper function for json dump, as int64 cannot be dumped."""
        for f in [np.int8, np.int16, np.int32, np.int64]:
            if isinstance(o, f):
                return int(o)
        raise TypeError

    json_str = json.dumps(pm, default=_convert)

    logger.info("starting julia process")
    julia_process = subprocess.Popen(
        [
            "julia",
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
    )
    julia_process.stdin.write(json_str)
    julia_process.stdin.close()
    while True:
        out = julia_process.stdout.readline()
        if out == "" and julia_process.poll() is not None:
            if julia_process.poll() == 0:
                logger.info("Julia process was successful.")
            else:
                raise exceptions.InfeasibleModelError("Julia process failed!")
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
