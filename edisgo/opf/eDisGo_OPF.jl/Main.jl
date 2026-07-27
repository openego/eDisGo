# Main.jl — command-line entry point for the eDisGo OPF, launched as a Julia
# subprocess by the Python side (`EDisGo.pm_optimize` / `edisgo/opf/powermodels_opf.py`).
#
# It activates the package environment, reads the serialised grid + flexibility data
# as a single JSON line from stdin, and takes five positional arguments:
#   ARGS[1] ding0_grid    grid id / name (used for the SOC-violation file name)
#   ARGS[2] results_path  directory for SOC-violation output
#   ARGS[3] method        "soc" (convex, Gurobi) or "nc" (non-convex, Ipopt)
#   ARGS[4] silence_moi   "True"/"False" — silence the solver
#   ARGS[5] warm_start    "True"/"False" — polish a tight SOC solution with Ipopt
# The optimised network (operation schedules of all flexibilities, solve status and
# time) is printed back to stdout as JSON.
cd(dirname(@__FILE__))
using Pkg
Pkg.activate("")
Pkg.instantiate()
try
    using eDisGo_OPF
    using PowerModels
    using Ipopt
    using JuMP
    using JSON
    using Gurobi
catch e
    Pkg.instantiate()
    using eDisGo_OPF
    using PowerModels
    using Ipopt
    using JuMP
    using JSON
    using Gurobi
end



PowerModels.logger_config!("debug")
json_str = readline(stdin)
ding0_grid = ARGS[1]
results_path = ARGS[2]
method = ARGS[3]
silence_moi = ARGS[4].=="True"
warm_start = ARGS[5].=="True"

# Set solver attributes
const ipopt = optimizer_with_attributes(Ipopt.Optimizer, MOI.Silent() => silence_moi, "sb" => "yes", "tol"=>1e-6)

"""
Run the eDisGo OPF for the JSON network read from stdin and print the optimised
network back to stdout as JSON.

Builds a multi-network from the input and solves it according to the `method`
argument: `"soc"` solves the convex second-order-cone model with Gurobi (and, if
`warm_start` is set and the SOC solution is tight, polishes it with a non-convex
Ipopt solve warm-started from it), while `"nc"` solves the non-convex model directly
with Ipopt. Solver infeasibilities are reported via an IIS conflict, and non-tight
SOC solutions are written to `results_path`.
"""
function optimize_edisgo()
  # read in data and create multinetwork
  gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => silence_moi, "FeasibilityTol"=>1e-4, "BarQCPConvTol"=>1e-4, "BarConvTol"=>1e-4, "BarHomogeneous"=>1)
  data_edisgo = eDisGo_OPF.parse_json(json_str)
  data_edisgo_mn = PowerModels.make_multinetwork(data_edisgo)

  if method == "soc" # Second order cone
    # Solve SOC model
    println("Starting convex SOC AC-OPF with Gurobi.")
    result_soc, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, SOCBFPowerModelEdisgo, gurobi)
    #println("Termination status: "*result_soc["termination_status"])
    # A feasible solution exists if the solver proved optimality OR reports a
    # feasible primal point (e.g. SUBOPTIMAL / ALMOST_OPTIMAL under the barrier
    # tolerances set above). Only when there is genuinely no primal solution do
    # we diagnose the infeasibility via an IIS conflict — calling
    # compute_conflict! on a feasible model raises Gurobi error 10015.
    # A usable primal solution exists if the solver proved optimality, or it
    # returns a feasible OR nearly-feasible primal point. The latter covers
    # suboptimal / almost-optimal / numerically loose terminations (under the
    # barrier tolerances set above) that still carry a valid dispatch — these
    # are used just like an optimal solution. Only a genuine no-primal-point
    # case is diagnosed via an IIS conflict.
    term_status = result_soc["termination_status"]
    primal_status = MOI.get(pm.model, MOI.PrimalStatus())
    has_solution = term_status == MOI.OPTIMAL ||
                   primal_status == MOI.FEASIBLE_POINT ||
                   primal_status == MOI.NEARLY_FEASIBLE_POINT
    if !has_solution
      # No primal point detected — diagnose the infeasibility via an IIS
      # conflict. Some numerically tricky but actually feasible models are
      # misreported here, and compute_conflict! then raises Gurobi error 10015
      # ("Cannot compute IIS on a feasible model"). In that case the model does
      # have a solution after all, so recover and use it instead of failing.
      try
        JuMP.compute_conflict!(pm.model)
        if MOI.get(pm.model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
          iis_model, _ = copy_conflict(pm.model)
          print(iis_model)
        end
      catch e
        println("compute_conflict! failed (model is feasible, status "*
                string(term_status)*"/"*string(primal_status)*
                "); using the solution: ", e)
        has_solution = true
      end
    end
    if has_solution
      if term_status != MOI.OPTIMAL
        println("SOC model terminated feasible but not optimal ("*
                string(term_status)*"); using the solution.")
      end
      # Check if the SOC constraint is tight on the solution that is actually
      # used. This is a pure arithmetic check on the primal point, so it applies
      # to suboptimal-but-feasible solutions as well — a loose relaxation means
      # the dispatch is not AC-exact and downstream reinforcement may be off.
      soc_tight, soc_dict = eDisGo_OPF.check_SOC_equality(result_soc, data_edisgo)
      # Save SOC violations if SOC is not tight
      if !soc_tight
        open(joinpath(results_path, ding0_grid*"_"*join(data_edisgo["flexibilities"])*".json"), "w") do f
            write(f, JSON.json(soc_dict))
        end
        println("SOC solution is not tight!")
      end
      PowerModels.update_data!(data_edisgo_mn, result_soc["solution"])
      data_edisgo_mn["solve_time"] = result_soc["solve_time"]
      data_edisgo_mn["status"] = result_soc["termination_status"]
      data_edisgo_mn["solver"] = "Gurobi"
      if soc_tight & warm_start
        println("Starting warm-start non-convex AC-OPF with IPOPT.")
        set_ac_bf_start_values!(data_edisgo_mn["nw"]["1"])
        result_nc_ws, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, NCBFPowerModelEdisgo, ipopt)
        PowerModels.update_data!(data_edisgo_mn, result_nc_ws["solution"])
        data_edisgo_mn["solve_time"] = result_nc_ws["solve_time"]
        data_edisgo_mn["status"] = result_nc_ws["termination_status"]
        data_edisgo_mn["solver"] = "Ipopt"
      end
    end
  elseif method == "nc" # Non-Convex
    # Solve NC model
    println("Starting cold-start non-convex AC-OPF with IPOPT.")
    result, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, NCBFPowerModelEdisgo, ipopt)
    PowerModels.update_data!(data_edisgo_mn, result["solution"])
    data_edisgo_mn["solve_time"] = result["solve_time"]
    data_edisgo_mn["status"] = result["termination_status"]
    data_edisgo_mn["solver"] = "Ipopt"
  end

  # Update network data with optimization results and print to stdout
  print(JSON.json(data_edisgo_mn))
end

if abspath(PROGRAM_FILE) == @__FILE__
  optimize_edisgo()
end
