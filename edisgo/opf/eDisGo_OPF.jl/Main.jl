cd(dirname(@__FILE__))
using Pkg
Pkg.activate("")
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
const gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => silence_moi, "Presolve" => 1, "FeasibilityTol"=>1e-4, "BarConvTol"=>1e-6, "BarQCPConvTol"=>1e-5)

function optimize_edisgo()
  # read in data and create multinetwork
  data_edisgo = parse_json(json_str)
  data_edisgo_mn = PowerModels.make_multinetwork(data_edisgo)

  if method == "soc" # Second order cone
    # Solve SOC model
    println("Starting convex SOC AC-OPF with Gurobi.")
    result_soc, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, SOCBFPowerModelEdisgo, gurobi
    # JuMP.write_to_file(pm.model, "model.mps")
    # grbtune "model.mps"
    #GRBtunemodel(unsafe_backend(pm.model))
    #GRBgetintattr(unsafe_backend(pm.model), "TuneResultCount", nresults)
    #GRBgettuneresult(unsafe_backend(pm.model), 0)

    # Find violating constraint if model is infeasible
    if result_soc["termination_status"] == MOI.INFEASIBLE
      JuMP.compute_conflict!(pm.model)

      if MOI.get(pm.model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        iis_model, _ = copy_conflict(pm.model)
        print(iis_model)
      end
    elseif result_soc["termination_status"] == MOI.OPTIMAL
      # Check if SOC constraint is tight
      soc_tight, soc_dict = eDisGo_OPF.check_SOC_equality(result_soc, data_edisgo)
      # Save SOC violations if SOC is not tight
      if !soc_tight
        open(joinpath(results_path, ding0_grid*"_"*join(data_edisgo["flexibilities"])*".json"), "w") do f
            write(f, JSON.json(soc_dict))
        end
      end
      PowerModels.update_data!(data_edisgo_mn, result_soc["solution"])
      data_edisgo_mn["solve_time"] = result_soc["solve_time"]
      if soc_tight & warm_start
        println("Starting warm-start non-convex AC-OPF with IPOPT.")
        set_ac_bf_start_values!(data_edisgo_mn["nw"]["1"])
        result_nc_ws, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, NCBFPowerModelEdisgo, ipopt)
        PowerModels.update_data!(data_edisgo_mn, result_nc_ws["solution"])
        data_edisgo_mn["solve_time"] = result_nc_ws["solve_time"]
      end
    else
      println("Termination status: "*result_soc["termination_status"])
    end
  elseif method == "nc" # Non-Convex
    # Solve NC model
    println("Starting cold-start non-convex AC-OPF with IPOPT.")
    result, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, NCBFPowerModelEdisgo, ipopt)
    PowerModels.update_data!(data_edisgo_mn, result["solution"])
    data_edisgo_mn["solve_time"] = result["solve_time"]
  end

  # Update network data with optimization results and print to stdout
  print(JSON.json(data_edisgo_mn))
end

if abspath(PROGRAM_FILE) == @__FILE__
  optimize_edisgo()
end
