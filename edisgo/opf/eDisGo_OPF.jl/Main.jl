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
function optimize_edisgo()
  # read in data and create multinetwork
  gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => silence_moi, "FeasibilityTol"=>1e-4, "BarQCPConvTol"=>1e-4, "BarConvTol"=>1e-4, "BarHomogeneous"=>1, "Threads"=>1, "Seed"=>42)
  data_edisgo = eDisGo_OPF.parse_json(json_str)
  data_edisgo_mn = PowerModels.make_multinetwork(data_edisgo)

  if method == "soc" # Second order cone
    # Solve SOC model
    println("Starting convex SOC AC-OPF with Gurobi.")
    result_soc, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, SOCBFPowerModelEdisgo, gurobi)
    #println("Termination status: "*result_soc["termination_status"])
    if result_soc["termination_status"] != MOI.OPTIMAL
      println("ERROR: SOC optimization failed with status: $(result_soc["termination_status"])")
      try
        JuMP.compute_conflict!(pm.model)
        if MOI.get(pm.model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
          iis_model, _ = copy_conflict(pm.model)
          print(iis_model)
        end
      catch e
        println("WARNING: Could not compute conflict (IIS): $e")
      end
      exit(1)
    elseif result_soc["termination_status"] == MOI.OPTIMAL
      # Check if SOC constraint is tight
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
      if warm_start
        # SOC->NC Skalierungsfix: CP-Grenzen auf SOC-optimale Werte setzen
        # SOC hat optimale Ladeleistung je CP bestimmt (z.B. 32 kW statt 500 GW).
        # Koeffizientenrange sinkt von [7e-9, 5e+8] auf [7e-9, ~0.05] -> Ipopt konvergiert.
        println("Adjusting CP bounds for NC based on SOC solution (scaling fix)...")
        for (nw_id, network) in data_edisgo_mn["nw"]
          # 1. CP-Ladeleistung: p_max = SOC-optimaler Wert * 10 (genug Spielraum fuer NC-Physik)
          # 10x statt 1%: NC braucht Umverteilungsfreiheit zwischen CPs (sonst LOCALLY_INFEASIBLE)
          for (cp_id, cp) in get(network, "electromobility", Dict())
            if haskey(cp, "pcp") && cp["pcp"] > 0
              cp["p_max"] = min(cp["pcp"] * 10.0, cp["p_max"])
            end
          end
          # 2. Para 14a-Generatoren: pmax = SOC-optimaler Wert * 10
          for (gen_id, gen) in get(network, "gen_cp_14a", Dict())
            if haskey(gen, "p")
              gen["pmax"] = min(max(gen["p"] * 10.0, get(gen, "pmin", 0.0)), gen["pmax"])
            end
          end
        end
        println("Starting warm-start non-convex AC-OPF with IPOPT.")
        eDisGo_OPF.set_ac_bf_start_values!(data_edisgo_mn["nw"]["1"])
        result_nc_ws, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, NCBFPowerModelEdisgo, ipopt; relax_integrality=true)
        nc_status = result_nc_ws["termination_status"]
        if nc_status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
          println("NC warm-start converged: $nc_status — using Ipopt solution.")
          PowerModels.update_data!(data_edisgo_mn, result_nc_ws["solution"])
          data_edisgo_mn["solve_time"] = result_nc_ws["solve_time"]
          data_edisgo_mn["status"] = nc_status
          data_edisgo_mn["solver"] = "Ipopt"
        else
          println("WARNING: NC warm-start did not converge ($nc_status) — keeping SOC solution.")
          data_edisgo_mn["solver"] = "Gurobi (SOC, NC failed)"
        end
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
