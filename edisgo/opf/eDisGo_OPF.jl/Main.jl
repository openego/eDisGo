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
    using Printf
catch e
    Pkg.instantiate()
    using eDisGo_OPF
    using PowerModels
    using Ipopt
    using JuMP
    using JSON
    using Gurobi
    using Printf
end



PowerModels.logger_config!("debug")
json_str = readline(stdin)
ding0_grid = ARGS[1]
results_path = ARGS[2]
method = ARGS[3]
silence_moi = ARGS[4].=="True"
warm_start = ARGS[5].=="True"


"""
    print_14a_debug_summary(data_mn; threshold_mw=0.001)

Print a per-timestep §14a activation summary to stderr after solving.

Output goes to stderr so it never mixes with the JSON result on stdout.
Each line shows HP and CP §14a activation in kW plus the approximate maximum
active-power line loading at that timestep.  Only timesteps where the total
§14a activation exceeds `threshold_mw` are printed.

Activation values are in per-unit (relative to s_base).  Multiply by the
transformer s_nom to convert to MW.
"""
function print_14a_debug_summary(data_mn; threshold_mw=0.001)
    println(stderr, "\n===== §14a Debug Summary (Julia) =====")

    nw_keys = sort(collect(keys(data_mn["nw"])), lt=(a, b) -> parse(Int, a) < parse(Int, b))
    n_active = 0

    for t in nw_keys
        nw_t = data_mn["nw"][t]

        # --- §14a HP activation ---
        hp14a_pu = 0.0
        if haskey(nw_t, "gen_hp_14a")
            for (_, gen) in nw_t["gen_hp_14a"]
                p_val = get(gen, "p", 0.0)
                if p_val > 0
                    hp14a_pu += p_val
                end
            end
        end

        # --- §14a CP activation ---
        cp14a_pu = 0.0
        if haskey(nw_t, "gen_cp_14a")
            for (_, gen) in nw_t["gen_cp_14a"]
                p_val = get(gen, "p", 0.0)
                if p_val > 0
                    cp14a_pu += p_val
                end
            end
        end

        total_pu = hp14a_pu + cp14a_pu
        if total_pu < threshold_mw
            continue
        end

        n_active += 1

        # --- Approximate max line loading: |pf| / rate_a ---
        max_loading = 0.0
        max_branch_id = ""
        if haskey(nw_t, "branch")
            for (b_id, branch_res) in nw_t["branch"]
                is_storage = get(get(nw_t["branch"], b_id, Dict()), "storage", 0) == 1
                if is_storage
                    continue
                end
                rate_a = get(branch_res, "rate_a", 0.0)
                pf_val = abs(get(branch_res, "pf", 0.0))
                if rate_a > 1e-9
                    loading = pf_val / rate_a
                    if loading > max_loading
                        max_loading = loading
                        max_branch_id = b_id
                    end
                end
            end
        end

        @printf(stderr,
            "  t=%4s | HP_14a=%7.2f kVA | CP_14a=%7.2f kVA | Total=%7.2f kVA | max_line=%.1f%% (%s)\n",
            t,
            hp14a_pu * 1000,
            cp14a_pu * 1000,
            total_pu * 1000,
            max_loading * 100,
            max_branch_id,
        )
    end

    println(stderr, "  §14a active timesteps (>$(threshold_mw*1000) kVA): $n_active / $(length(nw_keys))")
    println(stderr, "  Note: values are in per-unit × 1000 (≈ kVA at s_base=1 MVA).")
    println(stderr, "        Multiply by transformer s_nom [MVA] to get absolute MW.")
    println(stderr, "===== End §14a Debug Summary =====\n")
end


# Set solver attributes
const ipopt = optimizer_with_attributes(Ipopt.Optimizer, MOI.Silent() => silence_moi, "sb" => "yes", "tol"=>1e-6)
function optimize_edisgo()
  # read in data and create multinetwork
  gurobi = optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => silence_moi, "FeasibilityTol"=>1e-4, "BarQCPConvTol"=>1e-4, "BarConvTol"=>1e-4, "BarHomogeneous"=>1, "MIPGap"=>0.01, "Threads"=>4)
  data_edisgo = eDisGo_OPF.parse_json(json_str)
  data_edisgo_mn = PowerModels.make_multinetwork(data_edisgo)

  if method == "soc" # Second order cone
    # Solve SOC model
    println("Starting convex SOC AC-OPF with Gurobi.")
    result_soc, pm = eDisGo_OPF.solve_mn_opf_bf_flex(data_edisgo_mn, SOCBFPowerModelEdisgo, gurobi)
    #println("Termination status: "*result_soc["termination_status"])
    if result_soc["termination_status"] != MOI.OPTIMAL
      # if result_soc["termination_status"] == MOI.SUBOPTIMAL_TERMINATION
      #   PowerModels.update_data!(data_edisgo_mn, result_soc["solution"])
      # else
      JuMP.compute_conflict!(pm.model)
      if MOI.get(pm.model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        iis_model, _ = copy_conflict(pm.model)
        print(iis_model)
      end
      #end
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
      print_14a_debug_summary(data_edisgo_mn)
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
    print_14a_debug_summary(data_edisgo_mn)
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
