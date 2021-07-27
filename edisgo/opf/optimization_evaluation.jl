#=
optimization_evaluation:
- Julia version: 
- Author: RL-INSTITUT\jaap.pedersen
- Date: 2019-12-19
=#
using edisgoOPF
using PowerModels
PowerModels.silence()
path = ARGS[1]
network_name = ARGS[2]
results_path = ARGS[3]
scenario_folder = "$(path)/edisgo_scenario_data"

network_files = "$(scenario_folder)/$(network_name)"

solution_files = "$(results_path)/$(network_name)"

edisgoOPF.greet()
println(network_files)
println(network_name)

edisgoOPF.run_edisgo_opf_problem(network_files,solution_files)


# data = edisgoOPF.read_edisgo_problem(network_name)
#
# pm = PowerModels.GenericPowerModel(data,SOCBFForm);
# edisgoOPF.post_method_edisgo(pm)


# println("helllo python")
