using edisgoOPF
using JuMP
using PowerModels


function parse_file_to_network_data(filename::String)
    network_data = PowerModels.parse_file(filename);
    return network_data
end

function build_powermodels_test_case(filename::String;T=1)
    #filename = "test/data/nesta_case9_kds__rad_modified.m"
    network_data = PowerModels.parse_file(filename);
    PowerModels.print_summary(network_data)
    #T = 1
    data = PowerModels.replicate(network_data,T)
    # set up powermodel data structure
    pm = PowerModels.GenericPowerModel(data,SOCBFForm)
    return pm
end

function build_opf_simple_nep_problem(filename::String)
    pm = build_powermodels_test_case(filename)
    edisgoOPF.post_opf_bf_nep(pm)
    return pm
end

function build_opf_simple_nep_problem(pm::PowerModels.GenericPowerModel)
    edisgoOPF.post_opf_bf_nep(pm)
end

function run_opf_simple_nep_problem(filename::String)
    pm = build_powermodels_test_case(filename)
    edisgoOPF.post_opf_bf_nep(pm)
    JuMP.setsolver(pm.model,IpoptSolver())
    status = JuMP.solve(pm.model)
    return status
end

function run_opf_simple_nep_problem(pm::GenericPowerModel)
    post_opf_bf_nep(pm)
    JuMP.setsolver(pm.model,IpoptSolver())
    status = JuMP.solve(pm.model)
    return status
end