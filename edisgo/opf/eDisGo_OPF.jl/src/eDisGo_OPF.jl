module eDisGo_OPF

using PowerModels
using InfrastructureModels
using Memento
using JuMP
using Ipopt
using JSON
using Compat
using Gurobi

function __init__()
	# Print runtime versions when the module is initialized (runs on `using eDisGo_OPF`).
	try
		# Julia version
		println("[eDisGo_OPF] Julia ", VERSION)

		# Try to locate PowerModels on LOAD_PATH and read its Project.toml for version
		pm_path = Base.find_package("PowerModels")
		if pm_path !== nothing
			pkgdir = normpath(joinpath(pm_path, "..", ".."))
			proj = joinpath(pkgdir, "Project.toml")
			if isfile(proj)
				# simple parse: look for a line starting with `version =`
				for line in eachline(proj)
					s = strip(line)
					if startswith(s, "version") && occursin("=", s)
						ver = strip(split(s, "=")[2])
						println("[eDisGo_OPF] PowerModels " * ver)
						return
					end
				end
				println("[eDisGo_OPF] PowerModels found at " * pkgdir * " (version unknown)")
			else
				println("[eDisGo_OPF] PowerModels found at " * pkgdir)
			end
		else
			println("[eDisGo_OPF] PowerModels not listed in project dependencies")
		end
	catch e
		@warn("eDisGo_OPF: could not query package versions: $e")
	end
end

const _pm_global_keys = Set(["time_series", "per_unit"])
const pm_it_name = "pm"
const pm_it_sym = Symbol(pm_it_name)

# include functions extending PowerModels functions
include("core/types.jl")
include("core/base.jl")
include("core/constraint.jl")
include("core/constraint_template.jl")
include("core/data.jl")
include("core/objective.jl")
include("core/solution.jl")
include("core/variables.jl")
include("form/bf.jl")
include("prob/opf_bf.jl")
include("io/common.jl")
include("io/json.jl")
#include("../test/opf_test_case.jl")

# export new types of PowerModels
export BFPowerModelEdisgo, SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo

end
