"""
    eDisGo_OPF

Julia implementation of eDisGo's multi-period optimal power flow (OPF) for
distribution grids. Extends
[PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl) with a radial
branch-flow formulation and eDisGo's flexibilities: battery storage,
electromobility (charging points), heat pumps with thermal storage, demand-side
management, generation curtailment, and overlying-grid (HV) requirements.

The package is normally driven from Python via `EDisGo.pm_optimize`, which
serialises the grid and flexibility data to JSON, launches Julia on `Main.jl`, and
reads the optimised operation schedules back.

# Problem formulations
- `BFPowerModelEdisgo` — base radial branch-flow model.
- `SOCBFPowerModelEdisgo` — second-order-cone relaxation (convex; solved with Gurobi).
- `NCBFPowerModelEdisgo` — non-convex, exact model (solved with Ipopt).

# Main entry points
- `build_mn_opf_bf_flex` / `solve_mn_opf_bf_flex` — assemble and solve the
  multi-network OPF.
- `parse_json` / `correct_network_data!` — read and validate the input data.

The problem is configured by `opf_version` (1–4, choosing the objective and whether
grid restrictions / overlying-grid requirements are enforced) and by `method`
(`"soc"` or `"nc"`).
"""
module eDisGo_OPF

using PowerModels
using InfrastructureModels
using Memento
using JuMP
using Ipopt
using JSON
using Compat
using Gurobi

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
