"""
    TestUtils

Shared fixtures and helpers for the `eDisGo_OPF` test suite.

The module is deliberately dependency-free apart from the packages `eDisGo_OPF`
itself already depends on, so it can be loaded from any test file (see
`test/testsetup.jl`).

Two kinds of network fixtures are offered:

* [`pm_network`](@ref) — the *real* eDisGo test grid (143 buses, 4 time steps),
  generated from `tests/data/ding0_test_network_1` by
  `test/data/generate_fixtures.py` and checked in as JSON. Use it whenever a test
  needs data that is representative of what the Python side actually sends.
* [`small_network`](@ref) — a hand-written 4-bus network with exactly one
  component of every eDisGo flexibility type. Use it for unit tests that need to
  reason about individual components, or that have to construct degenerate cases
  (inactive components, missing fields, …).

Both return plain `Dict{String,Any}`s in single-network form; use
[`multinetwork`](@ref) and [`build_model`](@ref) to turn them into a multi-network
data dict and an instantiated (but unsolved) `AbstractBFModelEdisgo`.
"""
module TestUtils

using eDisGo_OPF
using Gurobi
using PowerModels
using InfrastructureModels
using JuMP
using JSON
import Memento

export FIXTURE_DIR,
    fixture_path,
    pm_network,
    small_network,
    add_curtailment_14a!,
    multinetwork,
    build_model,
    build_small_model,
    gurobi_available,
    num_constraints_total,
    constraint_strings,
    variable_keys,
    capture_warnings,
    silence_powermodels!

"Directory holding the checked-in JSON fixtures."
const FIXTURE_DIR = normpath(joinpath(@__DIR__, "..", "data"))

"Absolute path of the fixture file `name` inside [`FIXTURE_DIR`](@ref)."
fixture_path(name::AbstractString) = joinpath(FIXTURE_DIR, name)

# Raw fixture contents are cached per file: parsing is cheap, reading the same
# 150 kB file dozens of times is not. Callers always get an independent copy.
const _RAW_FIXTURES = Dict{String,String}()

function _raw_fixture(name::AbstractString)
    get!(_RAW_FIXTURES, name) do
        read(fixture_path(name), String)
    end
end

"""
    pm_network(; overlying_grid=false, opf_version=nothing, validate=true)

Load the eDisGo test grid in PowerModels format.

`overlying_grid` selects the fixture with overlying-grid (HV) requirements
(`pm_test_network_og.json`, `opf_version` 3/4) instead of the one without
(`pm_test_network.json`, `opf_version` 1/2). `opf_version`, when given, overwrites
the version stored in the fixture — the four OPF versions differ only in that
field, the `"curt"` entry of `"flexibilities"` and the presence of the HV
requirements, so the two fixtures cover all four. `validate=false` skips
`correct_network_data!`, which is what you want when a test needs the unvalidated
input.

Each call returns a fresh dict; mutating it does not affect other tests.
"""
function pm_network(; overlying_grid::Bool=false, opf_version::Union{Nothing,Int}=nothing,
                    validate::Bool=true)
    name = overlying_grid ? "pm_test_network_og.json" : "pm_test_network.json"
    data = eDisGo_OPF.parse_json(_raw_fixture(name); validate=validate)
    if opf_version !== nothing
        data["opf_version"] = opf_version
        # Versions 2 and 4 curtail non-dispatchable generation, 1 and 3 do not.
        flex = data["flexibilities"]
        if opf_version in (2, 4)
            "curt" in flex || pushfirst!(flex, "curt")
        else
            filter!(!isequal("curt"), flex)
        end
    end
    return data
end

"""
    small_network(; opf_version=2, num_steps=2, overlying_grid=false)

Build a minimal but complete eDisGo OPF network by hand.

Topology (a radial feeder plus the virtual branch of the battery storage)::

    bus 1 (slack) ──branch 1── bus 2 ──branch 2── bus 3 ──branch 3── bus 4 (storage bus)

Bus 3 carries one component of every flexibility type (dispatchable and
non-dispatchable generator, load, heat pump with heat storage, charging point and
DSM load), so every code path that iterates over a component type sees exactly one
element. Branch 3 is the *virtual* storage branch (`"storage" => true`), which the
model treats differently from a physical line.

Returns a single-network dict including a `"time_series"` block with `num_steps`
time steps; pass it through [`multinetwork`](@ref) to obtain the multi-network form
the OPF is built from.
"""
function small_network(; opf_version::Int=2, num_steps::Int=2,
                       overlying_grid::Bool=false)
    steps = num_steps
    rep(x) = fill(x, steps)

    flexibilities = ["cp", "hp", "dsm", "storage"]
    opf_version in (2, 4) && pushfirst!(flexibilities, "curt")

    data = Dict{String,Any}(
        "name" => "small_test_network",
        "baseMVA" => 1,
        "per_unit" => true,
        "source_version" => 2,
        "sourcetype" => "eDisGo",
        "time_elapsed" => 1,
        "opf_version" => opf_version,
        "flexibilities" => flexibilities,
        "bus" => Dict{String,Any}(
            "1" => _bus(1; bus_type=3),
            "2" => _bus(2),
            "3" => _bus(3),
            "4" => _bus(4; storage=true),
        ),
        "branch" => Dict{String,Any}(
            "1" => _branch(1, 1, 2),
            "2" => _branch(2, 2, 3),
            # Virtual branch connecting the battery storage to the grid.
            "3" => _branch(3, 3, 4; storage=true, storage_pf=0.9),
        ),
        "gen" => Dict{String,Any}(
            "1" => _gen(1, 2; pmax=0.5),
        ),
        "gen_nd" => Dict{String,Any}(
            "1" => _gen(1, 3; pmax=1.0),
        ),
        "gen_slack" => Dict{String,Any}(
            "1" => _gen(1, 1; pmax=0.0, pmin=0.0, qmin=0.0),
        ),
        "load" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "Load_1", "load_bus" => 3,
                "pd" => 0.2, "qd" => 0.0657, "pf" => 0.95, "sign" => 1,
                "status" => true,
            ),
        ),
        "storage" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "Storage_1", "storage_bus" => 4,
                "energy" => 0.0, "energy_rating" => 0.4,
                "charge_rating" => 0.4, "discharge_rating" => 0.4,
                "charge_efficiency" => 0.9, "discharge_efficiency" => 0.9,
                "pmin" => -0.4, "pmax" => 0.4,
                "qmin" => -0.19, "qmax" => 0.19,
                "ps" => 0.0, "qs" => 0.0, "pf" => 0.9, "sign" => -1,
                "p_loss" => 0, "q_loss" => 0, "r" => 0, "x" => 0,
                "soc_initial" => 0.0, "soc_end" => 0.0,
                "thermal_rating" => 1000, "status" => true,
                "virtual_branch" => "3",
            ),
        ),
        "heatpumps" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "Heat_Pump_1", "hp_bus" => 3,
                "cop" => 5.0, "pd" => 1.0, "p_min" => 0, "p_max" => 2.0,
                "q_min" => 0.0, "q_max" => 0.0, "pf" => 1.0, "sign" => 1,
            ),
        ),
        "heat_storage" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "Heat_Pump_1", "storage_bus" => 3,
                "capacity" => 4.0, "charge_efficiency" => 1.0,
                "discharge_efficiency" => 1.0, "energy" => 0, "p_loss" => 0.04,
                "ps" => 0, "soc_initial" => 0.0, "soc_end" => 0.0,
                "status" => true,
            ),
        ),
        "electromobility" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "Charging_Point_1", "cp_bus" => 3,
                "energy" => 0.5, "eta" => 0.9,
                "e_min" => 0.0, "e_max" => 1.0, "p_min" => 0, "p_max" => 1.0,
                "q_min" => 0.0, "q_max" => 0.0, "pd" => 0, "qd" => 0,
                "pf" => 1.0, "sign" => 1,
            ),
        ),
        "dsm" => Dict{String,Any}(
            "1" => Dict{String,Any}(
                "index" => 1, "name" => "DSM_Load_1", "dsm_bus" => 3,
                "charge_efficiency" => 1, "discharge_efficiency" => 1,
                "energy" => 0.0, "e_min" => -0.3, "e_max" => 0.3,
                "p_min" => -0.3, "p_max" => 0.3,
                "q_min" => -0.098, "q_max" => 0.098,
                "pd" => 0, "qd" => 0, "pf" => 0.95, "sign" => 1,
            ),
        ),
        "shunt" => Dict{String,Any}(),
        "switch" => Dict{String,Any}(),
        "dcline" => Dict{String,Any}(),
        "HV_requirements" => Dict{String,Any}(),
        "time_series" => Dict{String,Any}(
            "num_steps" => steps,
            "gen" => Dict{String,Any}("1" => Dict{String,Any}(
                "pg" => rep(0.3), "qg" => rep(-0.145))),
            "gen_nd" => Dict{String,Any}("1" => Dict{String,Any}(
                "pg" => rep(0.6), "qg" => rep(-0.29))),
            "gen_slack" => Dict{String,Any}("1" => Dict{String,Any}(
                "pg" => rep(0.0), "qg" => rep(0.0))),
            "load" => Dict{String,Any}("1" => Dict{String,Any}(
                "pd" => rep(0.2), "qd" => rep(0.0657))),
            "heatpumps" => Dict{String,Any}("1" => Dict{String,Any}(
                "cop" => rep(5.0), "pd" => rep(1.0))),
            "electromobility" => Dict{String,Any}("1" => Dict{String,Any}(
                "e_min" => rep(0.0), "e_max" => rep(1.0), "p_max" => rep(1.0))),
            "dsm" => Dict{String,Any}("1" => Dict{String,Any}(
                "e_min" => rep(-0.3), "e_max" => rep(0.3),
                "p_min" => rep(-0.3), "p_max" => rep(0.3))),
            "storage" => Dict{String,Any}(),
            "HV_requirements" => Dict{String,Any}(),
        ),
    )

    if overlying_grid
        hv = Dict{String,Any}()
        hv_ts = Dict{String,Any}()
        for (i, (name, p)) in enumerate((("curt", 0.0), ("cp", 0.4), ("hp", 1.0),
                                         ("dsm", 0.1), ("storage", -0.35)))
            hv[string(i)] = Dict{String,Any}("name" => name, "P" => p, "count" => steps)
            hv_ts[string(i)] = Dict{String,Any}("P" => rep(p))
        end
        data["HV_requirements"] = hv
        data["time_series"]["HV_requirements"] = hv_ts
    end

    return data
end

"""
    add_curtailment_14a!(data; pmax=[0.1, 0.1], max_hours_per_day=24, days=[0, 0], i=1, bus=3)

Add the §14a curtailment `i` of a heat pump or charging point at `bus` to the
single-network `data` of [`small_network`](@ref), with the upper bound `pmax` per
time step and the calendar `days` of the time steps (time series `day_14a`).
`max_hours_per_day < 24` sets a daily time budget.
"""
function add_curtailment_14a!(data::Dict{String,<:Any}; pmax::Vector=[0.1, 0.1],
                              max_hours_per_day::Real=24, days::Vector=[0, 0],
                              i::Int=1, bus::Int=3)
    get!(data, "curtailment_14a", Dict{String,Any}())[string(i)] = Dict{String,Any}(
        "index" => i, "name" => "Curtailment_14a_$i", "load_bus" => bus,
        "pmax" => pmax[1], "pf" => 0.95, "sign" => 1,
        "max_hours_per_day" => max_hours_per_day,
    )
    get!(data["time_series"], "curtailment_14a", Dict{String,Any}())[string(i)] =
        Dict{String,Any}("pmax" => pmax)
    data["time_series"]["day_14a"] = days
    return data
end

"Bus entry of [`small_network`](@ref)."
function _bus(i::Int; bus_type::Int=1, storage::Bool=false)
    return Dict{String,Any}(
        "index" => i, "bus_i" => i, "name" => "Bus_$i", "bus_type" => bus_type,
        "base_kv" => 20.0, "grid_level" => "mv", "zone" => 1,
        "vm" => 1, "va" => 0, "vmin" => 0.9, "vmax" => 1.1,
        "storage" => storage,
    )
end

"Branch entry of [`small_network`](@ref)."
function _branch(i::Int, f_bus::Int, t_bus::Int; storage::Bool=false,
                 storage_pf::Real=0, br_r::Real=0.01, br_x::Real=0.01)
    return Dict{String,Any}(
        "index" => i, "name" => "Line_$i", "f_bus" => f_bus, "t_bus" => t_bus,
        "br_r" => br_r, "br_x" => br_x, "g_fr" => 0.0, "g_to" => 0.0,
        "b_fr" => 0.0, "b_to" => 0.0, "br_status" => 1.0,
        "angmin" => -pi / 6, "angmax" => pi / 6, "shift" => 0.0, "tap" => 1.0,
        "rate_a" => 5.0, "rate_b" => 250.0, "rate_c" => 250.0,
        "transformer" => false, "length" => 1.0, "cost" => 10.0, "r" => br_r,
        "storage" => storage, "storage_pf" => storage_pf,
    )
end

"Generator entry of [`small_network`](@ref), shared by `gen`, `gen_nd` and `gen_slack`."
function _gen(i::Int, bus::Int; pmax::Real=1.0, pmin::Real=0.0,
              qmin::Real=-0.484, pf::Real=0.9)
    return Dict{String,Any}(
        "index" => i, "name" => "Generator_$i", "gen_bus" => bus,
        "gen_status" => 1, "mbase" => pmax, "vg" => 1,
        "pg" => 0.0, "qg" => -0.0, "pmin" => pmin, "pmax" => pmax,
        "qmin" => qmin, "qmax" => -0.0, "pf" => pf, "sign" => -1,
        "P" => 0, "Q" => 0,
    )
end

"""
    multinetwork(data)

Expand a single-network eDisGo data dict into the multi-network form the OPF is
built from — the same call `Main.jl` makes on the data it reads from stdin.
"""
multinetwork(data::Dict{String,<:Any}) = PowerModels.make_multinetwork(data)

"""
    build_model(model_type=SOCBFPowerModelEdisgo; build_method=eDisGo_OPF.build_mn_opf_bf_flex, kwargs...)

Instantiate (but do not solve) an OPF model on the real test grid. All keyword
arguments except `build_method` are forwarded to [`pm_network`](@ref).
"""
function build_model(model_type::Type=SOCBFPowerModelEdisgo;
                     build_method=eDisGo_OPF.build_mn_opf_bf_flex, kwargs...)
    return eDisGo_OPF.instantiate_model(multinetwork(pm_network(; kwargs...)),
                                        model_type, build_method)
end

"""
    build_small_model(model_type=SOCBFPowerModelEdisgo; build_method=eDisGo_OPF.build_mn_opf_bf_flex, kwargs...)

Instantiate (but do not solve) an OPF model on [`small_network`](@ref). All keyword
arguments except `build_method` are forwarded to `small_network`.
"""
function build_small_model(model_type::Type=SOCBFPowerModelEdisgo;
                           build_method=eDisGo_OPF.build_mn_opf_bf_flex, kwargs...)
    return eDisGo_OPF.instantiate_model(multinetwork(small_network(; kwargs...)),
                                        model_type, build_method)
end

"""
    gurobi_available()

Whether a usable Gurobi licence is present. Gurobi is a commercial solver and is
not available on every machine (nor in CI), so tests that need it must be guarded
with this predicate rather than assumed to run.
"""
function gurobi_available()
    try
        # Gurobi.jl loads without a licence; only creating an environment fails.
        Gurobi.Env()
        return true
    catch
        return false
    end
end

"Total number of constraints in `model`, across all constraint types."
function num_constraints_total(model::JuMP.Model)
    return sum(JuMP.num_constraints(model, F, S)
               for (F, S) in JuMP.list_of_constraint_types(model); init=0)
end

"""
    constraint_strings(model)

All constraints of `model` rendered as strings — handy for asserting that a
particular term or coefficient made it into the model without depending on JuMP's
internal representation.
"""
function constraint_strings(model::JuMP.Model)
    strs = String[]
    for (F, S) in JuMP.list_of_constraint_types(model)
        for c in JuMP.all_constraints(model, F, S)
            push!(strs, string(c))
        end
    end
    return strs
end

"Sorted keys of the variable dict of network `nw` of `pm`."
variable_keys(pm, nw::Int) = sort(collect(keys(PowerModels.var(pm, nw))))

"""
    silence_powermodels!()

Set the PowerModels logger to `"error"`. The OPF code logs a lot at `info`/`warn`
level; leaving that on makes the test output unreadable. Tests that assert on a
specific warning should re-enable logging locally instead.
"""
silence_powermodels!() = PowerModels.logger_config!("error")

"""
    capture_warnings(f) -> (result, warnings)

Run `f()` with the PowerModels logger — which `eDisGo_OPF` logs through — raised
to `"warn"`, and return the result of `f` together with the messages of all
warnings it logged, in order. The logger's handlers and level are restored
afterwards, so the rest of the suite stays silenced (see
[`silence_powermodels!`](@ref)). Exceptions thrown by `f` propagate unchanged.
"""
function capture_warnings(f)
    logger = Memento.getlogger(PowerModels)
    io = IOBuffer()
    handlers = copy(Memento.gethandlers(logger))
    result = nothing
    try
        logger.handlers = Dict{Any,Memento.Handler}()
        push!(logger, Memento.DefaultHandler(io, Memento.DefaultFormatter("{level}|{msg}")))
        Memento.setlevel!(logger, "warn") do
            Memento.setpropagating!(logger, false) do
                result = f()
            end
        end
    finally
        logger.handlers = handlers
    end
    lines = split(String(take!(io)), '\n'; keepempty=false)
    warnings = [String(split(l, '|'; limit=2)[2]) for l in lines if startswith(l, "warn|")]
    return result, warnings
end

end # module TestUtils
