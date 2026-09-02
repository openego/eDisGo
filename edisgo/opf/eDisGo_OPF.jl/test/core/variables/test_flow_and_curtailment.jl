# Tests for the branch-flow, bus-voltage, line-loading and curtailment variables
# of `src/core/variables.jl` — the first block of the file, from
# `variable_branch_power_radial` through `variable_dsm_storage_energy`.
#
# TODO: not written yet. See test/FUNKTIONSABDECKUNG.md for the functions still
# missing tests and test/TESTKONVENTIONEN.md for how to write them.

include(joinpath(@__DIR__, "..", "..", "testsetup.jl"))
