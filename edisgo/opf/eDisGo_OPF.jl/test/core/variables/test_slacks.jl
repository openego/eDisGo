# Tests for the slack variables of `src/core/variables.jl` — grid restrictions,
# heat pump / heat storage, slack generator and overlying-grid requirements, from
# `variable_slack_grid_restrictions` to the end of the file.
#
# TODO: not written yet. See test/FUNKTIONSABDECKUNG.md for the functions still
# missing tests and test/TESTKONVENTIONEN.md for how to write them.

include(joinpath(@__DIR__, "..", "..", "testsetup.jl"))
