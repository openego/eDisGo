"""
Abstract supertype for all eDisGo branch-flow OPF models. Concrete subtypes
(`BFPowerModelEdisgo`, `SOCBFPowerModelEdisgo`, `NCBFPowerModelEdisgo`) select how
the branch-flow equations are treated (base, second-order-cone relaxation, or
non-convex/exact).
"""
abstract type AbstractBFModelEdisgo <: AbstractBFQPModel end

"""
Base radial branch-flow model for the eDisGo OPF — the eDisGo extension of
PowerModels' branch-flow (`DistFlow`) formulation. Applicable to problem
formulations whose name ends in `_bf` (e.g. `build_mn_opf_bf_flex`).

The branch-flow model is the natural formulation for the tree-shaped (radial)
distribution grids eDisGo works with. For every branch `i → j` it couples the
sending-end active and reactive power `(P, Q)`, the squared branch current `ccm`
(`= I²`) and the squared bus voltages `w` (`= V²`), together with a nodal power
balance at each bus, thermal/current limits, voltage limits and the flexibility
power/energy bands. The defining coupling between power, current and voltage is
the quadratic relation `P² + Q² = V²·I²`, which is what makes an exact AC OPF
non-convex.

This type carries the shared model structure; *how* that quadratic coupling is
enforced is fixed by the concrete subtypes `SOCBFPowerModelEdisgo` (convex
relaxation) and `NCBFPowerModelEdisgo` (exact, non-convex) — one of those is what
you actually solve. All quantities are in per-unit on the common base power
`s_base`.
"""
mutable struct BFPowerModelEdisgo <: AbstractBFModelEdisgo @pm_fields end

abstract type AbstractSOCBFModelEdisgo <: AbstractBFModelEdisgo end

"""
Second-order-cone (SOC) relaxation of the radial branch-flow model. Applicable to
problem formulations whose name ends in `_bf`.

The non-convex coupling `P² + Q² = V²·I²` is relaxed to the convex inequality
`P² + Q² ≤ V²·I²` — a second-order cone (see `constraint_model_current` for
`AbstractSOCBFModelEdisgo`). This turns the OPF into a convex program, solved with
**Gurobi**: fast, reliable and globally optimal. For radial grids the relaxation is
usually *exact* — the inequality is tight at the optimum, so the solution is also
feasible for the original AC problem. `check_SOC_equality` flags any branches and
time steps where it is not tight; there, running with `warm_start=true` recovers a
feasible AC solution by polishing with the non-convex model
(`NCBFPowerModelEdisgo`). This is the model behind the default `method="soc"`.
"""
mutable struct SOCBFPowerModelEdisgo <: AbstractSOCBFModelEdisgo @pm_fields end


abstract type AbstractNCBFModelEdisgo <: AbstractBFModelEdisgo end

"""
Non-convex (NC), exact radial branch-flow model. Applicable to problem
formulations whose name ends in `_bf`.

Keeps the exact quadratic equality `P² + Q² = V²·I²` (added as a nonlinear
`@NLconstraint`; see `constraint_model_current` for `AbstractNCBFModelEdisgo`), so
no relaxation is involved. The resulting non-convex program is solved with the
**Ipopt** interior-point solver: more accurate in principle, but slower and only
guaranteed to find a *local* optimum. Selected via `method="nc"` (cold start), or
used automatically as the warm-started polishing step when `method="soc"` is run
with `warm_start=true`, starting from the exact Gurobi SOC solution.
"""
mutable struct NCBFPowerModelEdisgo <: AbstractNCBFModelEdisgo @pm_fields end
