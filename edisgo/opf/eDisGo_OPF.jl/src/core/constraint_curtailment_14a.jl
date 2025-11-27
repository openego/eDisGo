# §14a EnWG Curtailment Constraints
# Diese Constraints implementieren §14a EnWG Abregelung als Alternative zu Netzausbau
# §14a erlaubt Netzbetreibern die Leistung auf MINIMAL 4.2 kW zu begrenzen

"""
Add §14a EnWG curtailment constraints for charging points and heat pumps.

§14a EnWG allows network operators to curtail devices to a MINIMUM of 4.2 kW.
This is implemented as a HARD constraint - devices cannot go below 4.2 kW!

Correct interpretation:
- Devices normally run at their requested power (can be > 4.2 kW)
- §14a allows curtailment DOWN TO a minimum of 4.2 kW
- Below 4.2 kW is NOT allowed (hard constraint, no slack!)

Formulation:
  pcp >= curtailment_limit (HARD constraint)
  pcp_14a_curt >= p_max - pcp (tracking variable for cost calculation)

Where:
- Device power is bounded: [4.2 kW, p_max]
- If grid issues require < 4.2 kW: other constraints will be violated instead
- This ensures §14a minimum is ALWAYS respected

Cost structure (hierarchy from cheap to expensive):
1. Normal flexibility (storage, redispatch): Factor 0.6
2. §14a curtailment: Factor 100 (expensive! last resort before grid violations)
3. Grid violations (voltage/current): Factor 10,000

Examples:
- No grid issues: Device runs at p_max (e.g., 11 kW), pcp_14a_curt = 0, no §14a costs
- Grid congestion: Device curtailed to 6 kW, pcp_14a_curt = 5 kW, costs = 100 × 5 = 500
- Critical: Device at 4.2 kW minimum, pcp_14a_curt = 6.8 kW, costs = 100 × 6.8 = 680
- Impossible: If even 4.2 kW too much → grid violations occur (cost: 10,000)

This ensures §14a is used only when normal flexibility is insufficient,
but prevents grid violations when possible.
"""
function constraint_curtailment_14a!(pm::AbstractPowerModel, nw::Int)
    # Check if curtailment_14a is defined in the data
    curtailment_data = get(PowerModels.ref(pm, nw), :curtailment_14a, nothing)

    if curtailment_data === nothing || curtailment_data == "nothing"
        # No curtailment constraints
        return
    end

    curtailment_limit_mw = get(curtailment_data, "max_power_mw", 0.0042)  # 4.2 kW MINIMUM
    components = get(curtailment_data, "components", [])

    # Apply HARD minimum power constraint to charging points
    for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
        if isempty(components) || get(cp, "name", "") in components
            pcp = PowerModels.var(pm, nw, :pcp, i)
            pcp_14a_curt = PowerModels.var(pm, nw, :pcp_14a_curt, i)

            # Only apply constraint if p_max > curtailment_limit
            # (doesn't make sense to enforce 4.2 kW minimum on a 3 kW device)
            if cp["p_max"] >= curtailment_limit_mw
                # HARD CONSTRAINT: pcp >= 4.2 kW
                # No slack variable - this is absolute!
                #
                # Examples for 11 kW device:
                #   pcp can be: [4.2, 11.0] kW
                #   pcp cannot be: < 4.2 kW (infeasible!)
                #
                # If grid cannot handle 4.2 kW: grid violations will occur
                JuMP.@constraint(pm.model, pcp >= curtailment_limit_mw)

                # Track curtailment amount for cost calculation
                # pcp_14a_curt measures how much power is curtailed due to §14a
                # Example: p_max=11 kW, pcp=6 kW → pcp_14a_curt=5 kW (curtailment amount)
                # MUST be equality (==) so negative costs work correctly!
                JuMP.@constraint(pm.model, pcp_14a_curt == cp["p_max"] - pcp)
            end
        end
    end

    # Apply HARD minimum power constraint to heat pumps
    for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
        if isempty(components) || get(hp, "name", "") in components
            php = PowerModels.var(pm, nw, :php, i)
            php_14a_curt = PowerModels.var(pm, nw, :php_14a_curt, i)

            # Only apply constraint if p_max > curtailment_limit
            if hp["p_max"] >= curtailment_limit_mw
                # HARD CONSTRAINT: php >= 4.2 kW (no slack!)
                JuMP.@constraint(pm.model, php >= curtailment_limit_mw)

                # Track curtailment amount for cost calculation
                # php_14a_curt measures how much power is curtailed due to §14a
                # MUST be equality (==) so negative costs work correctly!
                JuMP.@constraint(pm.model, php_14a_curt == hp["p_max"] - php)
            end
        end
    end
end
