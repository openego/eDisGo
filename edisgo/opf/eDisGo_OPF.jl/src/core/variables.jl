# =============================================================================
# VARIABLES.JL - VARIABLE DEFINITIONS FOR EDISGO OPTIMAL POWER FLOW
# =============================================================================
# Diese Datei definiert alle Optimierungsvariablen für das eDisGo OPF-Problem.
# Das System modelliert ein Verteilnetz mit verschiedenen Flexibilitätsoptionen:
# - Batteriespeicher mit optimierbarer Kapazität
# - Demand-Side Management (DSM)
# - Wärmepumpen und Wärmespeicher
# - Elektromobilität (Ladepunkte)
# - Verschiedene Slack-Variablen für Betriebsrestriktionen
# =============================================================================

# =============================================================================
# 1. NETZWERK-GRUNDVARIABLEN
# =============================================================================
# Diese Sektion definiert die grundlegenden Variablen für die Modellierung
# der elektrischen Netzwerkphysik: Leistungsflüsse, Spannungen und Ströme.

"""
Definiert sowohl aktive als auch reaktive Leistungsfluss-Variablen für radiale Netzwerke.
Ruft die spezifischen Funktionen für P und Q auf.
"""
function variable_branch_power_radial(pm::AbstractPowerModel; kwargs...)
    variable_branch_power_real_radial(pm; kwargs...)      # Wirkleistungsflüsse
    variable_branch_power_imaginary_radial(pm; kwargs...) # Blindleistungsflüsse
end

"""
Variable: `p[l,i,j]` für `(l,i,j)` in `arcs_from`

Definiert die Wirkleistungsfluss-Variablen für alle gerichteten Leitungsbögen.
Diese Variablen repräsentieren die Wirkleistung [kW], die von Bus i zu Bus j
über Leitung l fließt.

Parameter:
- pm: PowerModel Objekt
- nw: Netzwerk-ID (für Multiperioden-Optimierung)
- bounded: Ob Leistungsgrenzen gesetzt werden sollen
- report: Ob Ergebnisse in die Lösung geschrieben werden sollen
"""
function variable_branch_power_real_radial(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    # Erstelle Wirkleistungsfluss-Variable für alle Leitungsbögen (von-nach)
    p = PowerModels.var(pm, nw)[:p] = JuMP.@variable(pm.model,
        [(l,i,j) in PowerModels.ref(pm, nw, :arcs_from)],
        base_name="$(nw)_p",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "p_start")
    )

    # Setze Leistungsgrenzen basierend auf thermischen Leitungskapazitäten
    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(PowerModels.ref(pm, nw, :branch), PowerModels.ref(pm, nw, :bus))

        for arc in PowerModels.ref(pm, nw, :arcs_from)
            l,i,j = arc
            # Untere Grenze (kann negativ sein für bidirektionale Flüsse)
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(p[arc], flow_lb[l])
            end
            # Obere Grenze (thermische Kapazität der Leitung)
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(p[arc], flow_ub[l])
            end
        end
    end

    # Setze Startwerte für bessere Solver-Konvergenz
    for (l,branch) in PowerModels.ref(pm, nw, :branch)
        # Fluss von Bus f_bus zu Bus t_bus
        if haskey(branch, "pf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(p[f_idx], branch["pf_start"])
        end
        # Rückfluss von Bus t_bus zu Bus f_bus
        if haskey(branch, "pt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(p[t_idx], branch["pt_start"])
        end
    end

    # Registriere Variable für Ergebnis-Reporting
    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :pf, PowerModels.ref(pm, nw, :arcs_from), p)
end

"""
Variable: `q[l,i,j]` für `(l,i,j)` in `arcs_from`

Definiert die Blindleistungsfluss-Variablen für alle gerichteten Leitungsbögen.
Diese Variablen repräsentieren die Blindleistung [kvar], die von Bus i zu Bus j
über Leitung l fließt. Blindleistung ist wichtig für Spannungsregelung.
"""
function variable_branch_power_imaginary_radial(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    # Erstelle Blindleistungsfluss-Variable für alle Leitungsbögen
    q = PowerModels.var(pm, nw)[:q] = JuMP.@variable(pm.model,
        [(l,i,j) in PowerModels.ref(pm, nw, :arcs_from)],
        base_name="$(nw)_q",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "q_start")
    )

    # Setze Blindleistungsgrenzen (typischerweise gleich den Wirkleistungsgrenzen)
    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(PowerModels.ref(pm, nw, :branch), PowerModels.ref(pm, nw, :bus))

        for arc in PowerModels.ref(pm, nw, :arcs_from)
            l,i,j = arc
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(q[arc], flow_lb[l])
            end
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(q[arc], flow_ub[l])
            end
        end
    end

    # Setze Startwerte für Blindleistungsflüsse
    for (l,branch) in PowerModels.ref(pm, nw, :branch)
        if haskey(branch, "qf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(q[f_idx], branch["qf_start"])
        end
        if haskey(branch, "qt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(q[t_idx], branch["qt_start"])
        end
    end

    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :qf, PowerModels.ref(pm, nw, :arcs_from), q)
end

"""
Variable: `w[i]` für alle Busse i (Spannungsmagnituden-Quadrat)

Definiert die Spannungsmagnituden-Quadrat-Variablen w = |V|² für alle Busse.
Diese Formulierung wird im Branch Flow Model verwendet und führt zu konvexen Constraints.
Ausgeschlossen sind Busse mit Speicher-Flag (virtuelle Busse).

Typische Werte: w = 1.0 entspricht Nennspannung (z.B. 400V → w = 0.16 kV²)
"""
function variable_bus_voltage_magnitude_sqr(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    # Filtere Busse: Keine virtuellen Speicher-Busse
    busses = [i for i in PowerModels.ids(pm, nw, :bus) if !(PowerModels.ref(pm, nw, :bus)[i]["storage"])]

    # Erstelle Spannungsmagnituden-Quadrat-Variable (immer ≥ 0)
    w = PowerModels.var(pm, nw)[:w] = JuMP.@variable(pm.model,
        [i in busses],
        base_name="$(nw)_w",
        lower_bound = 0,                    # Physikalisch: |V|² ≥ 0
        start = comp_start_value(PowerModels.ref(pm, nw, :bus, i), "w_start", 1.001)
    )

    # Setze Spannungsgrenzen basierend auf Betriebsgrenzen (z.B. ±10% Nennspannung)
    if bounded
        for (i, bus) in PowerModels.ref(pm, nw, :bus)
            if i in busses
                # Untere Grenze: (V_min)² (z.B. 0.9² = 0.81 p.u.)
                JuMP.set_lower_bound(w[i], bus["vmin"]^2)
                # Obere Grenze: (V_max)² (z.B. 1.1² = 1.21 p.u.)
                JuMP.set_upper_bound(w[i], bus["vmax"]^2)
            end
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :bus, :w, busses, w)
end

# =============================================================================
# 2. NETZWERK-BETRIEBSVARIABLEN
# =============================================================================
# Diese Sektion definiert Variablen für Netzwerkbetrieb und -überwachung.

"""
Wrapper-Funktion für Line Loading Variablen.
Ruft die spezifische Implementierung auf.
"""
function variable_max_line_loading(pm::AbstractPowerModel; kwargs...)
    variable_line_loading_max(pm; kwargs...)
end

"""
Variable: `ll[l,i,j]` für `(l,i,j)` in `arcs_from` (Line Loading)

Definiert die Leitungsbelastungs-Variablen, die das Verhältnis zwischen
aktuellem Stromfluss und maximaler Leitungskapazität repräsentieren.
ll = 1.0 bedeutet 100% Auslastung, ll > 1.0 bedeutet Überlastung.

Diese Variablen werden in Zielfunktionen verwendet, um Netzausbau zu vermeiden.
Nur für normale Leitungen (keine Speicher-Leitungen).
"""
function variable_line_loading_max(pm::AbstractPowerModel; nw::Int=nw_id_default, report::Bool=true)
    # Filtere Leitungen: Keine virtuellen Speicher-Leitungen
    branches = [(l, i, j) for (l, i, j) in PowerModels.ref(pm, nw, :arcs_from)
                if !PowerModels.ref(pm, 1, :branch)[l]["storage"]]

    # Erstelle Line Loading Variable (≥ 1, da 100% = Vollauslastung)
    ll = PowerModels.var(pm, nw)[:ll] = JuMP.@variable(pm.model,
        [(l,i,j) in branches],
        base_name="$(nw)_ll",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "ll_start"),
        lower_bound = 1         # Minimum 100% Auslastung
    )

    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :ll, branches, ll)
end

# =============================================================================
# 3. ERZEUGUNGS-VARIABLEN
# =============================================================================
# Diese Sektion definiert Variablen für Stromerzeugung und deren Kürzung.

"""
Wrapper-Funktion für Generation Curtailment Variablen.
Definiert sowohl aktive als auch reaktive Leistungskürzungen.
"""
function variable_gen_power_curt(pm::AbstractPowerModel; kwargs...)
    variable_gen_power_curt_real(pm; kwargs...)
    # variable_gen_power_curt_imaginary(pm; kwargs...)  # Oft nicht benötigt
end

"""
Variable: `pgc[j]` für `j` in `gen_nd` (Generation Curtailment)

Definiert die Wirkleistungskürzungs-Variablen für nicht-steuerbare Generatoren
(z.B. PV-Anlagen, Windkraftanlagen). Diese Variablen repräsentieren die Menge
an erneuerbarer Energie [kW], die abgeregelt (nicht eingespeist) wird.

Anwendung: Vermeidung von Netzüberlastungen durch Reduzierung der Einspeisung.
"""
function variable_gen_power_curt_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    # Erstelle Kürzungsvariable für alle nicht-steuerbaren Generatoren
    pgc = PowerModels.var(pm, nw)[:pgc] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_nd)],
        base_name="$(nw)_pgc",
        start = comp_start_value(PowerModels.ref(pm, nw, :gen_nd, i), "pgc_start")
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen_nd)
            # Keine negative Kürzung möglich
            JuMP.set_lower_bound(pgc[i], 0)
            # Maximale Kürzung = gesamte verfügbare Erzeugung
            JuMP.set_upper_bound(pgc[i], gen["pg"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen_nd, :pgc, PowerModels.ids(pm, nw, :gen_nd), pgc)
end

"""
Variable: `qgc[j]` für `j` in `gen_nd` (Reactive Generation Curtailment)

Definiert die Blindleistungskürzungs-Variablen für nicht-steuerbare Generatoren.
Wird seltener verwendet, kann aber für erweiterte Spannungsregelung relevant sein.
"""
function variable_gen_power_curt_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qgc = PowerModels.var(pm, nw)[:qgc] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_nd)],
        base_name="$(nw)_qgc"
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen_nd)
            # Blindleistungskürzung: meist zwischen ursprünglicher Erzeugung und 0
            JuMP.set_lower_bound(qgc[i], gen["qg"])
            JuMP.set_upper_bound(qgc[i], 0)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen_nd, :qgc, PowerModels.ids(pm, nw, :gen_nd), qgc)
end

# =============================================================================
# 4. BATTERIESPEICHER-VARIABLEN
# =============================================================================
# Diese Sektion definiert alle Variablen für Batteriespeicher-Systeme,
# einschließlich der neuen Funktionalität zur Speichergrößenoptimierung.

"""
Hauptfunktion für Batteriespeicher-Variablen.

Definiert alle relevanten Variablen für Batteriespeicher:
1. Wirkleistung (Laden/Entladen)
2. Energieinhalt (State of Charge)
3. Speicherkapazität (optimierbar) - NEUE FUNKTIONALITÄT

Diese umfassende Modellierung ermöglicht sowohl operationelle Optimierung
als auch Investitionsplanung für Speichersysteme.
"""
function variable_battery_storage(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_battery_storage_power_real(pm; kwargs...)  # P-Variable (kW)
    PowerModels.variable_storage_energy(pm; kwargs...)            # SOC-Variable (kWh)
    eDisGo_OPF.variable_storage_capacity(pm; kwargs...)           # Kapazität (kWh) - NEU
end

"""
Variable: `ps[i]` für alle Speicher i (Storage Power)

Definiert die Wirkleistungs-Variablen für Batteriespeicher [kW].
Positive Werte = Entladung (Einspeisung ins Netz)
Negative Werte = Ladung (Bezug aus dem Netz)

Diese Variable koppelt den Speicher mit dem Netz und ermöglicht
bidirektionale Energieflüsse je nach Netzbedarf.
"""
function variable_battery_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    # Erstelle Leistungsvariable für alle Speicher
    ps = PowerModels.var(pm, nw)[:ps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :storage)],
        base_name="$(nw)_ps",
        start = comp_start_value(PowerModels.ref(pm, nw, :storage, i), "ps_start")
    )

    if bounded
        for (i, storage) in PowerModels.ref(pm, nw, :storage)
            # Ladeleistung (negativ): begrenzt durch Ladeleistung
            JuMP.set_lower_bound(ps[i], storage["pmin"])  # z.B. -50 kW
            # Entladeleistung (positiv): begrenzt durch Entladeleistung
            JuMP.set_upper_bound(ps[i], storage["pmax"])  # z.B. +50 kW
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :storage, :ps, PowerModels.ids(pm, nw, :storage), ps)
end

"""
Variable: `e_cap[i]` für alle Speicher i (Optimizable Storage Capacity)

!!! NEUE FUNKTIONALITÄT: SPEICHERGRÖSSEN-OPTIMIERUNG !!!

Definiert die optimierbare Energiekapazität von Batteriespeichern [kWh].
Diese Variable ermöglicht es dem Optimierer, die wirtschaftlich optimale
Speichergröße zu bestimmen, basierend auf:
- Netzkosten (vermiedene Verluste, Überlastungen)
- Investitionskosten (CAPEX in €/kWh)
- Operationelle Vorteile (Flexibilität, Peak Shaving)

Die Speichergröße wird in den Constraints mit dem State of Charge verknüpft,
um physikalisch konsistente Energiebilanzen zu gewährleisten.
"""
function variable_storage_capacity(pm::AbstractPowerModel;
                                   nw::Int = nw_id_default,
                                   bounded::Bool = true,
                                   report::Bool = true)

    stor_ids = PowerModels.ids(pm, nw, :storage)

    # KERNVARIABLE: Optimierbare Energiekapazität in kWh
    e_cap = PowerModels.var(pm, nw)[:e_cap] = JuMP.@variable(pm.model,
        [i in stor_ids],
        base_name = "$(nw)_e_cap",
        lower_bound = 0.0,                       # Minimum: 0 MWh (kein Speicher)
        upper_bound = 1000,                      # Maximum: 10 MWh (anpassbar je nach Studie)
        start = 0.1                              # Startwert: 0.1 MWh (typische Hausbatterie)
    )

    # Setze spezifische Grenzen aus Eingangsdaten falls verfügbar
    if bounded
        for (i, storage) in PowerModels.ref(pm, nw, :storage)
            # Technische Obergrenze (z.B. Platzbeschränkungen)
            if haskey(storage, "energy_rating_max")
                JuMP.set_upper_bound(e_cap[i], storage["energy_rating_max"])
            end
            # Technische Untergrenze (z.B. Mindestgröße für Wirtschaftlichkeit)
            if haskey(storage, "energy_rating_min")
                JuMP.set_lower_bound(e_cap[i], storage["energy_rating_min"])
            end
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :storage, :e_cap, stor_ids, e_cap)
end

"""
Variable: `qs[i]` für alle Speicher i (Storage Reactive Power)

Definiert die Blindleistungs-Variablen für Batteriespeicher [kvar].
Moderne Wechselrichter können unabhängig von der Wirkleistung Blindleistung
bereitstellen, was für lokale Spannungsregelung genutzt werden kann.
"""
function variable_battery_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qs = PowerModels.var(pm, nw)[:qs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :storage)],
        base_name="$(nw)_qs",
        start = comp_start_value(PowerModels.ref(pm, nw, :storage, i), "qs_start")
    )

    if bounded
        for (i, storage) in PowerModels.ref(pm, nw, :storage)
            # Blindleistungsgrenzen (oft symmetrisch um 0)
            JuMP.set_lower_bound(qs[i], storage["qmin"])  # z.B. -30 kvar
            JuMP.set_upper_bound(qs[i], storage["qmax"])  # z.B. +30 kvar
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :storage, :qs, PowerModels.ids(pm, nw, :storage), qs)
end

# =============================================================================
# 5. DEMAND-SIDE MANAGEMENT (DSM) VARIABLEN
# =============================================================================
# DSM modelliert flexible Lasten, die zeitlich verschoben werden können.

"""
Hauptfunktion für DSM-Variablen.
DSM wird als "virtueller Speicher" modelliert, der Lastverschiebungen ermöglicht.
"""
function variable_dsm_storage_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_dsm_storage_power_real(pm; kwargs...)      # Lastverschiebung
    # eDisGo_OPF.variable_dsm_storage_power_imaginary(pm; kwargs...) # Selten verwendet
    eDisGo_OPF.variable_dsm_storage_energy(pm; kwargs...)         # Energiedefizit/-überschuss
end

"""
Variable: `pdsm[i]` für alle DSM-Einheiten i (DSM Power)

Definiert die Leistungsverschiebungs-Variablen für DSM [kW].
Positive Werte = Zusätzlicher Verbrauch (Last erhöhen)
Negative Werte = Verbrauchsreduktion (Last senken)

Beispiele: Waschmaschinen, Geschirrspüler, flexible Industrieprozesse
"""
function variable_dsm_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pdsm = PowerModels.var(pm, nw)[:pdsm] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)],
        base_name="$(nw)_pdsm",
        start = comp_start_value(PowerModels.ref(pm, nw, :dsm, i), "pdsm_start")
    )

    if bounded
        dsm = PowerModels.ref(pm, nw, :dsm)
        for (i, s) in dsm
            # Grenzen für Lastverschiebung
            JuMP.set_lower_bound(pdsm[i], s["p_min"])  # z.B. -20 kW (Lastreduktion)
            JuMP.set_upper_bound(pdsm[i], s["p_max"])  # z.B. +20 kW (Lasterhöhung)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :dsm, :pdsm, PowerModels.ids(pm, nw, :dsm), pdsm)
end

"""
Variable: `qdsm[i]` für alle DSM-Einheiten i (DSM Reactive Power)

Blindleistungskomponente für DSM. Wird selten verwendet, da die meisten
flexiblen Lasten hauptsächlich Wirkleistung verschieben.
"""
function variable_dsm_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qdsm = PowerModels.var(pm, nw)[:qdsm] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)],
        base_name="$(nw)_qdsm",
    )
    if bounded
        dsm = PowerModels.ref(pm, nw, :dsm)
        for (i, s) in dsm
            JuMP.set_lower_bound(qdsm[i], s["q_min"])
            JuMP.set_upper_bound(qdsm[i], s["q_max"])
        end
    end
    report && PowerModels.sol_component_value(pm, nw, :dsm, :qdsm, PowerModels.ids(pm, nw, :dsm), qdsm)
end

"""
Variable: `dsme[i]` für alle DSM-Einheiten i (DSM Energy State)

Definiert den "virtuellen Energieinhalt" für DSM [kWh].
Repräsentiert das akkumulierte Energiedefizit oder -überschuss durch
Lastverschiebungen. Muss über den Optimierungshorizont ausgeglichen werden.

Beispiel: Verschiebung einer 2 kWh-Waschmaschinenladung um 3 Stunden.
"""
function variable_dsm_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    dsme = PowerModels.var(pm, nw)[:dsme] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)],
        base_name="$(nw)_dsme",
        start = comp_start_value(PowerModels.ref(pm, nw, :dsm, i), "dsme_start")
    )

    if bounded
        for (i, dsm) in PowerModels.ref(pm, nw, :dsm)
            # Grenzen für akkumulierte Energieverschiebung
            JuMP.set_lower_bound(dsme[i], dsm["e_min"])  # z.B. -50 kWh
            JuMP.set_upper_bound(dsme[i], dsm["e_max"])  # z.B. +50 kWh
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :dsm, :dsme, PowerModels.ids(pm, nw, :dsm), dsme)
end

# =============================================================================
# 6. WÄRMESYSTEM-VARIABLEN (SECTOR COUPLING)
# =============================================================================
# Diese Sektion modelliert die Kopplung zwischen Strom- und Wärmesektor.

"""
Hauptfunktion für Wärmespeicher-Variablen.
Modelliert thermische Speicher (z.B. Pufferspeicher, Warmwasserspeicher).
"""
function variable_heat_storage(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_heat_storage_power(pm; kwargs...)   # Thermische Leistung
    eDisGo_OPF.variable_heat_storage_energy(pm; kwargs...) # Wärmeinhalt
end

"""
Variable: `phs[i]` für alle Wärmespeicher i (Heat Storage Power)

Definiert die thermische Leistungs-Variablen [kW_th].
Positive Werte = Wärmeentnahme (Heizen)
Negative Werte = Wärmespeicherung

Diese Variable koppelt Wärmespeicher mit Wärmepumpen und Wärmebedarf.
"""
function variable_heat_storage_power(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phs = PowerModels.var(pm, nw)[:phs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)],
        base_name="$(nw)_phs",
        start = comp_start_value(PowerModels.ref(pm, nw, :heat_storage, i), "phs_start")
    )

    if bounded
        for (i, hs) in PowerModels.ref(pm, nw, :heat_storage)
            # Symmetrische Grenzen um Speicherkapazität
            JuMP.set_lower_bound(phs[i], -hs["capacity"])  # Maximale Speicherung
            JuMP.set_upper_bound(phs[i], hs["capacity"])   # Maximale Entnahme
        end
    end
    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :phs, PowerModels.ids(pm, nw, :heat_storage), phs)
end

"""
Variable: `hse[i]` für alle Wärmespeicher i (Heat Storage Energy)

Definiert den thermischen Energieinhalt [kWh_th].
Repräsentiert die gespeicherte Wärmemenge im Speicher.
Wichtig für thermische Trägheit und zeitliche Entkopplung von Erzeugung und Verbrauch.
"""
function variable_heat_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    hse = PowerModels.var(pm, nw)[:hse] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)],
        base_name="$(nw)_hse",
        start = comp_start_value(PowerModels.ref(pm, nw, :heat_storage, i), "hse_start")
    )

    if bounded
        for (i, hs) in PowerModels.ref(pm, nw, :heat_storage)
            JuMP.set_lower_bound(hse[i], 0)               # Minimum: leer
            JuMP.set_upper_bound(hse[i], hs["capacity"])  # Maximum: voll
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :hse, PowerModels.ids(pm, nw, :heat_storage), hse)
end

"""
Hauptfunktion für Wärmepumpen-Variablen.
Modelliert elektrisch betriebene Wärmepumpen als Bindeglied zwischen Strom- und Wärmesektor.
"""
function variable_heat_pump_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_heat_pump_power_real(pm; kwargs...)
    # eDisGo_OPF.variable_heat_pump_power_imaginary(pm; kwargs...)  # Selten benötigt
end

"""
Variable: `php[i]` für alle Wärmepumpen i (Heat Pump Power)

Definiert die elektrische Leistungsaufnahme von Wärmepumpen [kW_el].
Diese Variable verknüpft elektrische Leistung mit thermischer Leistung
über die Leistungszahl (COP = Coefficient of Performance).

Beziehung: P_thermal = COP × P_electrical
Typische COP-Werte: 3-5 (je nach Technologie und Temperaturniveau)
"""
function variable_heat_pump_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    php = PowerModels.var(pm, nw)[:php] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)],
        base_name="$(nw)_php",
        start = comp_start_value(PowerModels.ref(pm, nw, :heatpumps, i), "php_start")
    )

    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            # Elektrische Leistungsgrenzen der Wärmepumpe
            JuMP.set_lower_bound(php[i], hp["p_min"])  # z.B. 0 kW (Mindestlast)
            JuMP.set_upper_bound(php[i], hp["p_max"])  # z.B. 15 kW (Nennleistung)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :php, PowerModels.ids(pm, nw, :heatpumps), php)
end

"""
Variable: `qhp[i]` für alle Wärmepumpen i (Heat Pump Reactive Power)

Blindleistung der Wärmepumpen [kvar].
Moderne Wärmepumpen-Wechselrichter können für Netzdienstleistungen genutzt werden.
"""
function variable_heat_pump_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhp = PowerModels.var(pm, nw)[:qhp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)],
        base_name="$(nw)_qhp",
    )

    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            JuMP.set_lower_bound(qhp[i], hp["q_min"])
            JuMP.set_upper_bound(qhp[i], hp["q_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :qhp, PowerModels.ids(pm, nw, :heatpumps), qhp)
end

# =============================================================================
# 7. ELEKTROMOBILITÄTS-VARIABLEN
# =============================================================================
# Diese Sektion modelliert Elektrofahrzeuge und deren Ladeinfrastruktur.

"""
Hauptfunktion für Ladepunkt-Variablen.
Modelliert flexible Ladestrategien für Elektrofahrzeuge.
"""
function variable_cp_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_cp_power_real(pm; kwargs...)      # Ladeleistung
    # eDisGo_OPF.variable_cp_power_imaginary(pm; kwargs...) # Selten relevant
    eDisGo_OPF.variable_cp_energy(pm; kwargs...)          # Batteriezustand der EVs
end

"""
Variable: `pcp[i]` für alle Ladepunkte i (Charging Point Power)

Definiert die Ladeleistungs-Variablen für Elektrofahrzeuge [kW].
Positive Werte = Laden (Netzbezug)
Zukunft: Negative Werte = Vehicle-to-Grid (V2G, Netzeinspeisung)

Berücksichtigt Ladekurven, Ankunfts-/Abfahrtszeiten und Nutzerbedürfnisse.
"""
function variable_cp_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcp = PowerModels.var(pm, nw)[:pcp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)],
        base_name="$(nw)_pcp",
        start = comp_start_value(PowerModels.ref(pm, nw, :electromobility, i), "pcp_start")
    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            # Ladeleistungsgrenzen (abhängig von Ladegerät und Fahrzeug)
            JuMP.set_lower_bound(pcp[i], cp["p_min"])  # z.B. 0 kW (kein Laden)
            JuMP.set_upper_bound(pcp[i], cp["p_max"])  # z.B. 11 kW (AC) oder 50 kW (DC)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :pcp, PowerModels.ids(pm, nw, :electromobility), pcp)
end

"""
Variable: `qcp[i]` für alle Ladepunkte i (Charging Point Reactive Power)

Blindleistung der Ladeinfrastruktur.
Wird zunehmend für Netzdienstleistungen (Spannungsregelung) genutzt.
"""
function variable_cp_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qcp = PowerModels.var(pm, nw)[:qcp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)],
        base_name="$(nw)_qcp",
    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(qcp[i], cp["q_min"])
            JuMP.set_upper_bound(qcp[i], cp["q_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :qcp, PowerModels.ids(pm, nw, :electromobility), qcp)
end

"""
Variable: `cpe[i]` für alle Ladepunkte i (Charging Point Energy)

Definiert den Batteriezustand der Elektrofahrzeuge [kWh].
Repräsentiert die in der EV-Batterie gespeicherte Energie.
Berücksichtigt Fahrbedürfnisse und gewünschte Abfahrts-SOCs.

Wichtig für: Reichweitenangst, Ladeplanung, V2G-Potentiale
"""
function variable_cp_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    cpe = PowerModels.var(pm, nw)[:cpe] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)],
        base_name="$(nw)_cpe",
        start = comp_start_value(PowerModels.ref(pm, nw, :electromobility, i), "cpe_start")
    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            # Energiegrenzen der EV-Batterie
            JuMP.set_lower_bound(cpe[i], cp["e_min"])  # z.B. 10 kWh (Reserve für Notfall)
            JuMP.set_upper_bound(cpe[i], cp["e_max"])  # z.B. 60 kWh (Batteriekapazität)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :cpe, PowerModels.ids(pm, nw, :electromobility), cpe)
end

# =============================================================================
# 8. SLACK-VARIABLEN FÜR BETRIEBSRESTRIKTIONEN
# =============================================================================
# Slack-Variablen ermöglichen Constraint-Verletzungen gegen Strafe.
# Sie verbessern die numerische Stabilität und identifizieren kritische Situationen.

"""
Hauptfunktion für Netzrestriktions-Slack-Variablen.
Sammelt verschiedene Slack-Variablen für unterschiedliche Netzkomponenten.
"""
function variable_slack_grid_restrictions(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_hp_slack(pm; kwargs...)      # Wärmepumpen-Slack
    eDisGo_OPF.variable_load_slack(pm; kwargs...)    # Last-Slack (Load Shedding)
    eDisGo_OPF.variable_gen_slack(pm; kwargs...)     # Generator-Slack
    eDisGo_OPF.variable_ev_slack(pm; kwargs...)      # Elektromobilitäts-Slack
end

"""
Hauptfunktion für Wärmepumpen- und Wärmespeicher-Slack-Variablen.
Spezielle Slack-Variablen für die Sector-Coupling-Komponenten.
"""
function variable_slack_heat_pump_storage(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_hs_slack(pm; kwargs...)      # Wärmespeicher-Slack
    eDisGo_OPF.variable_hp2_slack(pm; kwargs...)     # Zusätzlicher Wärmepumpen-Slack
end

"""
Variable: `phss[i]` für alle Wärmespeicher i (Heat Storage Slack)

Slack-Variable für Wärmespeicher-Constraints [kW_th].
Ermöglicht temporäre Verletzung thermischer Bilanzgleichungen.
Wird in der Zielfunktion hoch bestraft, um realistische Lösungen zu erzwingen.
"""
function variable_hs_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phss = PowerModels.var(pm, nw)[:phss] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)],
        base_name="$(nw)_phss",
        lower_bound = 0.0  # Slack-Variablen sind immer ≥ 0
    )

    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :phss, PowerModels.ids(pm, nw, :heat_storage), phss)
end

"""
Variable: `phps2[i]` für alle Wärmepumpen i (Heat Pump Secondary Slack)

Zusätzliche Slack-Variable für komplexe Wärmepumpen-Constraints.
Wird verwendet, wenn mehrere thermische Constraints gleichzeitig kritisch werden.
"""
function variable_hp2_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps2 = PowerModels.var(pm, nw)[:phps2] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)],
        base_name="$(nw)_phps2",
        lower_bound = 0.0
    )

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :phps2, PowerModels.ids(pm, nw, :heatpumps), phps2)
end

"""
Variable: `phps[i]` für alle Wärmepumpen i (Heat Pump Primary Slack)

Primäre Slack-Variable für Wärmepumpen-Constraints [kW_el].
Ermöglicht Abweichungen von der optimalen Wärmepumpen-Operation.
Obergrenze basiert auf thermischem Bedarf und COP.
"""
function variable_hp_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps = PowerModels.var(pm, nw)[:phps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)],
        base_name="$(nw)_phps",
        lower_bound = 0.0
    )
    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            # Obergrenze: Maximaler Wärmebedarf geteilt durch COP
            JuMP.set_upper_bound(phps[i], max(hp["pd"]/hp["cop"], 0))
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :phps, PowerModels.ids(pm, nw, :heatpumps), phps)
end

"""
Variable: `pds[i]` für alle Lasten i (Load Shedding Slack)

Slack-Variable für Lastabschaltungen [kW].
Repräsentiert die Menge an Last, die in Notfällen abgeschaltet werden kann.
Wird sehr hoch bestraft, da Lastabschaltung das letzte Mittel ist.

Anwendung: Vermeidung von Netzzusammenbrüchen bei extremen Situationen.
"""
function variable_load_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pds = PowerModels.var(pm, nw)[:pds] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :load)],
        base_name="$(nw)_pds",
        lower_bound = 0.0,
    )

    if bounded
        for (i, load) in PowerModels.ref(pm, nw, :load)
            # Maximum: Komplette Last kann abgeschaltet werden
            JuMP.set_upper_bound(pds[i], load["pd"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :load, :pds, PowerModels.ids(pm, nw, :load), pds)
end

"""
Variable: `pgens[i]` für alle Generatoren i (Generator Slack)

Slack-Variable für Generator-Constraints [kW].
Ermöglicht Abweichungen von optimalen Generator-Fahrplänen.
Wird verwendet bei Generatoren mit komplexen Betriebsrestriktionen.
"""
function variable_gen_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pgens = PowerModels.var(pm, nw)[:pgens] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen)],
        base_name="$(nw)_pgens",
        lower_bound = 0.0,
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen)
            JuMP.set_upper_bound(pgens[i], gen["pg"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen, :pgens, PowerModels.ids(pm, nw, :gen), pgens)
end

"""
Variable: `pcps[i]` für alle Ladepunkte i (EV Charging Slack)

Slack-Variable für Elektromobilitäts-Constraints [kW].
Ermöglicht Abweichungen von optimalen Ladestrategien.
Wird verwendet, wenn Ladebedarfe und Netzrestriktionen kollidieren.
"""
function variable_ev_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcps = PowerModels.var(pm, nw)[:pcps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)],
        base_name="$(nw)_pcps",
        lower_bound = 0.0,
    )

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :pcps, PowerModels.ids(pm, nw, :electromobility), pcps)
end

# =============================================================================
# 9. SLACK-GENERATOR-VARIABLEN
# =============================================================================
# Spezielle Generatoren für Leistungsbilanz-Ausgleich.

"""
Hauptfunktion für Slack-Generator-Variablen.
Definiert sowohl aktive als auch reaktive Slack-Generation.
"""
function variable_slack_gen(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_slack_gen_real(pm; kwargs...)      # Wirk-Slack
    eDisGo_OPF.variable_slack_gen_imaginary(pm; kwargs...) # Blind-Slack
end

"""
Variable: `pgs[i]` für alle Slack-Generatoren i (Slack Generator Real Power)

Wirkleistungs-Slack-Generatoren [kW].
Gleichen Leistungsbilanzen aus, wenn andere Generatoren/Flexibilitäten
nicht ausreichen. Repräsentieren Verbindung zum übergeordneten Netz.
"""
function variable_slack_gen_real(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    pgs = PowerModels.var(pm, nw)[:pgs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_slack)],
        base_name="$(nw)_pgs"
    )
    report && PowerModels.sol_component_value(pm, nw, :gen_slack, :pgs, PowerModels.ids(pm, nw, :gen_slack), pgs)
end

"""
Variable: `qgs[i]` für alle Slack-Generatoren i (Slack Generator Reactive Power)

Blindleistungs-Slack-Generatoren [kvar].
Für Spannungsregelung und Blindleistungsbilanz.
"""
function variable_slack_gen_imaginary(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    qgs = PowerModels.var(pm, nw)[:qgs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_slack)],
        base_name="$(nw)_qgs"
    )
    report && PowerModels.sol_component_value(pm, nw, :gen_slack, :qgs, PowerModels.ids(pm, nw, :gen_slack), qgs)
end

# =============================================================================
# 10. HOCHSPANNUNGS-ANFORDERUNGEN (OVERLYING GRID)
# =============================================================================
# Variablen für Koordination mit dem übergeordneten Hochspannungsnetz.

"""
Hauptfunktion für HV-Anforderungs-Variablen.
Modelliert Flexibilitätsanforderungen aus dem übergeordneten Netz.
"""
function variable_slack_HV_requirements(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_slack_HV_requirements_real(pm; kwargs...)
    # eDisGo_OPF.variable_slack_HV_requirements_imaginary(pm; kwargs...)  # Selten verwendet
end

"""
Variable: `phvs[i]` für alle HV-Anforderungen i (HV Requirement Slack)

Slack-Variablen für Hochspannungsanforderungen [kW].
Ermöglichen Abweichungen von Sollwerten des übergeordneten Netzbetreibers.
Werden quadratisch bestraft, um große Abweichungen zu vermeiden.

Anwendung: TSO-DSO-Koordination, Systemdienstleistungen, Redispatch
"""
function variable_slack_HV_requirements_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phvs = PowerModels.var(pm, nw)[:phvs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :HV_requirements)],
        base_name="$(nw)_phvs",
        # Bewusst keine festen Grenzen gesetzt - werden dynamisch in Zielfunktion behandelt
        #lower_bound = -100,
        #upper_bound = 100
    )

    report && PowerModels.sol_component_value(pm, nw, :HV_requirements, :phvs, PowerModels.ids(pm, nw, :HV_requirements), phvs)
end

"""
Variable: `qhvs[i]` für alle HV-Anforderungen i (HV Requirement Reactive Slack)

Blindleistungs-Slack für Hochspannungsanforderungen [kvar].
Für erweiterte Spannungs- und Blindleistungskoordination zwischen Netzebenen.
"""
function variable_slack_HV_requirements_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhvs = PowerModels.var(pm, nw)[:qhvs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :HV_requirements)],
        base_name="$(nw)_qhvs",
    )

    report && PowerModels.sol_component_value(pm, nw, :HV_requirements, :qhvs, PowerModels.ids(pm, nw, :HV_requirements), qhvs)
end

# =============================================================================
# ENDE DER VARIABLEN-DEFINITIONEN
# =============================================================================
# Diese variables.jl definiert ein umfassendes Set von Optimierungsvariablen
# für moderne Verteilnetze mit hohem Anteil dezentraler, flexibler Ressourcen.
#
# Hauptkomponenten:
# 1. Netzwerk-Physik: Leistungsflüsse, Spannungen, Leitungsbelastungen
# 2. Flexibilitäten: Batteriespeicher (mit Größenoptimierung), DSM, Sector Coupling
# 3. Elektromobilität: Flexible Ladestrategien für EVs
# 4. Robustheit: Umfassende Slack-Variablen für numerische Stabilität
# 5. Netzebenen-Koordination: HV-DSO-Integration
#
# Die Speichergrößenoptimierung (variable_storage_capacity) ist eine neue
# Funktionalität, die simultane Investitions- und Betriebsoptimierung ermöglicht.
# =============================================================================
