# =============================================================================
# ZIELFUNKTIONEN FÜR EDISGO OPTIMAL POWER FLOW
# =============================================================================
# Diese Datei enthält verschiedene Zielfunktionen für die Optimierung von
# Energiesystemen mit unterschiedlichen Schwerpunkten:
# - Minimierung von Leitungsverlusten
# - Minimierung von Slack-Variablen (Lastabschaltungen, Kürzungen)
# - Minimierung der maximalen Leitungsbelastung
# - Berücksichtigung von Hochspannungsanforderungen (HV/Overlying Grid)
# - Optimierung der Speichergröße (neue Funktionalität)
# =============================================================================

"""
Funktion zur Minimierung der Leitungsverluste

Diese Zielfunktion minimiert die elektrischen Verluste in allen Leitungen des Netzes.
Die Verluste werden als Produkt aus Stromquadrat (ccm) und Widerstand (br_r) berechnet.
Nur Leitungen ohne Speicherelemente werden berücksichtigt.

Parameter:
- pm: PowerModel Objekt mit allen Netzwerkdaten und Variablen
"""
function objective_min_losses(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs aus dem Modell (für Multiperioden-Optimierung)
    nws = PowerModels.nw_ids(pm)

    # Erstelle ein Dictionary, das für jedes Netzwerk die Variable :ccm enthält
    # ccm = Strommagnituden-Quadrat für jede Leitung (I²)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Für jedes Netzwerk: Erstelle ein Dictionary für Leitungswiderstände (br_r)
    # Falls kein Widerstandswert vorhanden ist, wird Standardwert 1.0 verwendet
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionaries für aktive (p) und reaktive (q) Leistungsvariablen
    # Diese werden hier definiert aber nicht in der Zielfunktion verwendet
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)

    # Erstelle Dictionary für Leitungslängen (für potentielle Erweiterungen)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionary für Leitungskosten (für potentielle Erweiterungen)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionary für nominale Leitungskapazitäten (rate_a)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionary für Speicher-Flags aus dem ersten Netzwerk
    # storage = 0: normale Leitung, storage ≠ 0: Speicher-Leitung
    storage = Dict(i => get(branch, "storage", 1.0) for (i, branch) in PowerModels.ref(pm, 1, :branch))

    # Extrahiere positive Widerstandswerte für Skalierungsberechnungen
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]

    # ZIELFUNKTION: Minimierung der gesamten Leitungsverluste
    # Verluste = Σ(I² × R) für alle Leitungen ohne Speicher in allen Zeitperioden
    # Formel: P_loss = I² × R (Ohm'sches Gesetz für Verlustleistung)
    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws)
        # Hinweis: Weitere Terme könnten hinzugefügt werden, z.B. für Leitungsbelastung
    )
end

"""
Funktion zur Minimierung von Leitungsverlusten mit Bestrafung von Slack-Variablen

Diese erweiterte Zielfunktion kombiniert die Minimierung von Leitungsverlusten
mit der Bestrafung verschiedener "Slack"-Variablen, die Verletzungen von
Betriebsregeln oder nicht-ideale Betriebszustände repräsentieren.

Slack-Variablen repräsentieren:
- pgc: Kürzung erneuerbarer Energien (Generation Curtailment)
- pgens: Kürzung steuerbarer Generatoren
- pds: Lastabschaltungen (Load Shedding)
- pcps: Elektromobilitäts-Kürzungen
- phps/phps2: Wärmepumpen-Kürzungen
- phss: Zusätzliche Wärmepumpen-Slack-Variable
"""
function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs für Multiperioden-Optimierung
    nws = PowerModels.nw_ids(pm)

    # Erstelle Dictionary für Strommagnituden-Quadrat (:ccm) pro Netzwerk
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Erstelle Dictionary für Leitungswiderstände pro Netzwerk
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # === SLACK-VARIABLEN DEFINITIONEN ===
    # pgc: Kürzung nicht-steuerbarer Generatoren (z.B. PV, Wind)
    pgc   = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)

    # pgens: Kürzung steuerbarer Generatoren
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)

    # pds: Lastabschaltungen (Load Shedding) - Notfallmaßnahme
    pds   = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)

    # pcps: Kürzungen bei Elektromobilität (Charging Points)
    pcps  = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)

    # phps: Erste Wärmepumpen-Slack-Variable
    phps  = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)

    # phps2: Zweite Wärmepumpen-Slack-Variable (für komplexere Modelle)
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)

    # phss: Zusätzliche Wärmepumpen-Slack-Variable (z.B. für Wärmespeicher)
    phss  = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)

    # Gewichtungsfaktor zwischen Verlusten und Slack-Bestrafung
    # 0.6 bedeutet: 60% Slack-Bestrafung, 40% Verlustminimierung
    factor_slacks = 0.6

    # KOMBINIERTE ZIELFUNKTION:
    # Term 1: (1-factor_slacks) × Leitungsverluste
    # Term 2-7: factor_slacks × verschiedene Slack-Variablen
    # Term 8: Hohe Bestrafung (1e4) für kritische Wärmepumpen-Slacks
    return JuMP.@objective(pm.model, Min,
        # Hauptterm: Gewichtete Leitungsverluste (40% bei factor_slacks=0.6)
        (1 - factor_slacks) * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)

        # Bestrafung für Kürzung erneuerbarer Energien
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm, 1, :gen_nd))) for n in nws)

        # Bestrafung für Kürzung steuerbarer Generatoren
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm, 1, :gen))) for n in nws)

        # Bestrafung für Lastabschaltungen (sehr unerwünscht)
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm, 1, :load))) for n in nws)

        # Bestrafung für Elektromobilitäts-Kürzungen
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm, 1, :electromobility))) for n in nws)

        # Bestrafung für Wärmepumpen-Kürzungen
        + factor_slacks  * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)

        # Sehr hohe Bestrafung für kritische Wärmepumpen-Verletzungen
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
    )
end

"""
Funktion zur Minimierung der maximalen Leitungsbelastung mit Speichergrößenoptimierung

Diese Zielfunktion kombiniert drei Hauptziele:
1. Minimierung der Leitungsverluste
2. Minimierung der maximalen Leitungsbelastung (Netzausbau vermeiden)
3. Optimierung der Speichergröße mit Investitionskosten (NEUE FUNKTIONALITÄT)

Die Speichergrößenoptimierung ermöglicht es dem System, die optimale
Energiekapazität (MWh) für Batteriespeicher zu bestimmen.
"""
function objective_min_line_loading_max(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # Erstes Netzwerk für einige spezifische Variablen
    nw1 = first(nws)

    # === STANDARD-VARIABLEN ===
    # Strommagnituden-Quadrat für Verlustberechnung
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Leitungswiderstände für alle Netzwerke
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Leitungsbelastungsvariable (Line Loading) - nur für erstes Netzwerk
    ll = PowerModels.var(pm, nw1, :ll)

    # Leitungslängen und -kosten für Belastungsberechnung
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # === SPEICHERGRÖSSENOPTIMIERUNG (NEUE FUNKTIONALITÄT) ===
    # Prüfe ob Speicherkapazitätsvariable existiert
    if !haskey(PowerModels.var(pm, nw1), :e_cap)
        @warn "Speicherkapazitätsvariable :e_cap nicht gefunden. Optimierung ohne Speicherkosten."
        # Fallback: Zielfunktion ohne Speicherkosten
        return JuMP.@objective(pm.model, Min,
            # Nur Verluste und Leitungsbelastung
            0.9 * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)
            + 0.1 * sum(
                (ll[(b, i, j)] - 1) * c[nw1][b] * l[nw1][b]
                for (b, i, j) in PowerModels.ref(pm, nw1, :arcs_from)
                if get(PowerModels.ref(pm, nw1, :branch, b), "storage", 0.0) == 0
            )
        )
    end

    # Hole Speicherkapazitätsvariable (optimierbare Speichergröße in MWh)
    e_cap = PowerModels.var(pm, nw1, :e_cap)

    # Hole alle Speicher-IDs
    storage_ids = PowerModels.ids(pm, nw1, :storage)

    # Investitionskosten für Speicher (€/MWh)
    # Dieser Wert kann angepasst werden je nach Marktsituation
    storage_capex_per_kwh = 1000000  # €/MWh - typische Batteriekosten 2024/2025 (Lithium-Ionen, stationär)

    # === GEWICHTUNGSFAKTOREN ===
    # Diese Faktoren bestimmen die relative Wichtigkeit der drei Zielkomponenten
    factor_losses = 0.4        # 40% - Verlustminimierung
    factor_line_loading = 0.1  # 10% - Leitungsbelastung
    factor_storage = 0.5       # 50% - Speicherinvestitionen

    # Validierung: Faktoren sollten sich zu 1.0 summieren
    total_factors = factor_losses + factor_line_loading + factor_storage
    if abs(total_factors - 1.0) > 1e-6
        @warn "Gewichtungsfaktoren summieren sich zu $total_factors statt 1.0"
    end

    # === KOMBINIERTE ZIELFUNKTION ===
    return JuMP.@objective(pm.model, Min,
        # Term 1: Leitungsverluste minimieren (I² × R)
        # Reduziert Energieverluste und Betriebskosten
        factor_losses * sum(sum(ccm[n][b] * r[n][b]
                               for (b, i, j) in PowerModels.ref(pm, n, :arcs_from))
                           for n in nws)

        # Term 2: Maximale Leitungsbelastung minimieren
        # (ll - 1) gibt Überlastung an: ll=1 bedeutet 100% Auslastung
        # Nur für Leitungen ohne Speicher, gewichtet mit Kosten×Länge
        + factor_line_loading * sum(
            (ll[(b, i, j)] - 1) * c[nw1][b] * l[nw1][b]
            for (b, i, j) in PowerModels.ref(pm, nw1, :arcs_from)
            if get(PowerModels.ref(pm, nw1, :branch, b), "storage", 0.0) == 0
        )

        # Term 3: Speicherinvestitionskosten (NEUE FUNKTIONALITÄT)
        # Minimiert CAPEX für Batteriespeicher: Kosten = Kapazität[MWh] × Preis[€/MWh]
        # Dies ermöglicht optimale Dimensionierung zwischen Nutzen und Investition
        + factor_storage * sum(e_cap[i] * storage_capex_per_kwh for i in storage_ids))

        # Strafkosten für Abweichungen von HV‑Leistungs‑Vorgaben
        # + HV_SLACK_COST * (sum(hv_slack_pos) + sum(hv_slack_neg)))
end

"""
Funktion zur Minimierung von Leitungsverlusten, Slack-Strafen und
Hochspannungsanforderungen (Overlying Grid - OG)

Diese erweiterte Zielfunktion berücksichtigt zusätzlich zu den Standard-Slack-Variablen
auch Anforderungen aus dem übergeordneten Hochspannungsnetz (Overlying Grid).
Dies ist relevant für die Koordination zwischen verschiedenen Spannungsebenen.
"""
function objective_min_losses_slacks_OG(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # === STANDARD-VARIABLEN ===
    # Strommagnituden-Quadrat für Verlustberechnung
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Leitungswiderstände
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # === STANDARD-SLACK-VARIABLEN ===
    pgc   = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)    # Generation Curtailment
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)  # Generator Slack
    pds   = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)    # Load Shedding
    pcps  = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)   # Charging Point Slack
    phps  = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)   # Heat Pump Slack 1
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)  # Heat Pump Slack 2
    phss  = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)   # Heat Storage Slack

    # === HOCHSPANNUNGS-SLACK-VARIABLEN (OG = OVERLYING GRID) ===
    # phvs: Slack-Variablen für Anforderungen aus dem übergeordneten HV-Netz
    phvs  = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    # === DYNAMISCHE SKALIERUNGSFAKTOREN ===
    # Berechne Skalierungsfaktor für HV-Slack basierend auf Netzwerkparametern
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]

    # Logarithmische Skalierung basierend auf maximalen Widerstandswerten
    # Dies sorgt für angemessene Gewichtung relativ zu anderen Termen
    factor_hv_slacks = exp10(floor(log10(maximum(parameters))) + 1)

    # Gewichtungsfaktor für Standard-Slack-Variablen
    factor_slacks = 0.6

    # === ERWEITERTE ZIELFUNKTION ===
    return JuMP.@objective(pm.model, Min,
        # Term 1: Leitungsverluste (40% bei factor_slacks=0.6)
        (1 - factor_slacks) * sum(sum(ccm[n][b] * r[n][b]
                                      for (b, i, j) in PowerModels.ref(pm, n, :arcs_from))
                                  for n in nws)

        # Terme 2-7: Standard-Slack-Bestrafung (je 60% Anteil)
        + factor_slacks * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm, 1, :gen_nd))) for n in nws)
        + factor_slacks * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm, 1, :gen))) for n in nws)
        + factor_slacks * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm, 1, :load))) for n in nws)
        + factor_slacks * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm, 1, :electromobility))) for n in nws)
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)

        # Terme 8-9: HV-ANFORDERUNGEN (OVERLYING GRID)
        # Quadratische Bestrafung für bessere Konvergenz
        # Unterschiedliche Gewichtung für DSM vs. andere Flexibilitäten
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 * flex["count"]
                                     for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)
                                     if flex["name"] != "dsm") for n in nws)

        # Geringere Bestrafung für DSM-basierte HV-Anforderungen (Faktor 1e-1)
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 * flex["count"]
                                            for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)
                                            if flex["name"] == "dsm") for n in nws)

        # Term 10: Sehr hohe Bestrafung für kritische Wärmepumpen-Verletzungen
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
    )
end

"""
Funktion zur Minimierung der Leitungsbelastung mit Berücksichtigung von
Hochspannungsanforderungen (Overlying Grid - OG)

Diese Zielfunktion fokussiert sich auf die Minimierung der maximalen Leitungsbelastung
unter gleichzeitiger Berücksichtigung von Anforderungen aus dem übergeordneten
Hochspannungsnetz. Sie ist besonders geeignet für Szenarien, in denen Netzausbau
vermieden werden soll.
"""
function objective_min_line_loading_max_OG(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # === STANDARD-VARIABLEN ===
    # Strommagnituden-Quadrat für Verlustberechnung
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Leitungswiderstände für alle Netzwerke
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Leitungsbelastungsvariable (nur erstes Netzwerk)
    ll = PowerModels.var(pm, 1, :ll)

    # Leitungsparameter für Belastungsberechnung
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Speicher-Flag für Leitungsfilterung
    storage = Dict(i => get(branch, "storage", 1.0) for (i, branch) in PowerModels.ref(pm, 1, :branch))

    # === HOCHSPANNUNGS-VARIABLEN ===
    # HV-Slack-Variablen für übergeordnete Netzanforderungen
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    # === DYNAMISCHE PARAMETERBERECHNUNG ===
    # Extrahiere positive Widerstandswerte für Skalierung
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]

    # Berechne kombinierte Kostenparameter (Länge × Kosten)
    # Diese werden für die Gewichtung der Leitungsbelastung verwendet
    parameters2 = [l[1][i] * c[1][i] for i in keys(c[1])]
    parameters2 = parameters2[parameters2 .> 0]

    # === GEWICHTUNGSFAKTOREN ===
    # Geringer Faktor für Leitungsbelastung (1% der Gesamtfunktion)
    factor_ll = 0.01
    println("Line Loading Factor: ", factor_ll)

    # Dynamischer HV-Faktor basierend auf Netzwerkskalen
    # 7.5 ist ein empirischer Skalierungsfaktor
    factor_hv_slacks = 7.5 * exp10(floor(log10(maximum(0.01 * parameters2))) + 1)
    println("HV Slacks Factor: ", factor_hv_slacks)

    # === SPEZIALISIERTE ZIELFUNKTION ===
    return JuMP.@objective(pm.model, Min,
        # Term 1: Leitungsverluste (99% bei factor_ll=0.01)
        # Hauptfokus auf Effizienz der Energieübertragung
        (1 - factor_ll) * sum(sum(ccm[n][b] * r[n][b]
                                  for (b, i, j) in PowerModels.ref(pm, n, :arcs_from))
                              for n in nws)

        # Term 2: Leitungsbelastung (1% bei factor_ll=0.01)
        # Minimiert Überlastung: (ll-1) = 0 bedeutet 100% Auslastung
        # Gewichtet mit Leitungskosten und -länge, nur für Nicht-Speicher-Leitungen
        + factor_ll * sum((ll[(b, i, j)] - 1) * c[1][b] * l[1][b]
                          for (b, i, j) in PowerModels.ref(pm, 1, :arcs_from)
                          if storage[b] == 0)

        # Term 3: HV-Anforderungen für Nicht-DSM-Flexibilitäten
        # Quadratische Bestrafung für bessere mathematische Eigenschaften
        + factor_hv_slacks * sum(sum(phvs[n][i]^2
                                     for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)
                                     if flex["name"] != "dsm") for n in nws)

        # Term 4: HV-Anforderungen für DSM-Flexibilitäten
        # Geringere Gewichtung (1e-1 = 10%) da DSM flexibler ist
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2
                                            for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)
                                            if flex["name"] == "dsm") for n in nws)
    )
end

# =============================================================================
# ENDE DER ZIELFUNKTIONEN
# =============================================================================
# Verwendungshinweise:
# 1. objective_min_losses: Reine Verlustminimierung
# 2. objective_min_losses_slacks: Verluste + Slack-Bestrafung
# 3. objective_min_line_loading_max: Verluste + Leitungsbelastung + SPEICHEROPTIMIERUNG
# 4. objective_min_losses_slacks_OG: Vollständige Funktion mit HV-Koordination
# 5. objective_min_line_loading_max_OG: Leitungsbelastung + HV-Koordination
#
# Die Funktion objective_min_line_loading_max enthält die neue Funktionalität
# zur Speichergrößenoptimierung, die in der Konversation entwickelt wurde.
# =============================================================================
