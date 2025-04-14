# Funktion zur Minimierung der Leitungsverluste
function objective_min_losses(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs aus dem Modell
    nws = PowerModels.nw_ids(pm)

    # Erstelle ein Dictionary, das für jedes Netzwerk die Variable :ccm enthält
    # (z.B. ein Maß für den Leitungsstrom oder Spannungsbezug)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Für jedes Netzwerk: Erstelle ein Dictionary, das für jede Branch-ID den Widerstand (br_r) abruft
    # Falls kein Wert vorhanden ist, wird der Standardwert 1.0 verwendet.
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionaries für die aktiven (p) und reaktiven (q) Leistungsvariablen pro Netzwerk
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)

    # Erstelle ein Dictionary für die Länge jeder Leitung (Standardwert: 1.0)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle ein Dictionary für die Kosten jeder Leitung (Standardwert: 1.0)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle ein Dictionary für die nominale Kapazität (rate_a) der Leitungen (Standardwert: 1.0)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle ein Dictionary für den "storage"-Parameter jedes Zweigs aus dem ersten Netzwerk
    storage = Dict(i => get(branch, "storage", 1.0) for (i, branch) in PowerModels.ref(pm, 1, :branch))

    # Extrahiere positive Widerstandswerte als Parameter (aus dem ersten Netzwerk)
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]

    # Definiere das Optimierungsziel: Minimierung der Leitungsverluste
    # Es wird über alle Netzwerke und alle „Arcs“ (gerichtete Leitungssegmente) summiert,
    # wobei nur Leitungen ohne Speicherelemente berücksichtigt werden.
    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws)
        # Optional: Ein weiterer Term könnte hinzugefügt werden, um die Leitungslast zusätzlich zu bestrafen.
    )
end

# Funktion zur Minimierung von Leitungsverlusten und gleichzeitiger Bestrafung von Slack-Variablen (z.B. Lastabschaltung, Kürzungen)
function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # Erstelle Dictionary für die :ccm-Variablen pro Netzwerk
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Erstelle Dictionary für Widerstände jeder Branch (Standardwert: 1.0)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionaries für verschiedene Slack-Variablen:
    # pgc: Nicht-dispatchable (nicht steuerbare) Kürzungen
    # pgens: Dispatchable (steuerbare) Kürzungen
    # pds: Lastabschaltungen (Load Shedding)
    # pcps: Kürzungen im Bereich cp (z.B. Elektromobilität)
    # phps, phps2: Kürzungen bei Wärmepumpen
    # phss: Weitere Slack-Variable für Wärmepumpen
    pgc   = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds   = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps  = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps  = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)
    phss  = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)

    # Setze einen Gewichtungsfaktor für die Slack-Terme
    factor_slacks = 0.6

    # Definiere das Ziel:
    # - Ein Anteil ((1 - factor_slacks)) wird der Minimierung der Leitungsverluste zugeordnet.
    # - Der Anteil factor_slacks bestraft verschiedene Slack-Variablen (Kürzungen, Load Shedding etc.).
    return JuMP.@objective(pm.model, Min,
        (1 - factor_slacks) * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm, 1, :gen_nd))) for n in nws)
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm, 1, :gen))) for n in nws)
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm, 1, :load))) for n in nws)
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm, 1, :electromobility))) for n in nws)
        + factor_slacks  * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
    )
end

# Funktion zur Minimierung der maximalen Linienbelastung kombiniert mit Leitungsverlusten
function objective_min_line_loading_max(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # Erstelle Dictionary für die :ccm-Variablen
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Erstelle Dictionary für Widerstände der Branches
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Variable für die aktuelle oder maximale Linienbelastung (ll)
    ll = PowerModels.var(pm, 1, :ll)

    # Erstelle Dictionaries für die Länge und Kosten der Leitungen
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle ein Dictionary für den Speicherparameter jeder Branch (wird zur Filterung genutzt)
    storage = Dict(i => get(branch, "storage", 1.0) for (i, branch) in PowerModels.ref(pm, 1, :branch))

    # Gewichtungsfaktor für den Linienbelastungsterm
    factor_ll = 0.1

    # Definiere das Ziel:
    # - Der erste Term minimiert die Leitungsverluste.
    # - Der zweite Term minimiert die Abweichung der Linienbelastung von 1 (also (ll - 1)),
    #   gewichtet mit Kosten und Länge der jeweiligen Leitung, jedoch nur für Leitungen ohne Speicher.
    return JuMP.@objective(pm.model, Min,
        (1 - factor_ll) * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)
        + factor_ll * sum((ll[(b, i, j)] - 1) * c[1][b] * l[1][b] for (b, i, j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)
    )
end

# Funktion zur Minimierung von Leitungsverlusten, Slack-Strafen und zusätzlichen Hochspannungsanforderungen (Overlying Grid, OG)
function objective_min_losses_slacks_OG(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # Erstelle Dictionary für :ccm-Variablen
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Erstelle Dictionary für Widerstände jeder Branch
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle Dictionaries für die Slack-Variablen (analog zur vorherigen Funktion)
    pgc   = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds   = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps  = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps  = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)
    phss  = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)

    # Zusätzliche Slack-Variable für Hochspannungskomponenten
    phvs  = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    # Berechne einen Skalierungsfaktor für HV-Slack-Strafen basierend auf den Widerstandswerten
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]
    factor_hv_slacks = exp10(floor(log10(maximum(parameters))) + 1)

    # Gewichtungsfaktor für alle Slack-Terme
    factor_slacks = 0.6

    # Definiere das Ziel:
    # - Der erste Term minimiert die Leitungsverluste.
    # - Die folgenden Terme bestrafen verschiedene Arten von Slack (Kürzungen, Load Shedding, etc.).
    # - Zusätzlich werden HV-Anforderungen mittels der quadrierten phvs-Variablen bestraft.
    #   Für Elemente, deren Name "dsm" ist, wird ein geringerer Straffaktor (1e-1) verwendet.
    # - Ein hoher Strafterm (1e4) bestraft zusätzlich bestimmte Wärmepumpen-Slackvariablen.
    return JuMP.@objective(pm.model, Min,
        (1 - factor_slacks) * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm, 1, :gen_nd))) for n in nws)
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm, 1, :gen))) for n in nws)
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm, 1, :load))) for n in nws)
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm, 1, :electromobility))) for n in nws)
        + factor_slacks  * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 * flex["count"]
                                     for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] != "dsm") for n in nws)
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 * flex["count"]
                                            for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] == "dsm") for n in nws)
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
    )
end

# Funktion zur Minimierung der Linienbelastung unter Berücksichtigung von HV-Anforderungen (Overlying Grid, OG)
function objective_min_line_loading_max_OG(pm::AbstractBFModelEdisgo)
    # Erhalte alle Netzwerk-IDs
    nws = PowerModels.nw_ids(pm)

    # Erstelle Dictionary für :ccm-Variablen
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)

    # Erstelle Dictionary für Widerstände jeder Branch
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Variable für Linienbelastung (ll)
    ll = PowerModels.var(pm, 1, :ll)

    # Erstelle Dictionaries für Länge und Kosten der Leitungen
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i, branch) in PowerModels.ref(pm, n, :branch)) for n in nws)

    # Erstelle ein Dictionary für den Speicherparameter jeder Branch
    storage = Dict(i => get(branch, "storage", 1.0) for (i, branch) in PowerModels.ref(pm, 1, :branch))

    # Zusätzliche Variable für HV-Anforderungen (phvs)
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    # Extrahiere positive Widerstandswerte aus dem ersten Netzwerk
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .> 0]

    # Berechne Parameter, die sich aus Länge multipliziert mit Kosten ergeben (nur positive Werte)
    parameters2 = [l[1][i] * c[1][i] for i in keys(c[1])]
    parameters2 = parameters2[parameters2 .> 0]

    # Gewichtungsfaktor für den Linienbelastungsterm
    factor_ll = 0.01
    println(factor_ll)

    # Dynamische Berechnung des HV-Faktors basierend auf den Skalen von l*c
    factor_hv_slacks = 7.5 * exp10(floor(log10(maximum(0.01 * parameters2))) + 1)
    println(factor_hv_slacks)

    # Definiere das Ziel:
    # - Der erste Term minimiert die Leitungsverluste.
    # - Der zweite Term minimiert die Abweichung der Linienbelastung ((ll - 1)),
    #   gewichtet mit Kosten und Länge, nur für Leitungen ohne Speicher.
    # - Anschließend werden HV-Anforderungen über die quadrierten phvs-Variablen bestraft,
    #   wobei ein geringerer Faktor (1e-1) für Elemente mit "dsm" genutzt wird.
    return JuMP.@objective(pm.model, Min,
        (1 - factor_ll) * sum(sum(ccm[n][b] * r[n][b] for (b, i, j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)
        + factor_ll * sum((ll[(b, i, j)] - 1) * c[1][b] * l[1][b] for (b, i, j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] != "dsm") for n in nws)
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] == "dsm") for n in nws)
    )
end
