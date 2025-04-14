# Diese Funktion setzt Startwerte für verschiedene Variablen in einem AC-Base-Flow-Modell.
# Dabei werden für unterschiedliche Netzkomponenten (Generatoren, DSM, Storage, Elektromobilität, Wärmepumpen und Wärmespeicher)
# die Startwerte (mit dem Suffix _start) auf die aktuellen Werte gesetzt.
function set_ac_bf_start_values!(network::Dict{String,<:Any})
    # Für alle nicht-dispatchable Generatoren ("gen_nd") wird der Startwert "pgc_start" gesetzt.
    for (i, gen) in network["gen_nd"]
        gen["pgc_start"] = gen["pgc"]
    end

    # Für alle Slack-Generatoren ("gen_slack") werden die Startwerte für aktive und reaktive Leistungen gesetzt.
    for (i, gen) in network["gen_slack"]
        gen["pgs_start"] = gen["pgs"]
        gen["qgs_start"] = gen["qgs"]
    end

    # Für alle DSM-Elemente (Demand Side Management) werden die Startwerte für die aktiven (pdsm) und reaktiven (dsme) DSM-Leistungen gesetzt.
    for (i, dsm) in network["dsm"]
        dsm["pdsm_start"] = dsm["pdsm"]
        dsm["dsme_start"] = dsm["dsme"]
    end

    # Für alle Speicherelemente im Netzwerk werden die Startwerte für die momentane Leistung (ps) und den Energiespeicher (se) gesetzt.
    for (i, s) in network["storage"]
        s["ps_start"] = s["ps"]
        s["se_start"] = s["se"]
    end

    # Für alle Elemente der Elektromobilität wird der Startwert "pcp_start" gesetzt.
    for (i, cp) in network["electromobility"]
        cp["pcp_start"] = cp["pcp"]
    end

    # Für alle Wärmepumpen wird der Startwert "php_start" gesetzt.
    for (i, hp) in network["heatpumps"]
        hp["php_start"] = hp["php"]
    end

    # Für alle Wärmespeicherelemente werden die Startwerte für die aktive Wärmespeicherung (phs) und den Energiespeicher (hse) gesetzt.
    for (i, hs) in network["heat_storage"]
        hs["phs_start"] = hs["phs"]
        hs["hse_start"] = hs["hse"]
    end
end

"""
Überprüft, ob die Bustypen für eine Power-Flow-Studie geeignet sind und korrigiert sie, falls notwendig.

Die Hauptüberprüfungen sind:
  - Alle Busse vom Typ 2 (PV) müssen mindestens einen aktiven, angeschlossenen Generator besitzen.
  - Es darf genau ein Bus vom Typ 3 (Slack-Bus) mit einem aktiven Generator vorhanden sein.

Voraussetzung ist, dass das Netzwerk aus einer einzigen verbundenen Komponente besteht.
"""
function correct_bus_types!(data::Dict{String,<:Any})
    # Ruft intern die Korrekturfunktion _correct_bus_types! über den Wrapper apply_pm! auf.
    apply_pm!(eDisGo_OPF._correct_bus_types!, data)
end

# Diese Funktion übernimmt die eigentliche Logik zur Korrektur der Bustypen.
function _correct_bus_types!(pm_data::Dict{String,<:Any})
    # Erstelle ein Dictionary, in dem für jeden Bus (identifiziert durch "index") zunächst eine leere Liste gespeichert wird.
    # In dieser Liste werden später die Indizes der aktiven Generatoren, die an diesem Bus angeschlossen sind, abgelegt.
    bus_gens = Dict(bus["index"] => [] for (i, bus) in pm_data["bus"])

    # Durchlaufe alle Generatoren im Abschnitt "gen" und speichere aktive Generatoren (gen_status ungleich 0)
    for (i, gen) in pm_data["gen"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end

    # Wiederhole den Vorgang für nicht-dispatchable Generatoren ("gen_nd")
    for (i, gen) in pm_data["gen_nd"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end

    # Und für Slack-Generatoren ("gen_slack")
    for (i, gen) in pm_data["gen_slack"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end

    # Variable, die angibt, ob bereits ein Slack-Bus (Typ 3) gefunden wurde
    slack_found = false

    # Durchlaufe alle Busse im Netzwerk und prüfe bzw. korrigiere den Bustyp
    for (i, bus) in pm_data["bus"]
        idx = bus["index"]
        if bus["bus_type"] == 1  # PQ-Bus
            # Falls an einem PQ-Bus aktive Generatoren vorhanden sind,
            # könnte man den Bus theoretisch zu PV (Typ 2) ändern – hier ist der Code auskommentiert.
            if length(bus_gens[idx]) != 0
                # Memento.warn(_LOGGER, "active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 2")
                # bus["bus_type"] = 2
            end
        elseif bus["bus_type"] == 2  # PV-Bus
            # Falls an einem PV-Bus keine aktiven Generatoren vorhanden sind, ändere den Typ zu PQ (Typ 1).
            if length(bus_gens[idx]) == 0
                Memento.warn(_LOGGER, "no active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 1")
                bus["bus_type"] = 1
            end
        elseif bus["bus_type"] == 3  # Slack-Bus
            if length(bus_gens[idx]) != 0
                # Wenn an einem Slack-Bus aktive Generatoren vorhanden sind, markiere, dass ein Slack-Bus gefunden wurde.
                slack_found = true
            else
                # Wenn keine aktiven Generatoren am Slack-Bus vorhanden sind, setze den Bustyp auf PQ (Typ 1).
                Memento.warn(_LOGGER, "no active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 1")
                bus["bus_type"] = 1
            end
        elseif bus["bus_type"] == 4  # Inaktiver Bus
            # Bei inaktiven Bussen wird nichts geändert.
        else  # Unbekannter Bustyp
            # Setze standardmäßig den Bustyp auf 1 (PQ) und wechsle zu 2 (PV), falls aktive Generatoren vorhanden sind.
            new_bus_type = 1
            if length(bus_gens[idx]) != 0
                new_bus_type = 2
            end
            Memento.warn(_LOGGER, "bus $(bus["bus_i"]) has an unrecongized bus_type $(bus["bus_type"]), updating to bus_type $(new_bus_type)")
            bus["bus_type"] = new_bus_type
        end
    end

    # Falls kein Slack-Bus (Typ 3) gefunden wurde, wähle den größten Generator aus dem Abschnitt "gen"
    if !slack_found
        gen = _biggest_generator(pm_data["gen"])
        if length(gen) > 0
            # Setze den Bus, an dem der größte Generator angeschlossen ist, als Slack-Bus.
            gen_bus = gen["gen_bus"]
            ref_bus = pm_data["bus"]["$(gen_bus)"]
            ref_bus["bus_type"] = 3
            Memento.warn(_LOGGER, "no reference bus found, setting bus $(gen_bus) as reference based on generator $(gen["index"])")
        else
            # Falls keine aktiven Generatoren gefunden wurden, wird ein Fehler gemeldet.
            Memento.error(_LOGGER, "no generators found in the given network data, correct_bus_types! requires at least one generator at the reference bus")
        end
    end
end
