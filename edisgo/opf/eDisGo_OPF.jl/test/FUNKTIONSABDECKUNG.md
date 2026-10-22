# Funktionsabdeckung `eDisGo_OPF.jl`

Checkliste aller Funktionen und Typen des Pakets. Abgehakt wird erst, wenn es
einen aussagekräftigen `@testset` gibt (siehe
[`TESTKONVENTIONEN.md`](TESTKONVENTIONEN.md), Abschnitt 3) und die vollständige
Suite grün ist.

Der Zusatz hinter dem Namen ist der Dispatch-Typ des ersten Arguments — er
unterscheidet die mehrfach definierten Methoden (z. B. `constraint_model_current`
und `constraint_store_state`).

Legende: `[ ]` offen · `[x]` getestet · `[~]` teilweise getestet (Begründung
dahinter notieren)

## Arbeitsstand (Stand 2026-10-22)

**84 von 84 Einträgen erledigt**, Suite grün: 3446 bestandene Assertions,
35 `@test_broken`, 0 Fehlschläge, Laufzeit rund 4 min mit Gurobi-Lizenz (davon
rund 2 min für die `Main.jl`-Subprozesse; ohne Gurobi, wie in der CI, entfallen
die beiden SOC-Läufe).

| Quelldatei | Status |
| --- | --- |
| `src/core/types.jl` | fertig (6/6) |
| `src/core/base.jl` | fertig (3/3) |
| `src/core/data.jl` | fertig (3/3) |
| `src/core/solution.jl` | fertig (2/2) |
| `src/core/constraint.jl` | fertig (6/6) |
| `src/core/constraint_template.jl` | fertig (5/5) |
| `src/core/objective.jl` | fertig (5/5) |
| `src/io/json.jl` | fertig (1/1) |
| `src/io/common.jl` | fertig (1/1) |
| `src/core/variables.jl` | fertig (40/40) — aufgeteilt in `test/core/variables/test_flow_and_curtailment.jl` (16), `test_flexibilities.jl` (10), `test_slacks.jl` (14) |
| `src/form/bf.jl` | fertig (9/9) |
| `src/prob/opf_bf.jl` | fertig (2/2) |
| `Main.jl` | fertig (1/1) |

Offen ist nur noch Aufräumarbeit: Mehrere Testdateien haben gleichartige lokale
Hilfsfunktionen (leeres Modell ohne Build-Inhalt, Schranken auslesen,
`report`-Prüfung ohne Seiteneffekt auf `pm.sol`, stdout abfangen), die nach
`TestUtils` gehören.

## Im Produktivcode gefundene Mängel

Beim Testschreiben gefunden, **bewusst nicht behoben** (gehört in einen eigenen
Fix-Branch); die Tests halten jeweils das Ist-Verhalten fest und markieren das
Soll mit `@test_broken`:

1. **`_LOGGER` ist im Modul `eDisGo_OPF` nicht definiert** (`isdefined(...) ==
   false`). An 13 Stellen wird `Memento.warn(_LOGGER, ...)` bzw.
   `Memento.error(_LOGGER, ...)` mit dem blanken Namen aufgerufen — in
   `io/json.jl`, `core/constraint.jl`, `core/constraint_template.jl`,
   `core/base.jl:221` und `core/data.jl`. Jeder dieser Pfade wirft
   `UndefVarError(:_LOGGER)` statt zu warnen. Betrifft unter anderem alle
   `time_elapsed`-Fallbacks: die Nebenbedingung wird dann gar nicht erst angelegt.
   Fix wäre ein `const _LOGGER = PowerModels._LOGGER` im Modul.
2. **`_biggest_generator` ebenfalls nicht sichtbar** — `_correct_bus_types!`
   bricht in jedem korrigierenden Zweig ab. Fällt produktiv nicht auf, weil
   `correct_network_data!` den Aufruf auskommentiert hat.
3. **`ref_add_core!` wirft `KeyError`, sobald ein Bus deaktiviert wird**
   (`bus_type == 4`): die eDisGo-eigenen Komponenten (`dsm`, `heatpumps`,
   `electromobility`, `gen_nd`, `gen_slack`) werden nicht mitgefiltert.
4. **`solve_model` ignoriert `multinetwork`/`multiconductor` stillschweigend.**
   PowerModels wirft bei widersprüchlicher Flagge einen Fehler; hier fällt eine
   falsch geformte Datenstruktur nicht mehr auf.
5. **`constraint_model_current` behandelt virtuelle Speicher-Branches
   uneinheitlich**: die NC-Variante überspringt sie (`form/bf.jl:128`), die
   SOC-Variante nicht (`form/bf.jl:101`).
6. **`constraint_HV_requirements` hat keinen `else`-Zweig** — bei unbekanntem
   `hv_req["name"]` bleibt `pflex` undefiniert und `sum(pflex)` wirft
   `UndefVarError(:pflex)`.
7. **Debug-`println`s in `objective_min_line_loading_max_OG`**
   (`core/objective.jl:138,140`) schreiben bei jedem Modellbau mit `opf_version`
   3 auf stdout.
8. **`phss`-Term in `objective_min_losses_slacks`/`_OG`** summiert über die
   Wärmepumpen-Indizes, obwohl `phss` je Wärmespeicher angelegt wird. In beiden
   Fixtures sind die Indizes identisch, bei abweichender Nummerierung gäbe es
   einen `KeyError`.
9. **`correct_network_data!` lässt `check_reference_bus` und `correct_bus_types!`
   auskommentiert** — ein Netz ohne Slack-Bus läuft ohne Warnung durch.
   `check_connectivity` prüft die eDisGo-eigenen Komponenten gar nicht auf
   gültige Bus-Referenzen.
10. **Der SOC→NC-Warmstart in `Main.jl` ist nie lauffähig.** `Main.jl:127` ruft
    `set_ac_bf_start_values!` ohne Modulpräfix auf, `eDisGo_OPF` exportiert die
    Funktion aber nicht. Bei `method="soc"`, `warm_start=True` und straffer
    SOC-Lösung bricht das Skript nach dem Gurobi-Lauf mit `UndefVarError` ab
    (Exit-Code 1, kein Ergebnis-JSON); Python meldet dann „Julia process
    failed!“. Fix wäre `eDisGo_OPF.set_ac_bf_start_values!`.
11. **In `opf_version` 3 ist die Abregelung `pgc` von der Leistungsbilanz
    entkoppelt.** `build_mn_opf_bf_flex` legt `pgc` für Version 3 an
    (`prob/opf_bf.jl:36-38`), `constraint_HV_requirements` rechnet es auf die
    Anforderung `"curt"` an (`core/constraint.jl:145`), aber
    `constraint_power_balance` nimmt `pgc` nur für Version 2/4 in die Knotenbilanz
    (`form/bf.jl:211`). Die HV-Anforderung wird so durch eine Abregelung erfüllt,
    die im Netz nicht wirkt; `from_powermodels` zieht genau dieses `pgc` für
    Version 3/4 trotzdem von der Einspeisung ab (`edisgo/io/powermodels_io.py`),
    die Lastflüsse der OPF passen also nicht zum zurückgeschriebenen Ergebnis.
    Nebenbei: `TestUtils.pm_network(opf_version=3)` entfernt `"curt"` aus
    `flexibilities`, Python setzt es für 2, 3 und 4 (`powermodels_io.py:149`).
12. **`variable_slack_gen_real`/`_imaginary` ignorieren die Startwerte**
    (`core/variables.jl:554-570`). `set_ac_bf_start_values!` schreibt
    `pgs_start`/`qgs_start`, die Variablen werden aber ohne `start` angelegt —
    der Slack-Generator wird beim Warmstart (sobald Mangel 10 behoben ist) nicht
    vorbelegt. Ebenso setzt `set_ac_bf_start_values!` kein `cpe_start`, obwohl
    `variable_cp_energy` es lesen würde.
13. **`variable_branch_power_real_radial`/`_imaginary_radial` werfen bei
    `pt_start`/`qt_start` einen `KeyError`** (`core/variables.jl:39-42` bzw.
    `:73-76`): die Variablen existieren nur auf `arcs_from`, der Startwert wird
    aber auf den To-Arc gesetzt. Latent, weil derzeit niemand Branch-Startwerte
    schreibt.
14. **`variable_line_loading_max` liest das `storage`-Flag aus Netz 1 statt aus
    `nw`** (`core/variables.jl:118`). Folgenlos, solange nur mit `nw=1`
    aufgerufen wird.
15. **`ref_add_core!` filtert `heat_storage` nicht nach `status`** (und
    `gen_slack` nicht nach `gen_status`), anders als `storage`. Ein
    abgeschalteter Wärmespeicher bekommt `phs`/`hse` mit vollen Schranken.
    Produktiv unauffällig, weil Python `status` immer auf `True` setzt.
16. **Bei nur einem Zeitschritt fehlt die Endbedingung des Ladepunkts**
    (Gl. 3.25 für t = τ): sie steht nur in `constraint_cp_state`, das
    `prob/opf_bf.jl:86-92` erst ab dem zweiten Zeitschritt aufruft. Batterie,
    Wärmespeicher und DSM bekommen beide Bedingungen.

Kleinere Befunde ohne `@test_broken`: Bei unbekanntem `method` gibt `Main.jl`
das ungelöste Netz ohne `status`/`solver`/`solve_time` mit Exit-Code 0 aus
(Python meldet dann irreführend „Julia process failed!“). Wegen Mangel 5 ist das
SOC-Modell keine echte Relaxation des NC-Modells, das SOC-Optimum liegt messbar
über dem NC-Optimum. Der Docstring von `variable_branch_power_imaginary_radial`
spricht von `arcs`, der Code nutzt `arcs_from`.

### `Main.jl` → `test/test_main.jl`

- [x] `optimize_edisgo` (Bug: der SOC-Pfad mit `warm_start` bricht bei straffer
  Relaxation mit `UndefVarError(:set_ac_bf_start_values!)` ab, siehe Mangel 10 —
  als `@test_broken` dokumentiert; die beiden SOC-Läufe brauchen Gurobi und
  werden ohne Lizenz übersprungen, in der CI läuft nur der NC-Lauf)

### `src/core/types.jl` → `test/core/test_types.jl`

- [x] `AbstractBFModelEdisgo` — abstract type
- [x] `BFPowerModelEdisgo` — mutable struct
- [x] `AbstractSOCBFModelEdisgo` — abstract type
- [x] `SOCBFPowerModelEdisgo` — mutable struct
- [x] `AbstractNCBFModelEdisgo` — abstract type
- [x] `NCBFPowerModelEdisgo` — mutable struct

### `src/core/base.jl` → `test/core/test_base.jl`

- [x] `solve_model` — Dict
- [x] `instantiate_model` — Dict
- [x] `ref_add_core!` — Dict

### `src/core/data.jl` → `test/core/test_data.jl`

- [x] `set_ac_bf_start_values!` — Dict
- [~] `correct_bus_types!` — Dict (Weiterreichen an den Worker für Einzel- und
  Multinetzwerk getestet; die mutierenden Pfade nur als `@test_broken`, weil der
  Worker `_correct_bus_types!` defekt ist — siehe dort)
- [~] `_correct_bus_types!` — Dict (alle Zweige abgedeckt, aber jeder
  korrigierende Zweig bricht mit `UndefVarError` ab: `_LOGGER` und
  `_biggest_generator` sind interne PowerModels-Namen und in `eDisGo_OPF`
  nicht sichtbar. Soll-Verhalten als `@test_broken` festgehalten)

### `src/core/solution.jl` → `test/core/test_solution.jl`

- [x] `sol_component_value_radial` — AbstractPowerModel
- [x] `check_SOC_equality`

### `src/core/variables.jl` → `test/core/test_variables.jl`

- [x] `variable_branch_power_radial` — AbstractPowerModel
- [x] `variable_branch_power_real_radial` — AbstractPowerModel (Mangel: `pt_start` wirft `KeyError`, siehe
  Mangel 13 — als `@test_broken` dokumentiert)
- [x] `variable_branch_power_imaginary_radial` — AbstractPowerModel (Mangel: `qt_start` wirft `KeyError`,
  siehe Mangel 13 — als `@test_broken` dokumentiert)
- [x] `variable_bus_voltage_magnitude_sqr` — AbstractPowerModel
- [x] `variable_max_line_loading` — AbstractPowerModel (Anmerkung: nimmt als einziger Wrapper kein
  `bounded` an, `MethodError` — per `@test_throws` festgehalten)
- [x] `variable_line_loading_max` — AbstractPowerModel (Mangel: `storage`-Flag wird aus Netz 1 statt
  aus `nw` gelesen, siehe Mangel 14 — als `@test_broken` dokumentiert;
  Anmerkung: der Default-Startwert 0 liegt unter der unteren Schranke 1)
- [x] `variable_gen_power_curt` — AbstractPowerModel
- [x] `variable_gen_power_curt_real` — AbstractPowerModel
- [x] `variable_gen_power_curt_imaginary` — AbstractPowerModel
- [x] `variable_battery_storage` — AbstractPowerModel
- [x] `variable_battery_storage_power_real` — AbstractPowerModel
- [x] `variable_battery_storage_power_imaginary` — AbstractPowerModel
- [x] `variable_dsm_storage_power` — AbstractPowerModel
- [x] `variable_dsm_storage_power_real` — AbstractPowerModel
- [x] `variable_dsm_storage_power_imaginary` — AbstractPowerModel
- [x] `variable_dsm_storage_energy` — AbstractPowerModel
- [x] `variable_heat_storage` — AbstractPowerModel
- [x] `variable_heat_storage_power` — AbstractPowerModel (Mangel: abgeschaltete Wärmespeicher bekommen
  trotzdem eine Variable, siehe Mangel 15 — als `@test_broken` dokumentiert)
- [x] `variable_heat_storage_energy` — AbstractPowerModel (Mangel: derselbe ignorierte `status` wie bei
  `phs` — als `@test_broken` dokumentiert)
- [x] `variable_heat_pump_power` — AbstractPowerModel
- [x] `variable_heat_pump_power_real` — AbstractPowerModel
- [x] `variable_heat_pump_power_imaginary` — AbstractPowerModel (Anmerkung: ohne `start`, `qhp_start`
  wird ignoriert; im OPF nie aufgerufen, weil der Aufruf in
  `variable_heat_pump_power` auskommentiert ist)
- [x] `variable_cp_power` — AbstractPowerModel
- [x] `variable_cp_power_real` — AbstractPowerModel
- [x] `variable_cp_power_imaginary` — AbstractPowerModel (Anmerkung: ohne `start`, `qcp_start` wird
  ignoriert; im OPF nie aufgerufen, Aufruf in `variable_cp_power` auskommentiert)
- [x] `variable_cp_energy` — AbstractPowerModel
- [x] `variable_slack_grid_restrictions` — AbstractBFModelEdisgo
- [x] `variable_slack_heat_pump_storage` — AbstractBFModelEdisgo
- [x] `variable_hs_slack` — AbstractBFModelEdisgo
- [x] `variable_hp2_slack` — AbstractBFModelEdisgo
- [x] `variable_hp_slack` — AbstractBFModelEdisgo
- [x] `variable_load_slack` — AbstractBFModelEdisgo
- [x] `variable_gen_slack` — AbstractBFModelEdisgo
- [x] `variable_ev_slack` — AbstractBFModelEdisgo
- [x] `variable_slack_gen` — AbstractBFModelEdisgo
- [x] `variable_slack_gen_real` — AbstractBFModelEdisgo (Mangel: `pgs_start` wird ignoriert, siehe
  Mangel 12 — als `@test_broken` dokumentiert)
- [x] `variable_slack_gen_imaginary` — AbstractBFModelEdisgo (Mangel: `qgs_start` wird ignoriert, siehe
  Mangel 12 — als `@test_broken` dokumentiert)
- [x] `variable_slack_HV_requirements` — AbstractPowerModel
- [x] `variable_slack_HV_requirements_real` — AbstractPowerModel
- [x] `variable_slack_HV_requirements_imaginary` — AbstractPowerModel

### `src/core/constraint.jl` → `test/core/test_constraint.jl`

- [x] `constraint_store_state_initial` — AbstractBFModelEdisgo
- [x] `constraint_store_state` — AbstractBFModelEdisgo
- [x] `constraint_cp_state_initial` — AbstractBFModelEdisgo (Bug: der
  `time_elapsed`-Fallback ruft `Memento.warn(_LOGGER, …)` auf; `_LOGGER` ist in
  `eDisGo_OPF` nicht definiert, deshalb `UndefVarError` statt Warnung und die
  Nebenbedingung entsteht gar nicht — als `@test_broken` dokumentiert)
- [x] `constraint_cp_state` — AbstractBFModelEdisgo (Bug: derselbe undefinierte
  `_LOGGER` im `time_elapsed`-Fallback wie oben — als `@test_broken` dokumentiert)
- [x] `constraint_hp_operation` — AbstractBFModelEdisgo
- [x] `constraint_HV_requirements` — AbstractBFModelEdisgo (Bug: für einen
  unbekannten `hv_req["name"]` gibt es keinen `else`-Zweig; `pflex` bleibt
  undefiniert und die Funktion wirft `UndefVarError(:pflex)` statt einer
  aussagekräftigen Fehlermeldung — als `@test_broken` dokumentiert)

### `src/core/constraint_template.jl` → `test/core/test_constraint_template.jl`

- [x] `constraint_power_balance_bf` — AbstractBFModelEdisgo (Anmerkung: der
  Leistungsfaktor-Term `bus_storage_pf` wird berechnet und übergeben, von
  `constraint_power_balance` aber nie verwendet — die Blindleistung der
  Batterie kommt über `storage_pf` des virtuellen Branches; ein Speicher ohne
  `pf` lässt das Template dennoch scheitern)
- [x] `constraint_voltage_magnitude_difference_radial` — AbstractBFModelEdisgo
- [x] `constraint_store_state` — AbstractBFModelEdisgo, `(pm, i; nw, kind)`
  (Mangel: fehlt `time_elapsed` im `ref`, soll laut Code eine Warnung folgen
  und auf 1.0 zurueckgefallen werden; die Warnung geht an `_LOGGER`, das im
  Modul `eDisGo_OPF` nicht definiert ist — es fliegt ein `UndefVarError`.
  Ist-Verhalten per `@test_throws`, Soll-Verhalten per `@test_broken`
  festgehalten)
- [x] `constraint_store_state` — AbstractBFModelEdisgo, `(pm, i, nw_1, nw_2, kind)`
  (Mangel: derselbe `_LOGGER`-Fehler auf zwei Pfaden — fehlendes
  `time_elapsed` in `nw_2` und Komponente in `nw_1` nicht vorhanden, wo auf
  `constraint_store_state_initial` zurueckgefallen werden soll)
- [x] `constraint_model_current` — AbstractPowerModel

### `src/core/objective.jl` → `test/core/test_objective.jl`

- [x] `objective_min_losses` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks` — AbstractBFModelEdisgo (Anmerkung: der
  `phss`-Term wird über die `heatpumps`-Indizes summiert, obwohl `phss` je
  `heat_storage` angelegt wird — bei abweichender Nummerierung ein `KeyError`;
  in den Fixtures stimmen die Indizes überein, daher nur als Kommentar notiert)
- [x] `objective_min_line_loading_max` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks_OG` — AbstractBFModelEdisgo
- [x] `objective_min_line_loading_max_OG` — AbstractBFModelEdisgo (Mangel: zwei
  übrig gebliebene `println`-Aufrufe schreiben bei jedem Aufruf `factor_ll` und
  `factor_hv_slacks` auf stdout; per stdout-Abfang im Test festgehalten)

### `src/form/bf.jl` → `test/form/test_bf.jl`

- [x] `variable_branch_current` — AbstractBFModel
- [x] `variable_bus_voltage` — AbstractBFModel
- [x] `variable_buspair_current_magnitude_sqr` — AbstractBFModel
- [x] `constraint_voltage_magnitude_difference` — AbstractBFModelEdisgo
- [x] `constraint_model_current` — AbstractSOCBFModelEdisgo (Mangel 5: die
  SOC-Variante schreibt die Nebenbedingung auch für den virtuellen
  Speicher-Branch — Ist-Verhalten festgehalten, gleiche Branch-Menge wie NC als
  `@test_broken`)
- [x] `constraint_model_current` — AbstractNCBFModelEdisgo (Gegenstück zu
  Mangel 5: der virtuelle Speicher-Branch wird übersprungen)
- [x] `constraint_max_line_loading` — AbstractSOCBFModelEdisgo
- [x] `constraint_max_line_loading` — AbstractNCBFModelEdisgo
- [x] `constraint_power_balance` — AbstractBFModelEdisgo (Bug: für
  `opf_version` 3 fehlt die Abregelung `pgc` in der Bilanz, siehe Mangel 11 —
  als `@test_broken` dokumentiert; Anmerkung: `bus_storage_pf` wird übergeben,
  aber nie verwendet — per Test festgehalten)

### `src/io/json.jl` → `test/io/test_json.jl`

- [x] `parse_json` — Union{IO — Bug: der `conductors`-Zweig referenziert das in
  `eDisGo_OPF` nicht definierte `_LOGGER` und wirft statt zu warnen einen
  `UndefVarError` (als `@test_broken` dokumentiert)

### `src/io/common.jl` → `test/io/test_common.jl`

- [x] `correct_network_data!` — Dict

### `src/prob/opf_bf.jl` → `test/prob/test_opf_bf.jl`

- [x] `solve_mn_opf_bf_flex`
- [x] `build_mn_opf_bf_flex` — AbstractBFModelEdisgo (Bugs: in Version 3 wirkt
  `pgc` nicht auf die Leistungsbilanz, siehe Mangel 11; bei nur einem Zeitschritt
  fehlt die Endbedingung des Ladepunkts, siehe Mangel 16 — beides als
  `@test_broken` dokumentiert)

<!-- Gesamtzahl der Einträge: 84 -->
