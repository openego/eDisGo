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

## Arbeitsstand (Stand 2026-09-02)

**32 von 84 Einträgen erledigt**, Suite grün: 1602 bestandene Assertions,
21 `@test_broken`, 0 Fehlschläge, Laufzeit rund 1:20 min.

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
| `src/core/variables.jl` | **offen (0/40)** — aufgeteilt in `test/core/variables/test_flow_and_curtailment.jl` (16), `test_flexibilities.jl` (10), `test_slacks.jl` (14); alle drei Dateien liegen als Stub bereit |
| `src/form/bf.jl` | **offen (0/9)** |
| `src/prob/opf_bf.jl` | **offen (0/2)** |
| `Main.jl` | **offen (0/1)** |

Nächste Schritte in dieser Reihenfolge: `src/form/bf.jl`, dann die drei Blöcke von
`src/core/variables.jl` (voneinander unabhängig, parallelisierbar), danach
`src/prob/opf_bf.jl` und zuletzt `Main.jl` (Ende-zu-Ende, braucht einen
Julia-Subprozess mit JSON über stdin/stdout).

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

### `Main.jl` → `test/test_main.jl`

- [ ] `optimize_edisgo`

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

- [ ] `variable_branch_power_radial` — AbstractPowerModel
- [ ] `variable_branch_power_real_radial` — AbstractPowerModel
- [ ] `variable_branch_power_imaginary_radial` — AbstractPowerModel
- [ ] `variable_bus_voltage_magnitude_sqr` — AbstractPowerModel
- [ ] `variable_max_line_loading` — AbstractPowerModel
- [ ] `variable_line_loading_max` — AbstractPowerModel
- [ ] `variable_gen_power_curt` — AbstractPowerModel
- [ ] `variable_gen_power_curt_real` — AbstractPowerModel
- [ ] `variable_gen_power_curt_imaginary` — AbstractPowerModel
- [ ] `variable_battery_storage` — AbstractPowerModel
- [ ] `variable_battery_storage_power_real` — AbstractPowerModel
- [ ] `variable_battery_storage_power_imaginary` — AbstractPowerModel
- [ ] `variable_dsm_storage_power` — AbstractPowerModel
- [ ] `variable_dsm_storage_power_real` — AbstractPowerModel
- [ ] `variable_dsm_storage_power_imaginary` — AbstractPowerModel
- [ ] `variable_dsm_storage_energy` — AbstractPowerModel
- [ ] `variable_heat_storage` — AbstractPowerModel
- [ ] `variable_heat_storage_power` — AbstractPowerModel
- [ ] `variable_heat_storage_energy` — AbstractPowerModel
- [ ] `variable_heat_pump_power` — AbstractPowerModel
- [ ] `variable_heat_pump_power_real` — AbstractPowerModel
- [ ] `variable_heat_pump_power_imaginary` — AbstractPowerModel
- [ ] `variable_cp_power` — AbstractPowerModel
- [ ] `variable_cp_power_real` — AbstractPowerModel
- [ ] `variable_cp_power_imaginary` — AbstractPowerModel
- [ ] `variable_cp_energy` — AbstractPowerModel
- [ ] `variable_slack_grid_restrictions` — AbstractBFModelEdisgo
- [ ] `variable_slack_heat_pump_storage` — AbstractBFModelEdisgo
- [ ] `variable_hs_slack` — AbstractBFModelEdisgo
- [ ] `variable_hp2_slack` — AbstractBFModelEdisgo
- [ ] `variable_hp_slack` — AbstractBFModelEdisgo
- [ ] `variable_load_slack` — AbstractBFModelEdisgo
- [ ] `variable_gen_slack` — AbstractBFModelEdisgo
- [ ] `variable_ev_slack` — AbstractBFModelEdisgo
- [ ] `variable_slack_gen` — AbstractBFModelEdisgo
- [ ] `variable_slack_gen_real` — AbstractBFModelEdisgo
- [ ] `variable_slack_gen_imaginary` — AbstractBFModelEdisgo
- [ ] `variable_slack_HV_requirements` — AbstractPowerModel
- [ ] `variable_slack_HV_requirements_real` — AbstractPowerModel
- [ ] `variable_slack_HV_requirements_imaginary` — AbstractPowerModel

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

- [ ] `variable_branch_current` — AbstractBFModel
- [ ] `variable_bus_voltage` — AbstractBFModel
- [ ] `variable_buspair_current_magnitude_sqr` — AbstractBFModel
- [ ] `constraint_voltage_magnitude_difference` — AbstractBFModelEdisgo
- [ ] `constraint_model_current` — AbstractSOCBFModelEdisgo
- [ ] `constraint_model_current` — AbstractNCBFModelEdisgo
- [ ] `constraint_max_line_loading` — AbstractSOCBFModelEdisgo
- [ ] `constraint_max_line_loading` — AbstractNCBFModelEdisgo
- [ ] `constraint_power_balance` — AbstractBFModelEdisgo

### `src/io/json.jl` → `test/io/test_json.jl`

- [x] `parse_json` — Union{IO — Bug: der `conductors`-Zweig referenziert das in
  `eDisGo_OPF` nicht definierte `_LOGGER` und wirft statt zu warnen einen
  `UndefVarError` (als `@test_broken` dokumentiert)

### `src/io/common.jl` → `test/io/test_common.jl`

- [x] `correct_network_data!` — Dict

### `src/prob/opf_bf.jl` → `test/prob/test_opf_bf.jl`

- [ ] `solve_mn_opf_bf_flex`
- [ ] `build_mn_opf_bf_flex` — AbstractBFModelEdisgo

<!-- Gesamtzahl der Einträge: 84 -->
