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
- [ ] `constraint_HV_requirements` — AbstractBFModelEdisgo

### `src/core/constraint_template.jl` → `test/core/test_constraint_template.jl`

- [ ] `constraint_power_balance_bf` — AbstractBFModelEdisgo
- [ ] `constraint_voltage_magnitude_difference_radial` — AbstractBFModelEdisgo
- [ ] `constraint_store_state` — AbstractBFModelEdisgo
- [ ] `constraint_store_state` — AbstractBFModelEdisgo
- [ ] `constraint_model_current` — AbstractPowerModel

### `src/core/objective.jl` → `test/core/test_objective.jl`

- [x] `objective_min_losses` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks` — AbstractBFModelEdisgo (Anmerkung: der
  `phss`-Term wird über die `heatpumps`-Indizes summiert, obwohl `phss` je
  `heat_storage` angelegt wird — bei abweichender Nummerierung ein `KeyError`;
  in den Fixtures stimmen die Indizes überein, daher nur als Kommentar notiert)
- [x] `objective_min_line_loading_max` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks_OG` — AbstractBFModelEdisgo
- [ ] `objective_min_line_loading_max_OG` — AbstractBFModelEdisgo

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
