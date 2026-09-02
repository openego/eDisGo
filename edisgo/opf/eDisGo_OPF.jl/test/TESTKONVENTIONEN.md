# Testkonventionen für `eDisGo_OPF.jl`

Diese Datei ist die verbindliche Vorgabe für alle, die Tests für das Julia-Paket
`eDisGo_OPF` schreiben — Menschen wie KI-Agents. Wer hier etwas ändert, ändert die
Vorgabe für alle folgenden Beiträge.

Ziel ist eine vollständige Testabdeckung: **jede** Funktion und jeder Typ des
Pakets bekommt Tests. Der Fortschritt wird in
[`FUNKTIONSABDECKUNG.md`](FUNKTIONSABDECKUNG.md) abgehakt.

## 1. Verzeichnisstruktur

Die Teststruktur spiegelt `src/` eins zu eins:

| Quelldatei                       | Testdatei                             |
| -------------------------------- | ------------------------------------- |
| `Main.jl`                        | `test/test_main.jl`                   |
| `src/core/base.jl`               | `test/core/test_base.jl`              |
| `src/core/constraint.jl`         | `test/core/test_constraint.jl`        |
| `src/core/constraint_template.jl`| `test/core/test_constraint_template.jl` |
| `src/core/data.jl`               | `test/core/test_data.jl`              |
| `src/core/objective.jl`          | `test/core/test_objective.jl`         |
| `src/core/solution.jl`           | `test/core/test_solution.jl`          |
| `src/core/types.jl`              | `test/core/test_types.jl`             |
| `src/core/variables.jl`          | `test/core/test_variables.jl`         |
| `src/form/bf.jl`                 | `test/form/test_bf.jl`                |
| `src/io/common.jl`               | `test/io/test_common.jl`              |
| `src/io/json.jl`                 | `test/io/test_json.jl`                |
| `src/prob/opf_bf.jl`             | `test/prob/test_opf_bf.jl`            |

Dazu kommen:

```
test/
├── runtests.jl            # Einstiegspunkt, Liste TEST_FILES, Testauswahl
├── testsetup.jl           # gemeinsame Präambel jeder Testdatei
├── testutils/TestUtils.jl # Fixtures und Hilfsfunktionen
├── data/                  # eingecheckte JSON-Fixtures + Generatorskript
├── test_smoke.jl          # Paket/Abhängigkeiten/Fixtures laden
├── TESTKONVENTIONEN.md    # diese Datei
└── FUNKTIONSABDECKUNG.md  # Checkliste aller Funktionen
```

**Neue Quelldatei ⇒ neue Testdatei**: Testdatei anlegen, in `TEST_FILES` in
`runtests.jl` eintragen und die Funktionen in `FUNKTIONSABDECKUNG.md` ergänzen.

## 2. Aufbau einer Testdatei

Jede Testdatei ist eigenständig lauffähig **und** über `runtests.jl` einbindbar.
Sie beginnt deshalb immer mit der gemeinsamen Präambel:

```julia
# Tests for `src/core/variables.jl`.

include(joinpath(@__DIR__, "..", "testsetup.jl"))   # ohne ".." für Dateien in test/

@testset "variable_branch_power_real_radial" begin
    ...
end
```

Regeln:

- **Ein `@testset` pro Funktion**, benannt exakt wie die Funktion. Gibt es mehrere
  Methoden (z. B. `constraint_model_current` für SOC und NC), werden diese in
  verschachtelten `@testset`s mit dem Dispatch-Typ im Namen getrennt:
  `@testset "AbstractSOCBFModelEdisgo" begin ... end`.
- Die `@testset`s stehen in derselben Reihenfolge wie die Funktionen in der
  Quelldatei.
- Kommentare und Docstrings **auf Englisch** (wie der übrige Code). Die
  Markdown-Dokumentation (diese Datei, `FUNKTIONSABDECKUNG.md`) ist **auf Deutsch**.

## 3. Was ein guter Test hier prüft

Das Paket baut Optimierungsmodelle; die meisten Funktionen geben nichts zurück,
sondern hängen Variablen oder Nebenbedingungen an ein `JuMP.Model`. Getestet wird
deshalb der beobachtbare Effekt:

1. **Variablenfunktionen** (`variable_*`)
   - Die Variable liegt unter dem erwarteten Schlüssel in `PowerModels.var(pm, nw)`.
   - Indexmenge stimmt (welche Komponenten/Arcs, welche ausgeschlossen sind —
     z. B. werden virtuelle Speicher-Branches bewusst übersprungen).
   - Schranken: `JuMP.has_lower_bound` / `JuMP.lower_bound` usw., und zwar sowohl
     für `bounded=true` als auch `bounded=false`.
   - Startwerte aus `*_start`-Feldern werden übernommen.
   - `report=true` schreibt den Eintrag in die Lösungsstruktur, `report=false` nicht.
2. **Constraint-Funktionen** (`constraint_*`)
   - Die Zahl der Nebenbedingungen wächst um den erwarteten Betrag
     (`TestUtils.num_constraints_total`).
   - Die Nebenbedingung enthält die erwarteten Terme/Koeffizienten
     (`TestUtils.constraint_strings` oder direkter Zugriff über
     `PowerModels.con(pm, nw, :name)`).
   - Sonderfälle: virtuelle Speicher-Branches, fehlendes `time_elapsed`,
     Komponente im Vorzeitschritt nicht vorhanden.
3. **Zielfunktionen** (`objective_*`)
   - `JuMP.objective_sense` und die Terme des `JuMP.objective_function`.
4. **Datenfunktionen** (`correct_*`, `set_*`, `parse_*`, `check_*`)
   - Rückgabewert bzw. mutiertes Dict exakt vergleichen.
   - Fehlerfälle mit `@test_throws` prüfen.
5. **Ende-zu-Ende** (`solve_*`, `build_*`, `Main.jl`)
   - Kleines Netz mit Ipopt lösen und Status sowie Größenordnung der Lösung
     prüfen. Keine exakten Solver-Zahlen auf viele Nachkommastellen festnageln —
     `atol`/`rtol` verwenden (Richtwert `atol=1e-4` für p.u.-Größen).

Grundsätze:

- **Kein Test ohne Aussage.** `@test true` nach einem `using` ist kein Test —
  Ausnahme sind die bewussten Smoke-Tests in `test_smoke.jl`.
- **Positiv- und Negativfall.** Wenn eine Funktion Komponenten filtert, wird
  geprüft, dass die erwarteten drin *und* die anderen draußen sind.
- **Keine Abhängigkeit von der Reihenfolge.** Julia-`Dict`s sind unsortiert;
  immer über `sort(...)` oder Mengenvergleich (`Set`) prüfen.
- **Tests dürfen sich nicht gegenseitig beeinflussen.** `TestUtils.pm_network()`
  und `TestUtils.small_network()` liefern bei jedem Aufruf frische Dicts.

## 4. Fixtures

Zwei Netze stehen zur Verfügung (siehe Docstrings in `testutils/TestUtils.jl`):

- `TestUtils.small_network(; opf_version, num_steps, overlying_grid)` — von Hand
  gebautes 4-Bus-Netz mit genau einer Komponente jedes Flexibilitätstyps. **Erste
  Wahl für Unit-Tests**: schnell, überschaubar, gezielt manipulierbar.
- `TestUtils.pm_network(; overlying_grid, opf_version, validate)` — das echte
  eDisGo-Testnetz (143 Busse, 4 Zeitschritte) aus
  `tests/data/ding0_test_network_1`. **Erste Wahl für Integrationstests** und
  überall dort, wo realistische Daten nötig sind.

Weiter:

- `TestUtils.multinetwork(data)` — Einzelnetz → Multinetwork (wie `Main.jl`).
- `TestUtils.build_model(...)` / `TestUtils.build_small_model(...)` — Modell
  instanziieren, ohne zu lösen.
- `TestUtils.gurobi_available()` — Gurobi ist kommerziell und in der CI nicht
  vorhanden. Tests, die Gurobi brauchen, **müssen** damit abgesichert werden:

  ```julia
  if TestUtils.gurobi_available()
      ...
  else
      @test_skip false
  end
  ```

  Für alles andere Ipopt verwenden — der ist frei verfügbar und deckt die
  nicht-konvexe Formulierung ab.

Die JSON-Fixtures in `test/data/` werden von `test/data/generate_fixtures.py`
erzeugt. Ändert sich das Eingabeschema auf Python-Seite
(`edisgo/io/powermodels_io.to_powermodels`), müssen die Fixtures neu erzeugt und
zusammen mit der Codeänderung committet werden.

## 5. Ausführen

Aus dem Verzeichnis `edisgo/opf/eDisGo_OPF.jl`:

```bash
# vollständige Suite
julia --project=. -e 'using Pkg; Pkg.test()'

# einzelne Datei (schnell, ohne Pkg.test-Overhead)
julia --project=. test/core/test_variables.jl

# Teilmenge über runtests.jl
julia --project=. test/runtests.jl variables objective
EDISGO_OPF_TESTS=core julia --project=. -e 'using Pkg; Pkg.test()'

# mit Coverage
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
```

Die Testabhängigkeiten stehen bewusst in `[extras]`/`[targets]` der
`Project.toml` und **nicht** in einer eigenen `test/Project.toml`: so läuft die
Suite gegen die eingecheckte `Manifest.toml`, also gegen exakt die Paketversionen,
die auch produktiv verwendet werden.

## 6. Arbeitsweise und Commits

Vorgehen pro Funktion:

1. Funktion in `FUNKTIONSABDECKUNG.md` auswählen (von oben nach unten).
2. Quellcode der Funktion lesen — inklusive der Stellen, die sie aufrufen.
3. `@testset` in der passenden Testdatei ergänzen (Reihenfolge wie in der Quelle).
4. Testdatei einzeln laufen lassen, dann die vollständige Suite.
5. Haken in `FUNKTIONSABDECKUNG.md` setzen.
6. **Ein Commit pro abgehakter Funktion.** Commit-Nachricht:

   ```
   test(opf-julia): add tests for <funktionsname>
   ```

   Bei mehreren Methoden einer Funktion, die zusammen getestet werden, reicht ein
   Commit für alle Methoden derselben Funktion.

   **Keine Attribution-Trailer.** Commits tragen ausschließlich den Autor aus der
   git-Konfiguration — keine `Co-Authored-By:`- oder `Claude-Session:`-Zeile, kein
   Hinweis auf ein KI-Werkzeug, weder im Betreff noch im Body.

Wird beim Testschreiben ein **Fehler im Produktivcode** gefunden: den Test so
schreiben, dass er das tatsächliche Verhalten dokumentiert, ihn mit
`@test_broken` markieren und den Fund im Commit-Text sowie in
`FUNKTIONSABDECKUNG.md` notieren. Produktivcode wird in diesem Arbeitsstrang
**nicht** geändert — das gehört in einen eigenen Fix-Branch.
