# Hinweise für Coding-Agents

## Julia-OPF-Paket (`edisgo/opf/eDisGo_OPF.jl`)

Für **alle** Arbeiten an der Testsuite des Julia-Pakets gelten verbindlich:

- [`edisgo/opf/eDisGo_OPF.jl/test/TESTKONVENTIONEN.md`](edisgo/opf/eDisGo_OPF.jl/test/TESTKONVENTIONEN.md)
  — Struktur, Aufbau einer Testdatei, Fixtures, Ausführung, Commit-Konvention.
- [`edisgo/opf/eDisGo_OPF.jl/test/FUNKTIONSABDECKUNG.md`](edisgo/opf/eDisGo_OPF.jl/test/FUNKTIONSABDECKUNG.md)
  — Checkliste aller Funktionen. Ziel ist die vollständige Abdeckung; pro
  abgehakter Funktion ein eigener Commit.

Kurzfassung: Testdateien spiegeln `src/`, ein `@testset` pro Funktion, Kommentare
im Code auf Englisch, Markdown-Dokumentation auf Deutsch. Tests laufen mit
`julia --project=. -e 'using Pkg; Pkg.test()'` aus `edisgo/opf/eDisGo_OPF.jl`.

## Commits

Commits und Pull-Request-Beschreibungen tragen ausschließlich den Autor aus der
git-Konfiguration. **Keine** `Co-Authored-By:`- oder `Claude-Session:`-Trailer,
kein "Generated with"-Hinweis, keine Erwähnung eines KI-Werkzeugs — weder im
Betreff noch im Body.

## Sprache

- Code, Docstrings und Code-Kommentare: **Englisch**.
- Markdown-Dokumentation in diesem Repository: **Deutsch**, sofern die Datei
  nicht bereits auf Englisch gepflegt wird (`README.md`, `doc/`, `AUTHORS.md`,
  `doc/whatsnew/`).
