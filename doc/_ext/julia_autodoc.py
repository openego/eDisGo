# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sphinx extension: render the eDisGo_OPF Julia API natively in the Sphinx docs.

Sphinx cannot autodoc Julia, and Documenter.jl only produces a *separate*
standalone HTML site. To get the Julia OPF API as a normal page inside the
Python documentation (same theme, same left-hand TOC, same search), this
extension parses the Julia *source* for docstrings — no Julia runtime, no Gurobi,
no extra build step — and writes a MyST-Markdown page that Sphinx renders like
any other page.

A docstring is the string literal directly preceding a definition (``function``,
``struct``, ``abstract type``, ``macro``). Empty ``""`` placeholders and strings
separated from the definition by a blank line are skipped, mirroring how Julia
itself attaches docstrings. The page is regenerated on every build, so it grows
automatically as docstrings are added to the Julia source.
"""

from __future__ import annotations

import os

# Source files in display order, with a friendly section title for each.
_FILE_ORDER = [
    ("core/types.jl", "Problem formulations"),
    ("prob/opf_bf.jl", "Problem definition & build"),
    ("core/base.jl", "Model setup"),
    ("core/variables.jl", "Variables"),
    ("core/constraint.jl", "Constraints — flexibilities & HV requirements"),
    ("core/constraint_template.jl", "Constraint templates"),
    ("form/bf.jl", "Branch-flow formulation"),
    ("core/objective.jl", "Objective functions"),
    ("core/data.jl", "Data preparation"),
    ("core/solution.jl", "Solution processing"),
    ("io/common.jl", "I/O — network validation"),
    ("io/json.jl", "I/O — JSON parsing"),
]

_INTRO = """\
# Julia OPF API (`eDisGo_OPF`)

```{note}
This page is generated automatically from the docstrings in the `eDisGo_OPF`
Julia source (`edisgo/opf/eDisGo_OPF.jl/src/`). Only documented symbols are
listed, so the reference fills in on its own as docstrings are added at the
source.
```

The optimal power flow (OPF) is implemented in Julia, in the `eDisGo_OPF`
package, which extends
[PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl) with a branch-flow
formulation for distribution grids and eDisGo's flexibilities (storage,
electromobility, heat pumps, demand-side management, curtailment).

From Python the OPF is run via {meth}`~edisgo.edisgo.EDisGo.pm_optimize`, which
serialises the grid and flexibility data to JSON, hands it to `eDisGo_OPF` and
reads the results back. For the modelling background — branch-flow physics, the
SOC vs. non-convex relaxations and the `opf_version` variants — see
{ref}`flexibility-opf`.

The three exported problem formulations are `BFPowerModelEdisgo` (base
branch-flow), `SOCBFPowerModelEdisgo` (second-order-cone relaxation, convex) and
`NCBFPowerModelEdisgo` (non-convex, exact).
"""


def _read_definition(lines, k):
    """If ``lines[k]`` starts a definition, return ``(name, signature)``.

    For functions the signature spans lines until the parentheses balance, to
    support multi-line argument lists. Returns ``(None, None)`` otherwise.
    """
    if k >= len(lines):
        return None, None
    stripped = lines[k].lstrip()
    name = None
    is_func = False
    if stripped.startswith("function "):
        is_func = True
        name = stripped[len("function ") :].split("(")[0].strip()
    elif stripped.startswith("mutable struct "):
        name = stripped[len("mutable struct ") :].split()[0].split("<")[0].strip()
    elif stripped.startswith("struct "):
        name = stripped[len("struct ") :].split()[0].split("<")[0].strip()
    elif stripped.startswith("abstract type "):
        name = stripped[len("abstract type ") :].split()[0].split("<")[0].strip()
    elif stripped.startswith("macro "):
        name = stripped[len("macro ") :].split("(")[0].strip()
    if not name:
        return None, None

    sig_lines = [lines[k].rstrip()]
    if is_func:
        depth = lines[k].count("(") - lines[k].count(")")
        kk = k + 1
        while depth > 0 and kk < len(lines):
            sig_lines.append(lines[kk].rstrip())
            depth += lines[kk].count("(") - lines[kk].count(")")
            kk += 1
    sig = "\n".join(sig_lines).strip()
    # Trim struct/type bodies (e.g. "... @pm_fields end") down to the declaration.
    if not is_func:
        for marker in (" @pm_fields", " end"):
            idx = sig.find(marker)
            if idx != -1:
                sig = sig[:idx].rstrip()
    return name, sig


def _parse_file(path):
    """Return a list of ``(name, signature, docstring)`` for documented defs."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    entries = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        doc = None
        doc_end = i

        if stripped.startswith('"""'):
            rest = stripped[3:]
            if rest.endswith('"""') and len(rest) > 3:
                doc = rest[:-3]
            else:
                body = [rest] if rest.strip() else []
                j = i + 1
                while j < n and '"""' not in lines[j]:
                    body.append(lines[j].rstrip("\n"))
                    j += 1
                if j < n:
                    before = lines[j][: lines[j].index('"""')]
                    if before.strip():
                        body.append(before.rstrip("\n"))
                doc = "\n".join(body)
                doc_end = j
        elif (
            stripped.startswith('"')
            and stripped.endswith('"')
            and len(stripped) >= 2
            and stripped.count('"') == 2
        ):
            doc = stripped[1:-1]

        if doc is not None:
            name, sig = _read_definition(lines, doc_end + 1)
            if name and doc.strip():
                entries.append((name, sig, doc.strip("\n")))
            i = doc_end + 1
            continue
        i += 1
    return entries


def _build_page(src_root):
    parts = [_INTRO]
    for relpath, title in _FILE_ORDER:
        path = os.path.join(src_root, relpath)
        if not os.path.isfile(path):
            continue
        entries = _parse_file(path)
        if not entries:
            continue
        parts.append(f"\n## {title}\n")
        parts.append(f"*Source: `edisgo/opf/eDisGo_OPF.jl/src/{relpath}`*\n")
        for name, sig, doc in entries:
            parts.append(f"\n### `{name}`\n")
            if sig:
                parts.append("```julia\n" + sig + "\n```\n")
            parts.append("\n" + doc + "\n")
    return "\n".join(parts) + "\n"


def _generate(app):
    src_root = os.path.join(app.srcdir, "..", "edisgo", "opf", "eDisGo_OPF.jl", "src")
    out_path = os.path.join(app.srcdir, "reference", "julia_api.md")
    if not os.path.isdir(src_root):
        # Julia package not present (e.g. a docs-only checkout): leave a stub so
        # the toctree entry still resolves and the build does not fail.
        content = _INTRO + (
            "\n```{warning}\nThe Julia source was not found at build time, so the "
            "API reference could not be generated.\n```\n"
        )
    else:
        content = _build_page(src_root)

    existing = None
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as fh:
            existing = fh.read()
    if existing != content:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(content)


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
