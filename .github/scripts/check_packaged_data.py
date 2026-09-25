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
"""
Check that all non-Python data files eDisGo needs at runtime actually ship in the
built distributions.

Everybody develops with ``pip install -e .``, where every file is reachable in the
source tree, and the test workflow installs from the checkout too -- so a missing
``package_data`` entry is invisible until someone installs a built artifact. That is
exactly how the Julia OPF sources (``edisgo/opf/eDisGo_OPF.jl``) went missing from
every release up to and including v0.2.1, breaking
:meth:`~edisgo.edisgo.EDisGo.pm_optimize` for non-editable installs such as eGo's
``edisgo @ git+https://github.com/openego/eDisGo.git@dev``.

Run it after building both artifacts::

    rm -rf dist build *.egg-info    # see the warning about stale egg-info below
    python -m build
    python .github/scripts/check_packaged_data.py

A stale ``eDisGo.egg-info/SOURCES.txt`` in the working tree makes the sdist look
complete even when ``package_data`` is wrong: setuptools keeps that manifest around
between builds and only ever adds to it, so files from an earlier (correct) build keep
being packed. Always delete it before checking a release build -- CI is unaffected
because it builds from a fresh checkout, where egg-info does not exist.

"""

from __future__ import annotations

import glob
import posixpath
import sys
import tarfile
import zipfile

# Files that must be present in both the sdist and the wheel, as paths relative to
# the "edisgo" package directory.
REQUIRED_FILES = [
    "opf/eDisGo_OPF.jl/Main.jl",
    "opf/eDisGo_OPF.jl/Project.toml",
    "opf/eDisGo_OPF.jl/Manifest.toml",
    "opf/eDisGo_OPF.jl/src/eDisGo_OPF.jl",
    "config/config_db_tables_default.cfg",
    "equipment/equipment-parameters_MV_cables.csv",
]

# Minimum number of matches per glob-like suffix, to catch a partially shipped tree
# (e.g. only the top level of eDisGo_OPF.jl, or a single preset).
REQUIRED_COUNTS = {
    "opf/eDisGo_OPF.jl/": 16,
    "run/presets/": 5,
    "config/": 6,
    "equipment/": 5,
}

# Nothing matching these may ever be shipped. OEP_TOKEN.txt lives in edisgo/config in
# many local checkouts (it is git-ignored, but sdists are built from the working tree).
FORBIDDEN_SUBSTRINGS = ["token", "secret", ".env"]


def _wheel_members(path: str) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def _sdist_members(path: str) -> list[str]:
    with tarfile.open(path, "r:gz") as tf:
        # strip the leading "eDisGo-<version>/" component
        return [
            member.name.split("/", 1)[1]
            for member in tf.getmembers()
            if member.isfile() and "/" in member.name
        ]


def _check(kind: str, path: str, members: list[str]) -> list[str]:
    errors = []
    package_files = [
        posixpath.relpath(name, "edisgo")
        for name in members
        if name.startswith("edisgo/")
    ]

    for required in REQUIRED_FILES:
        if required not in package_files:
            errors.append(f"{kind} {path}: missing edisgo/{required}")

    for prefix, minimum in REQUIRED_COUNTS.items():
        found = len([name for name in package_files if name.startswith(prefix)])
        if found < minimum:
            errors.append(
                f"{kind} {path}: only {found} file(s) under edisgo/{prefix}, "
                f"expected at least {minimum}"
            )

    for name in members:
        lowered = name.lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in lowered:
                errors.append(f"{kind} {path}: must not ship {name}")

    return errors


def main() -> int:
    wheels = sorted(glob.glob("dist/*.whl"))
    sdists = sorted(glob.glob("dist/*.tar.gz"))

    if not wheels or not sdists:
        print(
            "error: expected a wheel and an sdist in dist/ -- run 'python -m build' "
            f"first (found wheels={wheels}, sdists={sdists})",
            file=sys.stderr,
        )
        return 1

    if glob.glob("*.egg-info"):
        # The build itself creates egg-info, so this cannot tell whether a *stale* one
        # was present -- say what is trustworthy instead of crying wolf.
        print(
            "note: the wheel result below is authoritative. The sdist result is only "
            "meaningful if *.egg-info was deleted before building: its cached "
            "SOURCES.txt keeps packing files from earlier builds.",
            file=sys.stderr,
        )

    errors = []
    for path in wheels:
        errors += _check("wheel", path, _wheel_members(path))
    for path in sdists:
        errors += _check("sdist", path, _sdist_members(path))

    if errors:
        print("Packaged data files are incomplete:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nData files are declared in setup.py's package_data. Note that the "
            "patterns are resolved relative to the package they are listed under, so "
            "files inside a subpackage (edisgo/opf) need their own key.",
            file=sys.stderr,
        )
        return 1

    print(f"Packaged data files OK in {len(wheels)} wheel(s), {len(sdists)} sdist(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
