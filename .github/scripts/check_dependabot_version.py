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
Fail a Dependabot branch's CI when pip did not install the version Dependabot
bumped to.

Dependabot widens ranges, e.g. ``pytest < 9.2.0`` to ``pytest < 9.3.0``. If
something else holds the package back, pip silently resolves to an older release
that still satisfies the widened range, and CI turns green without ever running
the new release. This script turns that case into a failed job.

It reads the bumped packages and versions from the ``updated-dependencies``
block of the newest Dependabot commit on the branch and compares them with what
is installed. On branches other than ``dependabot/pip/*`` it does nothing.

Run it after installing the dependencies::

    python .github/scripts/check_dependabot_version.py
"""

import os
import re
import subprocess
import sys

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

BRANCH_PREFIX = "dependabot/pip/"
DEPTH = 20


def dependabot_updates():
    """Return ``[(name, version), ...]`` from the newest Dependabot commit."""
    # actions/checkout fetches a single commit; the Dependabot commit may sit
    # below commits added by hand or by "Update branch".
    subprocess.run(
        ["git", "fetch", "--quiet", f"--deepen={DEPTH}", "origin"], check=False
    )
    log = subprocess.run(
        ["git", "log", "--first-parent", f"-n{DEPTH}", "--format=%B%x00"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for message in log.split("\0"):
        updates = re.findall(
            r"- dependency-name: (\S+)\s+dependency-version: (\S+)", message
        )
        if updates:
            return updates
    return []


def main():
    branch = os.environ.get("GITHUB_REF_NAME", "")
    if not branch.startswith(BRANCH_PREFIX):
        print(f"Not a {BRANCH_PREFIX}* branch, nothing to check.")
        return 0

    updates = dependabot_updates()
    if not updates:
        print(
            f"::error::No Dependabot commit with 'updated-dependencies' found in "
            f"the last {DEPTH} commits of {branch}."
        )
        return 1

    python = ".".join(map(str, sys.version_info[:2]))
    failed = False
    for name, target in updates:
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"::warning::{name} is not installed in this job.")
            continue
        if Version(installed) < Version(target):
            failed = True
            print(
                f"::error::{name} {installed} is installed, but Dependabot bumped it "
                f"to {target}. pip fell back to an older release, so this job does "
                f"not test {target}. Either another requirement conflicts with it, "
                f"or {name} {target} no longer supports Python {python}."
            )
        else:
            print(f"{name} {installed} is installed (Dependabot: {target}).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
