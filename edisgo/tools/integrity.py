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
class IntegrityCollector:
    """Collect integrity issues and report them according to the selected mode."""

    def __init__(self, errors="warn"):
        if errors not in ("warn", "raise"):
            raise ValueError("errors must be 'warn' or 'raise'")

        self.errors = errors
        self.messages = []

    def report(self, message, logger):
        if self.errors == "raise":
            self.messages.append(message)
        else:
            logger.warning(message)

    def raise_if_errors(self):
        if self.errors == "raise" and self.messages:
            message = (
                f"Integrity check found {len(self.messages)} issue(s):\n- "
                + "\n- ".join(self.messages)
            )
            raise ValueError(message)
