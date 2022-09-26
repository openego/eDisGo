import logging
import os
import subprocess
import tempfile

import nbformat
import pytest


class TestExamples:
    def _notebook_run(self, path):
        """
        Execute a notebook via nbconvert and collect output.
        Returns (parsed nb object, execution errors)
        """
        dirname, __ = os.path.split(path)
        os.chdir(dirname)
        with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
            args = [
                "jupyter",
                "nbconvert",
                path,
                "--output",
                fout.name,
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=90",
            ]
            subprocess.check_call(args)

            fout.seek(0)
            nb = nbformat.read(fout, nbformat.current_nbformat)

        errors = [
            output
            for cell in nb.cells
            if "outputs" in cell
            for output in cell["outputs"]
            if output.output_type == "error"
        ]

        return nb, errors

    @pytest.mark.slow
    def test_plot_example_ipynb(self):
        examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        nb, errors = self._notebook_run(
            os.path.join(examples_dir_path, "plot_example.ipynb")
        )
        assert errors == []

    @pytest.mark.slow
    def test_electromobility_example_ipynb(self):
        examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        nb, errors = self._notebook_run(
            os.path.join(examples_dir_path, "electromobility_example.ipynb")
        )
        assert errors == []

    @pytest.mark.slow
    def test_edisgo_simple_example_ipynb(self):
        examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        nb, errors = self._notebook_run(
            os.path.join(examples_dir_path, "edisgo_simple_example.ipynb")
        )
        assert errors == []

    @classmethod
    def teardown_class(cls):
        logger = logging.getLogger("edisgo")
        logger.propagate = True
