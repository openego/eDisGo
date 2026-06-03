import logging
import os
import subprocess
import tempfile

import nbformat
import pytest
import pytest_notebook


class TestExamples:
    @classmethod
    def setup_class(self):
        self.examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )

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
        path = os.path.join(self.examples_dir_path, "plot_example.ipynb")
        notebook = pytest_notebook.notebook.load_notebook(path=path)
        result = pytest_notebook.execution.execute_notebook(
            notebook,
            with_coverage=False,
            timeout=600,
        )
        if result.exec_error is not None:
            print(result.exec_error)
        assert result.exec_error is None

    @pytest.mark.slow
    def test_electromobility_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "electromobility_example.ipynb")
        notebook = pytest_notebook.notebook.load_notebook(path=path)
        result = pytest_notebook.execution.execute_notebook(
            notebook,
            with_coverage=False,
            timeout=600,
        )
        if result.exec_error is not None:
            print(result.exec_error)
        assert result.exec_error is None

    @pytest.mark.slow
    def test_edisgo_simple_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "edisgo_simple_example.ipynb")
        notebook = pytest_notebook.notebook.load_notebook(path=path)
        result = pytest_notebook.execution.execute_notebook(
            notebook,
            with_coverage=False,
            timeout=600,
        )
        if result.exec_error is not None:
            print(result.exec_error)
        assert result.exec_error is None

    @classmethod
    def teardown_class(cls):
        logger = logging.getLogger("edisgo")
        logger.handlers.clear()
        logger.propagate = True

    # @pytest.mark.slow
    def test_edisgo_documentation_examples_ipynb(self):
        examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        nb, errors = self._notebook_run(
            os.path.join(examples_dir_path, "edisgo_documentation_examples.ipynb")
        )
        assert errors == []
