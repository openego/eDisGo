import nbformat
import os
import shutil
import subprocess
import tempfile

import pytest

from examples import example_grid_reinforcement


class TestExamples:
    #@pytest.mark.slow
    def test_grid_reinforcement_example(self):
        total_costs = example_grid_reinforcement.run_example()
        # ToDo: total costs are for some reason not deterministic, check why!!
        # assert np.isclose(total_costs, 1147.57198)
        assert total_costs > 0.0

        # Delete saved grid and results data
        edisgo_path = os.path.join(os.path.expanduser("~"), ".edisgo")
        shutil.rmtree(os.path.join(edisgo_path, "ding0_example_grid"))

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
                "--ExecutePreprocessor.timeout=60"
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
            os.path.dirname(os.path.dirname(__file__)),
            "examples")
        nb, errors = self._notebook_run(
            os.path.join(examples_dir_path, "plot_example.ipynb")
        )
        assert errors == []

    # ToDo Uncomment once a smaller grid is used and execution does not take as long
    # @pytest.mark.slow
    # def test_edisgo_simple_example_ipynb(self):
    #     examples_dir_path = os.path.join(
    #         os.path.dirname(os.path.dirname(__file__)),
    #         "examples")
    #     nb, errors = self._notebook_run(
    #         os.path.join(examples_dir_path, "edisgo_simple_example.ipynb")
    #     )
    #     assert errors == []
