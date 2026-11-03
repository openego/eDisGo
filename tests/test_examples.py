import logging
import os

import nbformat
import pytest

from nbclient import NotebookClient


class TestExamples:
    @classmethod
    def setup_class(cls):
        cls.examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )

    @pytest.mark.slow
    def test_plot_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "plot_example.ipynb")
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3").execute()

    @pytest.mark.slow
    @pytest.mark.oep
    def test_electromobility_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "electromobility_example.ipynb")
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3").execute()

    @pytest.mark.slow
    @pytest.mark.oep
    def test_edisgo_simple_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "edisgo_simple_example.ipynb")
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3").execute()

    @classmethod
    def teardown_class(cls):
        logger = logging.getLogger("edisgo")
        logger.handlers.clear()
        logger.propagate = True
