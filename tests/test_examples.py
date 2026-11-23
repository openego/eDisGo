import logging
import os

from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest

from nbclient import NotebookClient


class TestExamples:
    @classmethod
    def setup_class(cls):
        cls.examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )

    @pytest.mark.parametrize("working_directory", [".", "examples"])
    def test_plot_example_uses_local_grid(self, monkeypatch, working_directory):
        """Repository execution must select local inputs without downloading."""
        repo_root = Path(self.examples_dir_path).parent
        monkeypatch.chdir(repo_root / working_directory)
        nb = nbformat.read(repo_root / "examples" / "plot_example.ipynb", as_version=4)
        grid_loader = next(
            cell.source
            for cell in nb.cells
            if cell.cell_type == "code"
            and cell.source.startswith("def get_ding0_example_grid():")
        )
        requests_mock = Mock()
        requests_mock.get.side_effect = AssertionError("Unexpected grid download")
        namespace = {"os": os, "Path": Path, "requests": requests_mock}
        exec(grid_loader, namespace)

        grid_path = namespace["get_ding0_example_grid"]()

        assert Path(grid_path) == repo_root / "tests/data/ding0_test_network_1"
        requests_mock.get.assert_not_called()

    @pytest.mark.slow
    def test_plot_example_ipynb(self):
        path = os.path.join(self.examples_dir_path, "plot_example.ipynb")
        nb = nbformat.read(path, as_version=4)
        NotebookClient(
            nb,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": os.path.dirname(self.examples_dir_path)}},
        ).execute()

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
