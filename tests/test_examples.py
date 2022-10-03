import logging
import os

import pytest
import pytest_notebook


class TestExamples:
    @classmethod
    def setup_class(self):
        self.examples_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )

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
