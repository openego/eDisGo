"""Check token validation without creating engines or contacting the OEP."""

from types import SimpleNamespace
from unittest.mock import Mock

import conftest as test_config
import pytest


@pytest.mark.parametrize(
    "env_token, file_token, accepted",
    [
        (None, None, False),
        (None, b"", False),
        (None, b"a" * 40, True),
        (None, b" \n" + b"a" * 40 + b"\n", True),
        (None, b"invalid-file-token", False),
        (None, b"\xff", False),
        ("b" * 40, None, True),
        ("b" * 40, b"invalid-file-token", True),
        ("", b"a" * 40, False),
        (" \t", b"a" * 40, False),
        ("invalid-env-token", b"a" * 40, False),
        ("a" * 39, None, False),
        ("A" * 40, None, False),
    ],
    ids=[
        "missing",
        "empty-file",
        "valid-file",
        "file-whitespace-trimmed",
        "invalid-file",
        "unreadable-encoding",
        "valid-env",
        "env-overrides-invalid-file",
        "empty-env-overrides-valid-file",
        "whitespace-env-overrides-valid-file",
        "invalid-env-overrides-valid-file",
        "short-env",
        "uppercase-env",
    ],
)
def test_oep_engine_token_validation(
    monkeypatch, tmp_path, env_token, file_token, accepted
):
    if env_token is None:
        monkeypatch.delenv("OEP_TOKEN", raising=False)
    else:
        monkeypatch.setenv("OEP_TOKEN", env_token)

    token_path = tmp_path / "config" / "OEP_TOKEN.txt"
    if file_token is not None:
        token_path.parent.mkdir()
        token_path.write_bytes(file_token)
    find_spec = Mock(return_value=SimpleNamespace(origin=str(tmp_path / "__init__.py")))
    monkeypatch.setattr(
        test_config,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=find_spec)),
    )
    engine_factory = Mock()
    monkeypatch.setattr(test_config, "engine", engine_factory)
    request = SimpleNamespace(
        node=SimpleNamespace(get_closest_marker=lambda _: object())
    )

    # Invoke the fixture's generator directly so this regression test stays offline.
    fixture = test_config.oep_engine.__wrapped__(request)
    if accepted:
        assert next(fixture) is engine_factory.return_value
        engine_factory.assert_called_once_with(ssh=False)
        with pytest.raises(StopIteration):
            next(fixture)
        engine_factory.return_value.dispose.assert_called_once_with()
    else:
        with pytest.raises(pytest.fail.Exception) as failure:
            next(fixture)
        engine_factory.assert_not_called()
        message = str(failure.value)
        assert "OEP token" in message
        # Values supplied by users must never be exposed in validation errors.
        for value in (env_token, file_token):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            if value and len(value.strip()) >= 8:
                assert value not in message

    if env_token is not None:
        find_spec.assert_not_called()
