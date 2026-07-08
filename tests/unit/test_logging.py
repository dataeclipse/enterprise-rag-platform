import json

import pytest

from rag.observability.logging import configure_logging, get_logger


def test_json_logging_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=True)
    get_logger("test").info("hello", key="value")
    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["key"] == "value"
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", json_output=True)
    log = get_logger("test")
    log.info("suppressed")
    log.warning("visible")
    out = capsys.readouterr().out
    assert "suppressed" not in out
    assert "visible" in out
