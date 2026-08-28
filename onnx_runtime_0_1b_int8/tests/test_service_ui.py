from __future__ import annotations

from arktts_runtime.web_ui import TEST_PAGE


def test_local_ui_has_complete_tts_controls() -> None:
    assert '<textarea id="text">' in TEST_PAGE
    assert 'id="voice"' in TEST_PAGE
    assert 'id="generate"' in TEST_PAGE
    assert 'id="audio" controls' in TEST_PAGE
    assert 'id="download"' in TEST_PAGE
    assert "fetch('/api/tts'" in TEST_PAGE


def test_local_ui_has_registration_and_runtime_controls() -> None:
    assert 'id="registerView"' in TEST_PAGE
    assert 'id="referenceAudio"' in TEST_PAGE
    assert "fetch('/api/voices/register'" in TEST_PAGE
    assert "fetch('/api/system'" in TEST_PAGE
    assert "fetch('/api/runtime/reload'" in TEST_PAGE
