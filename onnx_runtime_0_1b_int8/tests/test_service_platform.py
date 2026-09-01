from __future__ import annotations


def test_system_stats_does_not_require_unix_resource(monkeypatch) -> None:
    import arktts_runtime.service as service

    monkeypatch.setattr(service, "resource", None)
    monkeypatch.setattr(service.platform, "system", lambda: "Windows")

    result = service.system_stats()

    assert result["memory"]["current_mb"] is None
    assert result["memory"]["peak_mb"] is None
    assert result["uptime_seconds"] >= 0
