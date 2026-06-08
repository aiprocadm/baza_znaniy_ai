"""Hardware probe is advisory: it warns but never decides or raises."""

from __future__ import annotations

from app.services.hardware_probe import probe, ProbeResult


def test_enough_ram_no_warning():
    r = probe(total_ram_gb=16.0, cores=8, has_cuda=False, model_needs_gb=4.0)
    assert isinstance(r, ProbeResult)
    assert r.ram_warning is False
    assert r.advice == ""


def test_low_ram_warns_with_advice():
    r = probe(total_ram_gb=2.0, cores=2, has_cuda=False, model_needs_gb=4.0)
    assert r.ram_warning is True
    assert "api" in r.advice.lower()  # suggests the lighter api profile


def test_probe_never_raises_on_unknown():
    r = probe(total_ram_gb=None, cores=None, has_cuda=False, model_needs_gb=4.0)
    assert r.ram_warning is False  # unknown -> no false alarm, no crash
