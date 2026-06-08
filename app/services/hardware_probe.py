"""Advisory hardware probe. Detects, never decides. Pure core + thin OS wrapper."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    total_ram_gb: Optional[float]
    cores: Optional[int]
    has_cuda: bool
    ram_warning: bool
    advice: str


def probe(
    *,
    total_ram_gb: Optional[float],
    cores: Optional[int],
    has_cuda: bool,
    model_needs_gb: float,
) -> ProbeResult:
    """Pure decision core (injected facts) - easy to test deterministically."""
    ram_warning = total_ram_gb is not None and total_ram_gb < model_needs_gb
    advice = ""
    if ram_warning:
        advice = (
            f"Available RAM ~{total_ram_gb:.1f} GB is below the ~{model_needs_gb:.1f} GB "
            f"the bundled model needs. Use the lighter 'api' profile with an external key, "
            f"or a smaller model. Startup continues but inference may be slow or fail."
        )
        LOGGER.warning(advice)
    return ProbeResult(total_ram_gb, cores, has_cuda, ram_warning, advice)


def probe_system(model_needs_gb: float = 4.0) -> ProbeResult:
    """Thin OS wrapper: gather real facts, then call the pure core."""
    ram_gb: Optional[float] = None
    try:
        import psutil  # optional

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = None
    cores = os.cpu_count()
    has_cuda = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return probe(total_ram_gb=ram_gb, cores=cores, has_cuda=has_cuda, model_needs_gb=model_needs_gb)
