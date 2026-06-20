"""Detached warm-start entrypoint: logs to a file via FileHandler (flushes per
record — Start-Process -RedirectStandardError buffers and looks frozen)."""

from __future__ import annotations

import logging
from pathlib import Path

from experiments.pravo_nn.mini_gpt.train import train

_DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    log = _DATA / "checkpoints" / "warmstart_run.log"
    handler = logging.FileHandler(log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger(__name__).info("RUN START (warm-start on mixed corpus)")
    train(
        max_steps=4000,
        batch_size=8,
        eval_interval=250,
        resume_from=_DATA / "checkpoints" / "ckpt_v1.pt",
    )
    logging.getLogger(__name__).info("RUN COMPLETE")
