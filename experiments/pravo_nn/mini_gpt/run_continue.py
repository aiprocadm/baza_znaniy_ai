"""Detached continuation entrypoint: resume the LIVE ckpt.pt (not ckpt_v1) and
train further. Logs to a file via FileHandler (flushes per record — a detached
Start-Process buffers stderr and looks frozen). Step budget is argv[1].

Usage: py -3.13 -m experiments.pravo_nn.mini_gpt.run_continue [max_steps]
The checkpoint's absolute `step` continues (6000 -> 6000 + max_steps); the LR
schedule warm-restarts over this run's max_steps (a cosine restart, intended)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from experiments.pravo_nn.mini_gpt.train import train

_DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    log = _DATA / "checkpoints" / "continue_run.log"
    handler = logging.FileHandler(log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger(__name__).info("RUN START (continue from ckpt.pt, max_steps=%d)", max_steps)
    train(
        max_steps=max_steps,
        batch_size=8,
        log_interval=50,  # finer cadence (~6 min/line) so monitoring sees progress, not a 28-min blind gap
        eval_interval=250,
        resume_from=_DATA / "checkpoints" / "ckpt.pt",
    )
    logging.getLogger(__name__).info("RUN COMPLETE")
