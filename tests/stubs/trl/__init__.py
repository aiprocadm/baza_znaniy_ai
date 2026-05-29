"""Minimal trl stub for offline tests.

Mirrors :class:`trl.DPOConfig` and :class:`trl.DPOTrainer` 0.11+
just enough for ``scripts/train_dpo.py`` to import, instantiate,
and call ``train()`` + ``save_model()``. ``DPOTrainer.train_calls``
is a class-level list so tests can assert on what was invoked.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DPOConfig:
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 4
    num_train_epochs: int = 1
    max_length: int = 1024
    max_prompt_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    max_steps: int = -1
    report_to: str = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


class DPOTrainer:
    """Stub mirroring the trl 0.11 DPOTrainer subset used by train_dpo.py."""

    train_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        args: DPOConfig | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        peft_config: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.peft_config = peft_config
        self._extras = kwargs

    def train(self) -> None:
        self.train_calls.append(
            {
                "model": self.model,
                "beta": self.args.beta if self.args is not None else None,
                "dataset_size": (len(self.train_dataset) if self.train_dataset is not None else 0),
            }
        )

    def save_model(self, output_dir: str) -> None:
        p = pathlib.Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}", encoding="utf-8")
