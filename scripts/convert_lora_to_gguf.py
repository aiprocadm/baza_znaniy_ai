#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter into GGUF format for llama.cpp runtimes."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LoRA adapter to GGUF")
    parser.add_argument("--base-model", required=True, help="Base GGUF model used during conversion")
    parser.add_argument("--adapter", required=True, type=Path, help="Path to adapter directory or safetensors file")
    parser.add_argument("--out", required=True, type=Path, help="Output GGUF file path")
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Optional path to llama.cpp convert-lora-to-gguf.py script (defaults to python -m llama_cpp.convert_lora)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def convert(args: argparse.Namespace) -> Path:
    adapter_path = args.adapter.expanduser().resolve()
    if adapter_path.is_file():
        adapter_dir = adapter_path.parent
    else:
        adapter_dir = adapter_path

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.script:
        script_path = args.script.expanduser().resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"Conversion script not found: {script_path}")
        cmd = [sys.executable, str(script_path)]
    else:
        cmd = [sys.executable, "-m", "llama_cpp.convert_lora"]

    cmd.extend(
        [
            "--to-gguf",
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(adapter_dir),
            "--output",
            str(output_path),
        ]
    )

    LOGGER.info("Running conversion: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        LOGGER.error("Conversion failed: %s", completed.stderr.strip())
        raise RuntimeError(f"Conversion command failed with exit code {completed.returncode}")

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file not created: {output_path}")

    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path = convert(args)
    except Exception:
        LOGGER.exception("LoRA conversion failed")
        return 1

    print(str(output_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
