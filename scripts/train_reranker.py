"""Train ``kbai-reranker-ru``: distill teacher scores into rubert-tiny2.

Student = ``cointegrated/rubert-tiny2`` + 1-logit sequence-classification head;
loss = BCEWithLogits against the teacher's soft scores (spec 2026-06-10 §3.3).
The saved directory is a plain HF checkpoint, loadable by
``sentence_transformers.CrossEncoder(<dir>)`` — i.e. directly usable as
``KB_RERANK_MODEL``. torch/transformers imports are lazy (stub safety).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

BASE_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_PAIRS = Path("var/data/rerank/pairs.jsonl")
DEFAULT_OUT = Path("var/models/kbai-reranker-ru")


def load_pairs(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise SystemExit(f"Empty pairs file: {path}")
    return rows


def soft_label(score: float) -> float:
    """Teacher scores are sigmoid-activated probabilities; tolerate raw logits."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


def split_by_query(
    rows: list[dict], *, val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Query-disjoint split: all pairs of one query land on the same side."""
    queries = sorted({r["query"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(queries)
    n_val = max(1, int(len(queries) * val_fraction))
    val_queries = set(queries[:n_val])
    train_rows = [r for r in rows if r["query"] not in val_queries]
    val_rows = [r for r in rows if r["query"] in val_queries]
    return train_rows, val_rows


def train(
    rows_train: list[dict],
    rows_val: list[dict],
    *,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    def collate(batch: list[dict]):
        enc = tokenizer(
            [b["query"] for b in batch],
            [b["text"] for b in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(
            [soft_label(float(b["teacher_score"])) for b in batch], dtype=torch.float32
        )
        return enc

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        rows_train, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for step, batch in enumerate(loader):
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {float(loss):.4f}")
        print(f"epoch {epoch} mean loss {total / max(1, len(loader)):.4f}")

    model.eval()
    loader_val = DataLoader(rows_val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    preds: list[float] = []
    gold: list[float] = []
    with torch.no_grad():
        for batch in loader_val:
            labels = batch.pop("labels")
            logits = model(**batch).logits.squeeze(-1)
            preds.extend(torch.sigmoid(logits).tolist())
            gold.extend(labels.tolist())
    pearson = float(np.corrcoef(preds, gold)[0, 1]) if len(preds) > 1 else float("nan")
    if math.isnan(pearson):
        print("WARNING: val_pearson is NaN — constant predictions? Check model output.")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {"val_pairs": len(preds), "val_pearson_vs_teacher": pearson}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="train_reranker")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows = load_pairs(Path(args.pairs))
    rows_train, rows_val = split_by_query(rows, val_fraction=args.val_fraction, seed=args.seed)
    if not rows_train:
        raise SystemExit(
            f"Empty train split ({len(rows_val)} val pairs from {len(rows)} total) — "
            "need at least 2 distinct queries."
        )
    print(f"train pairs: {len(rows_train)}, val pairs: {len(rows_val)}")
    metrics = train(
        rows_train,
        rows_val,
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )
    meta = {
        "base_model": BASE_MODEL,
        "pairs_file": args.pairs,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        **metrics,
    }
    (Path(args.out) / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
