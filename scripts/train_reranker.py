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


def group_by_query(rows: list[dict]) -> list[list[dict]]:
    """Group candidate rows by their ``query``, preserving first-seen order.

    Used by the pairwise loss path so each training group holds a whole query's
    candidate set (needed to form within-query (positive, negative) pairs).
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["query"], []).append(r)
    return list(groups.values())


def enumerate_pairs(group: list[dict], *, margin: float = 0.0) -> list[tuple[int, int]]:
    """Enumerate within-query (i, j) index pairs where candidate ``i`` outranks
    ``j`` by teacher score (``soft_label(score_i) - soft_label(score_j) > margin``).

    Returns *ordered* index pairs into ``group``; the first element is the more
    relevant ("positive") candidate. These feed a RankNet-style logistic pairwise
    loss: for each pair we want ``logit_i > logit_j``, i.e. the student reproduces
    the teacher's within-query ranking rather than its absolute scores. Pairwise
    ranking is far more sample-efficient than pointwise BCE when the number of
    unique queries is small (v1's failure mode), because the supervisory signal is
    the O(k^2) orderings inside each query, not k absolute targets.

    Ties (and near-ties within ``margin``) produce no pair, so they contribute no
    gradient. Order of returned pairs is deterministic (i ascending, then j).
    """
    labels = [soft_label(float(r["teacher_score"])) for r in group]
    pairs: list[tuple[int, int]] = []
    n = len(group)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if labels[i] - labels[j] > margin:
                pairs.append((i, j))
    return pairs


def query_grouped_batches(rows: list[dict], *, seed: int, min_pairs: int = 1) -> list[list[dict]]:
    """Build per-query batches for the pairwise path (one query == one batch).

    Queries are shuffled deterministically by ``seed``. Groups that cannot form at
    least ``min_pairs`` within-query pairs (e.g. all-equal teacher scores, or a
    single candidate) are dropped — they would contribute no ranking gradient.
    """
    groups = group_by_query(rows)
    groups = [g for g in groups if len(enumerate_pairs(g)) >= min_pairs]
    rng = random.Random(seed)
    rng.shuffle(groups)
    return groups


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
    loss: str = "bce",
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

    if loss == "pairwise":
        _train_pairwise(
            model,
            collate,
            rows_train,
            torch=torch,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
    else:
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
                loss_val = loss_fn(logits, labels)
                loss_val.backward()
                optimizer.step()
                total += float(loss_val)
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{len(loader)} loss {float(loss_val):.4f}")
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


def _train_pairwise(model, collate, rows_train, *, torch, epochs, lr, seed) -> None:
    """RankNet-style pairwise training loop (one query == one batch).

    Not unit-tested (needs torch); the pure pairing logic in ``enumerate_pairs`` /
    ``query_grouped_batches`` is. For each query we score all candidates once, then
    apply ``softplus(logit_j - logit_i)`` (== ``-log sigmoid(logit_i - logit_j)``)
    over every teacher-ordered pair so the student learns within-query ranking.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        batches = query_grouped_batches(rows_train, seed=seed + epoch)
        total = 0.0
        n_steps = 0
        for step, group in enumerate(batches):
            pairs = enumerate_pairs(group)
            if not pairs:
                continue
            enc = collate(group)
            enc.pop("labels")
            optimizer.zero_grad()
            logits = model(**enc).logits.squeeze(-1)
            i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            diff = logits[i_idx] - logits[j_idx]
            loss_val = torch.nn.functional.softplus(-diff).mean()
            loss_val.backward()
            optimizer.step()
            total += float(loss_val)
            n_steps += 1
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(batches)} "
                    f"pairs {len(pairs)} loss {float(loss_val):.4f}"
                )
        print(f"epoch {epoch} mean pairwise loss {total / max(1, n_steps):.4f}")


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
    parser.add_argument(
        "--loss",
        choices=("bce", "pairwise"),
        default="bce",
        help="bce = pointwise BCE vs teacher soft scores (v1 default); "
        "pairwise = RankNet within-query ranking (better at low query counts)",
    )
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
        loss=args.loss,
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
        "loss": args.loss,
        **metrics,
    }
    (Path(args.out) / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
