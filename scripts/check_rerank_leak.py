"""One-off: verify zero golden->train query overlap (spec §3.4)."""

import json
import sys
from pathlib import Path

from app.eval.dataset import load_golden
from scripts.build_rerank_dataset import GOLDEN_PUBLIC, normalize_question

golden = {normalize_question(item.question) for item in load_golden(GOLDEN_PUBLIC)}
train = {
    normalize_question(json.loads(line)["query"])
    for line in Path("var/data/rerank/pairs.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
overlap = golden & train
print(f"leak overlap: {len(overlap)}")
sys.exit(1 if overlap else 0)
