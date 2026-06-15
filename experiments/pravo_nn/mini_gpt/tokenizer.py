"""Byte-level BPE tokenizer, implemented from scratch (minbpe-style, but trained
over frequency-counted chunks for speed). Base vocab is the 256 bytes; merges add
ids 256.. . Encoding is byte-level, so any input round-trips losslessly.

Files written by `save`: `vocab.json` (id -> hex of the token's bytes),
`merges.txt` (one `a b` int pair per line, in merge order), and
`special_tokens.json`."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Split into maximal runs of whitespace OR non-whitespace; keeps spaces attached
# to their run so the BPE can learn space-prefixed legal tokens (" Кодекса").
_SPLIT_RE = re.compile(r"\s+|\S+")


def _merge_seq(seq: tuple[int, ...], pair: tuple[int, int], idx: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    def train(
        self,
        text: str,
        *,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # Frequency-counted chunks -> each is a tuple of byte ids.
        chunk_counts = Counter(_SPLIT_RE.findall(text))
        words: dict[tuple[int, ...], int] = {}
        for chunk, cnt in chunk_counts.items():
            key = tuple(chunk.encode("utf-8"))
            words[key] = words.get(key, 0) + cnt

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for seq, cnt in words.items():
                for pair in zip(seq, seq[1:]):
                    stats[pair] = stats.get(pair, 0) + cnt
            if not stats:
                break
            best = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            merges[best] = idx
            vocab[idx] = vocab[best[0]] + vocab[best[1]]
            words = {_merge_seq(seq, best, idx): cnt for seq, cnt in words.items()}

        self.merges = merges
        self.vocab = vocab
        self.special_tokens = {}
        for st in special_tokens or []:
            self.special_tokens[st] = len(self.vocab) + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = list(_merge_seq(tuple(ids), pair, self.merges[pair]))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        if not allowed_special or not self.special_tokens:
            return self._encode_ordinary(text)
        # Split out special tokens, encode the gaps ordinarily.
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        out: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inv_special:
                parts.append(inv_special[i].encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vocab.json").write_text(
            json.dumps({str(i): b.hex() for i, b in self.vocab.items()}),
            encoding="utf-8",
        )
        lines = [f"{a} {b}" for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])]
        (out / "merges.txt").write_text("\n".join(lines), encoding="utf-8")
        (out / "special_tokens.json").write_text(
            json.dumps(self.special_tokens, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir) -> "BPETokenizer":
        src = Path(in_dir)
        tok = cls()
        raw_vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
        tok.vocab = {int(i): bytes.fromhex(h) for i, h in raw_vocab.items()}
        merges: dict[tuple[int, int], int] = {}
        text = (src / "merges.txt").read_text(encoding="utf-8")
        for rank, line in enumerate(filter(None, text.splitlines())):
            a, b = (int(x) for x in line.split())
            merges[(a, b)] = 256 + rank
        tok.merges = merges
        tok.special_tokens = json.loads((src / "special_tokens.json").read_text(encoding="utf-8"))
        return tok
