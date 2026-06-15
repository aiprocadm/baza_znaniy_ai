import numpy as np
import torch

from experiments.pravo_nn.mini_gpt.data import encode_corpus, load_bin, get_batch
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = "Статья 1. Основные начала.\nСтатья 2. Регулируемые отношения.\n" * 20


def _tok():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    return tok


def test_encode_corpus_writes_uint16_bin(tmp_path):
    tok = _tok()
    out = tmp_path / "train.bin"
    n = encode_corpus(SAMPLE, tok, out)
    assert out.exists()
    arr = load_bin(out)
    assert arr.dtype == np.uint16
    assert len(arr) == n


def test_get_batch_shapes_and_shift(tmp_path):
    tok = _tok()
    out = tmp_path / "train.bin"
    encode_corpus(SAMPLE, tok, out)
    data = load_bin(out)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.int64
    # within each sampled window, y is x shifted by one position
    x1, y1 = get_batch(
        data, block_size=8, batch_size=4, device="cpu",
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(x1[:, 1:], y1[:, :-1])
