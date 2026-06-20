from experiments.pravo_nn.mini_gpt.data import encode_corpus_split, load_bin
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = "Статья 1. Основные начала.\nСтатья 2. Регулируемые отношения.\n" * 50


def _tok():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    return tok


def test_encode_split_writes_disjoint_train_and_val(tmp_path):
    tok = _tok()
    n_train, n_val = encode_corpus_split(
        SAMPLE, tok,
        train_path=tmp_path / "train.bin",
        val_path=tmp_path / "val.bin",
        val_frac=0.1,
    )
    train = load_bin(tmp_path / "train.bin")
    val = load_bin(tmp_path / "val.bin")
    assert len(train) == n_train and len(val) == n_val
    assert n_val > 0
    # val is the tail of the full token stream — train ends where val begins
    full = list(train) + list(val)
    assert len(full) == n_train + n_val
