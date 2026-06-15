from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = (
    "Статья 1. Основные начала гражданского законодательства.\n"
    "Статья 2. Отношения, регулируемые гражданским законодательством.\n"
    "Гражданское законодательство основывается на признании равенства."
)


def test_roundtrip_is_lossless():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_roundtrip_on_unseen_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    other = "Совершенно новый текст 42 — с пунктуацией!"
    assert tok.decode(tok.encode(other)) == other  # byte-level: never OOV


def test_training_reduces_token_count_vs_raw_bytes():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    n_tokens = len(tok.encode(SAMPLE))
    n_bytes = len(SAMPLE.encode("utf-8"))
    assert n_tokens < n_bytes  # merges must compress something


def test_vocab_size_is_respected():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert len(tok.vocab) == 300


def test_special_token_is_atomic():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])
    ids = tok.encode("привет<|endoftext|>мир", allowed_special=True)
    eot_id = tok.special_tokens["<|endoftext|>"]
    assert ids.count(eot_id) == 1
    assert tok.decode(ids) == "привет<|endoftext|>мир"


def test_save_load_round_trips(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])
    tok.save(tmp_path)
    reloaded = BPETokenizer.load(tmp_path)
    assert reloaded.encode(SAMPLE) == tok.encode(SAMPLE)
    assert reloaded.special_tokens == tok.special_tokens


def test_split_pattern_is_persisted_and_restored(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert tok.split_pattern  # populated after training
    tok.save(tmp_path)
    assert (tmp_path / "tokenizer_config.json").exists()
    reloaded = BPETokenizer.load(tmp_path)
    assert reloaded.split_pattern == tok.split_pattern


def test_loaded_tokenizer_ignores_changed_module_constant(tmp_path, monkeypatch):
    import experiments.pravo_nn.mini_gpt.tokenizer as tokmod
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    tok.save(tmp_path)
    expected = tok.encode(SAMPLE)
    # Simulate a future edit to the module-level split constant.
    monkeypatch.setattr(tokmod, "_SPLIT_RE", tokmod.re.compile(r"."))
    reloaded = BPETokenizer.load(tmp_path)
    assert reloaded.encode(SAMPLE) == expected  # uses the saved pattern, not the changed module one
