from dataclasses import replace

from experiments.pravo_nn.mini_gpt.config import GPTConfig, CPU_OVERNIGHT


def test_default_config_is_consistent():
    cfg = GPTConfig()
    # n_embd must be divisible by n_head (heads split the embedding evenly)
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.block_size > 0
    assert cfg.vocab_size > 256  # at least the 256 byte base + merges


def test_preset_is_a_gptconfig_and_overridable():
    assert isinstance(CPU_OVERNIGHT, GPTConfig)
    assert CPU_OVERNIGHT.n_embd % CPU_OVERNIGHT.n_head == 0
    smaller = replace(CPU_OVERNIGHT, n_layer=2)
    assert smaller.n_layer == 2
    assert smaller.n_embd == CPU_OVERNIGHT.n_embd  # other fields preserved
