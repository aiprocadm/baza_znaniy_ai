import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.model import GPT


def _tiny_cfg():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_forward_returns_logits_and_loss_shapes():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_one_optimizer_step_reduces_loss_on_tiny_batch():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, loss0 = model(x, targets=y)
    for _ in range(50):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
    _, loss1 = model(x, targets=y)
    assert loss1.item() < loss0.item()  # overfit probe: backward path works


def test_generate_appends_exactly_n_in_vocab_tokens():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 6)  # 1 prompt token + 5 new
    assert int(out.max()) < cfg.vocab_size and int(out.min()) >= 0
