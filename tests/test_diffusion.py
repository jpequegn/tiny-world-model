"""Tests for the masked diffusion language model."""

import torch
import pytest

from src.diffusion.generate import generate, generate_autoregressive
from src.diffusion.model import DiffusionTransformer
from src.diffusion.noise import forward_diffuse, mask_rate, sample_timesteps
from src.diffusion.train import CharDataset, DiffusionTrainConfig, train_diffusion
from src.diffusion.vocab import MASK_ID, VOCAB_SIZE, decode, encode

SEQ_LEN = 16
T = 20


def _small_model() -> DiffusionTransformer:
    return DiffusionTransformer(
        vocab_size=VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1, max_len=SEQ_LEN
    )


# ── Vocab ─────────────────────────────────────────────────────────────────────

def test_encode_decode_roundtrip():
    text = "hello world"
    assert decode(encode(text)) == text


def test_mask_id_not_in_decode():
    ids = [MASK_ID, MASK_ID, *encode("hi")]
    assert "<MASK>" not in decode(ids)


def test_vocab_size_positive():
    assert VOCAB_SIZE > 2


# ── Noise ─────────────────────────────────────────────────────────────────────

def test_mask_rate_bounds():
    t = torch.tensor([0, 5, 10, 20])
    r = mask_rate(t, T=20)
    assert (r >= 0).all() and (r <= 1).all()


def test_forward_diffuse_shape():
    x = torch.randint(2, VOCAB_SIZE, (4, SEQ_LEN))
    t = sample_timesteps(4, T, torch.device("cpu"))
    x_t = forward_diffuse(x, t, T, MASK_ID)
    assert x_t.shape == x.shape


def test_forward_diffuse_t0_no_mask():
    """At t=0 nothing should be masked."""
    x = torch.randint(2, VOCAB_SIZE, (8, SEQ_LEN))
    t = torch.zeros(8, dtype=torch.long)
    x_t = forward_diffuse(x, t, T, MASK_ID)
    assert (x_t == x).all()


def test_forward_diffuse_tT_all_masked():
    """At t=T everything should be masked (with high probability)."""
    torch.manual_seed(42)
    x = torch.randint(2, VOCAB_SIZE, (64, SEQ_LEN))
    t = torch.full((64,), T, dtype=torch.long)
    x_t = forward_diffuse(x, t, T, MASK_ID)
    mask_frac = (x_t == MASK_ID).float().mean().item()
    assert mask_frac > 0.95


# ── Model ─────────────────────────────────────────────────────────────────────

def test_model_output_shape():
    model = _small_model()
    x = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
    t = torch.randint(1, T + 1, (4,))
    logits = model(x, t)
    assert logits.shape == (4, SEQ_LEN, VOCAB_SIZE)


def test_model_param_count():
    model = _small_model()
    assert 0 < model.num_parameters() < 1_000_000


def test_model_gradients_flow():
    model = _small_model()
    x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    t = torch.randint(1, T + 1, (2,))
    loss = model(x, t).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


# ── Training ──────────────────────────────────────────────────────────────────

def test_train_loss_decreases():
    model = _small_model()
    cfg = DiffusionTrainConfig(T=T, epochs=30, batch_size=8, lr=3e-4, seq_len=SEQ_LEN, device="cpu")
    history = train_diffusion(model, cfg, sentences=["the cat sat on the mat"] * 8)
    assert history[-1] < history[0], f"Loss did not decrease: {history[0]:.4f} → {history[-1]:.4f}"


def test_char_dataset_len_and_shape():
    ds = CharDataset(["hello world", "foo bar"], seq_len=SEQ_LEN)
    assert len(ds) == 2
    item = ds[0]
    assert item.shape == (SEQ_LEN,)
    assert item.dtype == torch.long


# ── Generation ────────────────────────────────────────────────────────────────

def test_generate_returns_string():
    model = _small_model()
    text, trajectory = generate(model, seq_len=SEQ_LEN, steps=5, T=T)
    assert isinstance(text, str)
    assert len(trajectory) == 5


def test_generate_trajectory_progresses():
    """Each step should unmask more tokens (trajectory grows or stays same length)."""
    model = _small_model()
    _, trajectory = generate(model, seq_len=SEQ_LEN, steps=8, T=T)
    # The last step should have revealed at least as many chars as the first
    assert len(trajectory[-1]) >= len(trajectory[0])


def test_generate_autoregressive_returns_string():
    model = _small_model()
    text = generate_autoregressive(model, seq_len=SEQ_LEN, T=T)
    assert isinstance(text, str)
