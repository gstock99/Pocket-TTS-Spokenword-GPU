import math

import torch
from torch import nn


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: int | torch.Tensor = 0,
    max_period: int | float = 10_000,
):
    """
    Args:
    q (torch.Tensor): Queries, shape `[B, T, H, D]`.
    k (torch.Tensor): Keys, shape `[B, T, H, D]`.
    offset (int): Current offset, e.g. when streaming.
    max_period (float): Maximum period for the cos and sin.
    """

    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk)
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    half = D // 2
    ds = torch.arange(half, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    if T == 1:
        ts = torch.tensor(offset, device=q.device, dtype=torch.float32).view(1, 1, 1)
    else:
        ts = torch.arange(T, device=q.device, dtype=torch.float32) + offset
        ts = ts.view(-1, 1, 1)

    q = q.view(B, T, H, half, 2)
    k = k.view(B, Tk, Hk, half, 2)

    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(B, T, H, D), ko.view(B, Tk, Hk, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Pre-computes and caches frequency bands to avoid per-call tensor allocation.
    Pre-allocates a ts buffer for common sequence lengths.

    Args:
        max_period (float): Maximum period of the rotation frequencies.
        max_seq_len (int): Maximum sequence length to pre-allocate ts buffer for.
    """

    def __init__(self, max_period: float | int = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.max_period = max_period
        self._max_seq_len = max_seq_len
        # Lazily initialized on first forward call
        self._freqs_cache = None
        self._cached_half = None
        self._ts_buffer = None

    def _ensure_freqs(self, half: int, device: torch.device) -> torch.Tensor:
        if self._freqs_cache is None or self._cached_half != half:
            ds = torch.arange(half, device=device, dtype=torch.float32)
            self._freqs_cache = torch.exp(ds * (-math.log(self.max_period) * 2 / (half * 2)))
            self._cached_half = half
        return self._freqs_cache

    def _ensure_ts_buffer(self, T: int, offset, device: torch.device) -> torch.Tensor:
        if T == 1:
            return torch.tensor(offset, device=device, dtype=torch.float32).view(1, 1, 1)
        if self._ts_buffer is None or self._ts_buffer.shape[0] < T:
            self._ts_buffer = torch.arange(self._max_seq_len, device=device, dtype=torch.float32)
        return (self._ts_buffer[:T] + offset).view(-1, 1, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor | int):
        """Apply rope rotation using cached frequency bands and time buffer."""
        B, T, H, D = q.shape
        Bk, Tk, Hk, Dk = k.shape
        assert (B, T, D) == (Bk, Tk, Dk)
        assert D > 0 and D % 2 == 0

        half = D // 2
        freqs = self._ensure_freqs(half, q.device)
        ts = self._ensure_ts_buffer(T, offset, q.device)

        q = q.view(B, T, H, half, 2)
        k = k.view(B, Tk, Hk, half, 2)

        qr = q[..., 0].float()
        qi = q[..., 1].float()
        kr = k[..., 0].float()
        ki = k[..., 1].float()

        rotr = torch.cos(freqs * ts)
        roti = torch.sin(freqs * ts)

        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr
        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        dtype = q.dtype
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

        return qo.view(B, T, H, D), ko.view(B, Tk, Hk, D)
