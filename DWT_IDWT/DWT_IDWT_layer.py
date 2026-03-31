import torch
from torch import nn


class DWT_2D(nn.Module):
    """
    Lightweight 2D Haar DWT compatible with:
      xll, xlh, xhl, xhh = dwt(x)
    Input/outputs are NCHW tensors.
    """

    def __init__(self, wavename="haar"):
        super().__init__()
        if wavename != "haar":
            raise ValueError("This local DWT_2D supports only wavename='haar'.")

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW tensor, got shape {tuple(x.shape)}")
        if (x.shape[-2] % 2) != 0 or (x.shape[-1] % 2) != 0:
            raise ValueError("Input H/W must be even for Haar DWT.")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return ll, lh, hl, hh


class IDWT_2D(nn.Module):
    """
    Lightweight 2D Haar IDWT compatible with:
      x = idwt(xll, xlh, xhl, xhh)
    Input/outputs are NCHW tensors.
    """

    def __init__(self, wavename="haar"):
        super().__init__()
        if wavename != "haar":
            raise ValueError("This local IDWT_2D supports only wavename='haar'.")

    def forward(self, ll, lh, hl, hh):
        for t in (ll, lh, hl, hh):
            if t.dim() != 4:
                raise ValueError(f"Expected NCHW tensor, got shape {tuple(t.shape)}")
        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError("LL/LH/HL/HH must have identical shapes.")

        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        n, c_ch, h, w = ll.shape
        out = torch.empty((n, c_ch, h * 2, w * 2), dtype=ll.dtype, device=ll.device)
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d
        return out
