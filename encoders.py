"""Modality-specific retinal encoders with anatomy-aligned shared/private token heads.

Fundus encoder: 3x64x64 -> 4x4 macular cell tokens + 1 global token.
OCT encoder: 8x32x32 volume -> 4x4 macular cell tokens + 1 global token (same anatomical grid).
Each token = [shared (D_S) | private (D_P)].
"""
import torch, torch.nn as nn, torch.nn.functional as Fn

D_S, D_P = 32, 32
D_TOK = D_S + D_P
MAC_ROWS, MAC_COLS = slice(5, 11), slice(4, 10)   # macular square on the stride-4 fundus map


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())


class FundusEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(cbr(3, 32), cbr(32, 48, 2), cbr(48, 64, 2))      # 16x16
        self.deep = nn.Sequential(cbr(64, 96, 2), cbr(96, 96))                     # 8x8
        self.cell = nn.Conv2d(64, 64, 1)
        self.glob = nn.Linear(96, 64)
        self.sh = nn.Linear(64, D_S); self.pr = nn.Linear(64, D_P)
        self.dec = nn.Sequential(nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, x, decode=False):
        f16 = self.stem(x); f8 = self.deep(f16)
        c = Fn.adaptive_avg_pool2d(self.cell(f16[:, :, MAC_ROWS, MAC_COLS]), 4)       # B,64,4,4
        c = c.flatten(2).transpose(1, 2)                                             # B,16,64
        g = self.glob(f8.mean((2, 3)))[:, None]                                      # B,1,64
        t = torch.cat([c, g], 1)
        tok = torch.cat([self.sh(t), self.pr(t)], -1)                                # B,17,64
        return (tok, self.dec(f8)) if decode else tok


class OCTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scan = nn.Sequential(cbr(1, 24), cbr(24, 48, 2), cbr(48, 64, 2))     # per B-scan 8x8 (depth x width)
        self.depth = nn.Linear(64 * 8, 64)                                         # keeps layer positions
        self.mix = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.glob = nn.Linear(64, 64)
        self.sh = nn.Linear(64, D_S); self.pr = nn.Linear(64, D_P)
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def forward(self, v, decode=False):
        B = v.shape[0]
        s = self.scan(v.reshape(B * 8, 1, 32, 32))                                 # B*8,64,8,8
        z = s.permute(0, 3, 1, 2).reshape(B * 8, 8, 64 * 8)                         # width cols, (ch x depth)
        z = self.depth(z).reshape(B, 8, 8, 64).permute(0, 3, 1, 2)                 # B,64,scans,width
        m = self.mix(z)
        c = Fn.adaptive_avg_pool2d(m, 4).flatten(2).transpose(1, 2)                # B,16,64
        g = self.glob(m.mean((2, 3)))[:, None]
        t = torch.cat([c, g], 1)
        tok = torch.cat([self.sh(t), self.pr(t)], -1)
        if decode:
            return tok, self.dec(s).reshape(B, 8, 32, 32)
        return tok


def info_nce(a, b, tau=0.1):
    """Symmetric InfoNCE over anatomical cells: a, b (N,16,D_S); positives = same visit, same cell."""
    a = Fn.normalize(a.reshape(-1, a.shape[-1]), dim=-1); b = Fn.normalize(b.reshape(-1, b.shape[-1]), dim=-1)
    lg = a @ b.t() / tau
    y = torch.arange(lg.shape[0])
    return 0.5 * (Fn.cross_entropy(lg, y) + Fn.cross_entropy(lg.t(), y))


def orth_loss(tok):
    """Decorrelate shared and private halves (batch-standardised cross-correlation)."""
    t = tok.reshape(-1, D_TOK)
    s, p = t[:, :D_S], t[:, D_S:]
    s = (s - s.mean(0)) / (s.std(0) + 1e-5); p = (p - p.mean(0)) / (p.std(0) + 1e-5)
    c = s.t() @ p / t.shape[0]
    return (c ** 2).mean()
