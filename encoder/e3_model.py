"""E3: transformer artist encoder over KG concept tokens with KG-biased attention and Laplacian regularisation."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class BiasedLayer(nn.Module):
    def __init__(s, d, h, use_bias):
        super().__init__(); s.h, s.dh, s.use_bias = h, d // h, use_bias
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.alpha = nn.Parameter(torch.full((h,), 1.0))           # per-head strength of the KG bias
    def forward(s, x, pad, kg):
        B, L, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, s.h, s.dh).permute(2, 0, 3, 1, 4)
        att = q @ k.transpose(-1, -2) / math.sqrt(s.dh)
        if s.use_bias: att = att + s.alpha.view(1, -1, 1, 1) * kg[:, None]
        att = att.masked_fill(pad[:, None, None, :], -1e4).softmax(-1)
        x = x + s.o((att @ v).transpose(1, 2).reshape(B, L, D))
        return x + s.ff(s.n2(x)), att

class ArtistEncoder(nn.Module):
    def __init__(s, n_tok, d=128, h=4, layers=2, use_bias=True, init=None):
        super().__init__()
        s.tok = nn.Embedding(n_tok, d, padding_idx=0); s.typ = nn.Embedding(4, d); s.wt = nn.Linear(1, d)
        nn.init.normal_(s.tok.weight, std=0.1)
        if init is not None:
            with torch.no_grad(): s.tok.weight[1:] = init
        s.layers = nn.ModuleList([BiasedLayer(d, h, use_bias) for _ in range(layers)])
        s.norm, s.proj = nn.LayerNorm(d), nn.Linear(d, d)
    def forward(s, seq, typ, wts, kg, return_att=False):
        pad = seq == 0
        x = s.tok(seq) + s.typ(typ) + s.wt(wts[..., None])
        atts = []
        for l in s.layers: x, a = l(x, pad, kg); atts.append(a)
        x = s.norm(x); w = (wts + 1e-3) * (~pad)
        z = s.proj((x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6))
        z = F.normalize(z, dim=-1)
        return (z, atts) if return_att else z

def laplacian_penalty(emb, edges_i, edges_j, edges_w, sample=None):
    """Tr(Z^T L Z) = 1/2 sum_ij A_ij ||z_i - z_j||^2, averaged over (sampled) edges."""
    if sample is not None and sample < len(edges_i):
        k = torch.randint(len(edges_i), (sample,)); edges_i, edges_j, edges_w = edges_i[k], edges_j[k], edges_w[k]
    d = emb[edges_i] - emb[edges_j]
    return (edges_w * (d * d).sum(1)).mean()
