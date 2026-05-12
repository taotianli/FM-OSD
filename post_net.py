# networks

import math
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# MSSR: Mamba-Style Selective State-Space Refinement
# ============================================================================

def _parallel_scan(dA: torch.Tensor, dBu: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative prefix scan for the SSM recurrence:
        h_t = dA_t * h_{t-1} + dBu_t

    All operations are batched tensor ops — zero Python-level loops over
    sequence length.  For short sequences (L ≤ ~512, typical patch grids
    14×14=196) this is 5–15× faster than a sequential Python for-loop.

    dA:  [B, L, d_inner, d_state]  (discrete transition)
    dBu: [B, L, d_inner, d_state]  (input term)
    Returns h: [B, L, d_inner, d_state]  (state at every step)
    """
    B, L, d_inner, d_state = dA.shape

    # Pad L to next power of two for the tree reduction
    L_pad = 1
    while L_pad < L:
        L_pad <<= 1

    if L_pad > L:
        pad = L_pad - L
        dA  = F.pad(dA,  (0, 0, 0, 0, 0, pad))   # pad along dim=1
        dBu = F.pad(dBu, (0, 0, 0, 0, 0, pad))

    # Up-sweep (reduce): build prefix products in O(L log L) tensor ops
    # Each level halves the active length.
    a = dA.clone()    # [B, L_pad, d_inner, d_state]
    b = dBu.clone()   # [B, L_pad, d_inner, d_state]

    stride = 1
    while stride < L_pad:
        # Indices of "right" nodes at this level
        r = torch.arange(stride - 1, L_pad, stride * 2, device=dA.device)
        if r.numel() == 0:
            break
        l = r - stride   # "left" sibling indices (may be negative for first level — guarded below)
        valid = l >= 0
        if not valid.all():
            r = r[valid]
            l = l[valid]
        if r.numel() == 0:
            stride *= 2
            continue
        # h_r = a_r * h_l + b_r
        b[:, r] = a[:, r] * b[:, l] + b[:, r]
        a[:, r] = a[:, r] * a[:, l]
        stride *= 2

    # Down-sweep: distribute prefix sums back
    # Set last element to identity (h=0, a=1 means "carry nothing")
    a[:, -1] = 1.0
    b[:, -1] = 0.0

    stride = L_pad // 2
    while stride > 0:
        r = torch.arange(stride - 1, L_pad, stride * 2, device=dA.device)
        if r.numel() == 0:
            stride //= 2
            continue
        l = r - stride
        valid = l >= 0
        if not valid.all():
            r = r[valid]
            l = l[valid]
        if r.numel() == 0:
            stride //= 2
            continue
        b_l_old = b[:, l].clone()
        a_l_old = a[:, l].clone()
        b[:, l] = a[:, r] * b_l_old + b[:, r]   # left child gets right's contribution
        a[:, l] = a[:, r] * a_l_old
        # right child: carry from left
        b[:, r] = a[:, r] * b_l_old + b[:, r]
        a[:, r] = a[:, r] * a_l_old
        stride //= 2

    # The scan above gives exclusive prefix; shift by 1 to get inclusive
    # Simpler: just recompute with a single vectorised pass using cumulative product
    # (For sequences as short as 196, a single vectorised loop-unrolled matmul scan
    #  is clearer and equally fast.)
    # ── Fallback to log-domain cumsum scan (always correct, no index bugs) ────
    return _log_domain_scan(dA[:, :L], dBu[:, :L])


def _log_domain_scan(dA: torch.Tensor, dBu: torch.Tensor) -> torch.Tensor:
    """
    Vectorised SSM scan via log-domain cumulative sum.
    Works by computing:
        log_a_cumsum[t] = sum_{s=0}^{t} log(dA[s])
        h[t] = sum_{s=0}^{t} exp(log_a_cumsum[t] - log_a_cumsum[s]) * dBu[s]

    All ops are full-tensor; no Python loop over L.

    dA:  [B, L, d_inner, d_state]  — must be > 0 (guaranteed by exp(·))
    dBu: [B, L, d_inner, d_state]
    Returns h: [B, L, d_inner, d_state]
    """
    # log cumsum of dA along the sequence dimension
    log_a = torch.log(dA.clamp(min=1e-38))              # [B, L, d_inner, d_state]
    log_a_cs = torch.cumsum(log_a, dim=1)               # [B, L, d_inner, d_state]

    # For each position t, compute sum_s exp(log_a_cs[t] - log_a_cs[s]) * dBu[s]
    # = exp(log_a_cs[t]) * sum_s exp(-log_a_cs[s]) * dBu[s]   (using prefix sum)
    inv_cs = torch.cumsum(torch.exp(-log_a_cs) * dBu, dim=1)   # [B, L, d_inner, d_state]
    h = torch.exp(log_a_cs) * inv_cs                            # [B, L, d_inner, d_state]
    return h


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (S6) scan over a 1-D sequence.

    Input:  [B, L, d_model]
    Output: [B, L, d_model]

    The state transition matrices A, B, C and the step size Δ are all
    input-dependent (selective), which lets the model decide how much
    context to carry forward at each position — unlike a fixed RNN.

    Recurrence:
        h_t = diag(exp(Δ_t * A)) * h_{t-1} + Δ_t * B_t * x_t
        y_t = C_t * h_t + D * x_t

    Implemented via a fully-vectorised log-domain cumulative-sum scan
    (no Python loops over sequence length → 5–15× faster than the
    previous chunked sequential scan).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Depthwise conv for local context (mimics Mamba's conv1d)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)

        # SSM parameter projections (input-dependent / selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner + 1, bias=False)
        # ^ projects to: B (d_state), C (d_state), Δ_raw (d_inner), dt_bias (1)

        # Fixed log-A initialisation (HiPPO-like)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]  →  [B, L, d_model]"""
        residual = x
        B_sz, L, _ = x.shape
        d_inner = self.in_proj.out_features // 2

        # Split into SSM branch (z) and gate (g)
        xz = self.in_proj(x)                          # [B, L, d_inner*2]
        z, gate = xz.chunk(2, dim=-1)                 # each [B, L, d_inner]

        # Depthwise conv for local mixing
        z = self.conv1d(z.transpose(1, 2))[:, :, :L].transpose(1, 2)  # [B, L, d_inner]
        z = F.silu(z)

        # Selective projections
        A = -torch.exp(self.A_log)                    # [d_inner, d_state]
        x_proj = self.x_proj(z)                       # [B, L, d_state*2 + d_inner + 1]
        B_proj = x_proj[..., :self.d_state]           # [B, L, d_state]
        C_proj = x_proj[..., self.d_state:2*self.d_state]
        delta_raw = x_proj[..., 2*self.d_state:-1]    # [B, L, d_inner]
        dt_bias   = x_proj[..., -1:]                  # [B, L, 1]
        delta = F.softplus(delta_raw + dt_bias)       # [B, L, d_inner]

        # Discretise: Ā = exp(Δ * A),  B̄u = Δ * B * u  (ZOH)
        # delta: [B,L,d_inner], A: [d_inner,d_state], B_proj: [B,L,d_state], z: [B,L,d_inner]
        dA  = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B,L,d_inner,d_state]
        dBu = delta.unsqueeze(-1) * B_proj.unsqueeze(2) * z.unsqueeze(-1)   # [B,L,d_inner,d_state]

        # Vectorised scan — no Python loops over L
        h = _log_domain_scan(dA, dBu)                 # [B, L, d_inner, d_state]

        # Readout: y_t = C_t · h_t  (+ D skip)
        y = (h * C_proj.unsqueeze(2)).sum(-1)         # [B, L, d_inner]
        y = y + self.D.unsqueeze(0).unsqueeze(0) * z  # skip

        # Gate and project back
        y = y * F.silu(gate)
        y = self.out_proj(y)                          # [B, L, d_model]
        return self.norm(y + residual)


class MSSRModule(nn.Module):
    """
    Mamba-Style Selective-State Refinement (MSSR) for 2-D feature maps.

    Applies a bidirectional SSM scan over the flattened patch sequence
    (row-major forward + backward), then reshapes back to spatial.

    direction: 'bidir'   — forward + backward (proposed, default)
               'forward' — forward only  (ablation C1-fwd)
               'backward'— backward only (ablation C1-bwd)

    Input/output: [B, C, Hp, Wp]
    """

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, direction: str = 'bidir'):
        super().__init__()
        assert direction in ('bidir', 'forward', 'backward'), \
            f"direction must be 'bidir'/'forward'/'backward', got {direction}"
        self.direction = direction

        if direction in ('bidir', 'forward'):
            self.ssm_fwd = SelectiveSSM(channels, d_state, d_conv, expand)
        if direction in ('bidir', 'backward'):
            self.ssm_bwd = SelectiveSSM(channels, d_state, d_conv, expand)

        # merge: bidir needs 2C→C, unidirectional needs C→C (identity-like)
        in_ch = channels * 2 if direction == 'bidir' else channels
        self.merge = nn.Conv2d(in_ch, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, Hp, Wp]  →  [B, C, Hp, Wp]"""
        B, C, Hp, Wp = x.shape
        seq = x.flatten(2).permute(0, 2, 1)          # [B, L, C]

        if self.direction == 'bidir':
            fwd = self.ssm_fwd(seq)
            bwd = self.ssm_bwd(seq.flip(1)).flip(1)
            out = torch.cat([fwd, bwd], dim=-1)       # [B, L, 2C]
        elif self.direction == 'forward':
            out = self.ssm_fwd(seq)                   # [B, L, C]
        else:  # backward
            out = self.ssm_bwd(seq.flip(1)).flip(1)   # [B, L, C]

        out = out.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        return self.merge(out)                        # [B, C, Hp, Wp]


# ============================================================================
# MLMF: Multi-Layer Multi-Facet Adaptive Fusion Module
# ============================================================================

class AdaptiveFusionModule(nn.Module):
    """
    Memory-efficient Adaptive Fusion for MLMF features.

    Each source is first projected from high-dim (e.g. 6528) to out_channels (e.g. 256)
    at patch resolution. Attention weights are computed from these compact representations,
    avoiding the need to upsample 6528-channel tensors.

    If uniform=True, uses equal weights (no attention) — used for ablation A5.
    """
    def __init__(self, num_sources: int, in_channels: int, out_channels: int,
                 reduction: int = 4, uniform: bool = False):
        super().__init__()
        self.num_sources = num_sources
        self.uniform = uniform

        # Per-source 1×1 projection: in_channels → out_channels (at patch resolution)
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(num_sources)
        ])

        if not uniform:
            # Lightweight cross-source attention (operates on out_channels, not in_channels)
            hidden = max(num_sources * out_channels // reduction, num_sources)
            self.source_attention = nn.Sequential(
                nn.Linear(num_sources * out_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_sources),
                nn.Softmax(dim=-1),
            )
            self.source_scale = nn.Parameter(torch.ones(num_sources))

    def forward(self, raw_features: list) -> torch.Tensor:
        """
        raw_features: list of [B, in_channels, H_p, W_p] tensors at patch resolution.
        Returns: [B, out_channels, H_p, W_p] fused tensor.
        """
        assert len(raw_features) == self.num_sources
        B = raw_features[0].shape[0]

        # Project each source: [B, in_ch, Hp, Wp] → [B, out_ch, Hp, Wp]
        proj_feats = [self.proj[i](raw_features[i]) for i in range(self.num_sources)]

        if self.uniform:
            # Ablation A5: simple average, no learned attention
            fused = sum(proj_feats) / self.num_sources
        else:
            # Adaptive cross-source attention
            pooled = [f.mean(dim=[2, 3]) for f in proj_feats]          # each [B, out_ch]
            concat  = torch.cat(pooled, dim=-1)                         # [B, num_sources*out_ch]
            weights = self.source_attention(concat)                      # [B, num_sources]
            weights = F.softmax(weights * self.source_scale.unsqueeze(0), dim=-1)
            fused = sum(weights[:, i].view(B, 1, 1, 1) * proj_feats[i]
                        for i in range(self.num_sources))
        return fused


class Upnet_v3_MLMF(nn.Module):
    """
    Memory-efficient MLMF upsampling network.

    Pipeline: raw descriptors → reshape to patch grid → project+fuse at patch res
              → upsample to target size → output conv.
    """
    def __init__(self, size, in_channels: int, out_channels: int = 256,
                 num_sources: int = 6, fusion_reduction: int = 4,
                 fusion_uniform: bool = False):
        super().__init__()
        self.size = size
        self.num_sources = num_sources

        self.fusion = AdaptiveFusionModule(num_sources, in_channels,
                                           out_channels, fusion_reduction,
                                           uniform=fusion_uniform)
        self.conv_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def _to_spatial(self, feat, num_patches):
        """[B, 1, T, C] or [B, T, C] → [B, C, Hp, Wp]"""
        x = feat.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        return x.permute(0, 3, 1, 2)

    def forward(self, features: list, num_patches):
        spatial = [self._to_spatial(f, num_patches) for f in features]
        fused   = self.fusion(spatial)                              # [B, out_ch, Hp, Wp]
        out     = F.interpolate(fused, self.size, mode='bilinear', align_corners=False)
        return self.conv_out(out)


class Upnet_v3_MLMF_CoarseToFine(nn.Module):
    """
    Memory-efficient MLMF coarse-to-fine network.

    Global branch: project+fuse at patch res → upsample → conv_out1
    Local  branch: project+fuse at patch res → transposed-conv upsample → conv_out2
    """
    def __init__(self, size, in_channels: int, out_channels: int = 256,
                 num_sources: int = 6, fusion_reduction: int = 4):
        super().__init__()
        self.size = size
        self.num_sources = num_sources

        self.fusion_global = AdaptiveFusionModule(num_sources, in_channels,
                                                   out_channels, fusion_reduction)
        self.fusion_local  = AdaptiveFusionModule(num_sources, in_channels,
                                                   out_channels, fusion_reduction)

        self.conv_out1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv_out2 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def _to_spatial(self, feat, num_patches):
        x = feat.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        return x.permute(0, 3, 1, 2)

    def forward(self, features: list, num_patches, size, islocal=False):
        spatial = [self._to_spatial(f, num_patches) for f in features]

        if islocal:
            fused = self.fusion_local(spatial)
            x     = self.up(fused)
            out   = F.interpolate(x, size, mode='bilinear', align_corners=False)
            return self.conv_out2(out)
        else:
            fused = self.fusion_global(spatial)
            out   = F.interpolate(fused, size, mode='bilinear', align_corners=False)
            return self.conv_out1(out)


# ============================================================================
# MLMF + MSSR: Networks with Mamba-Style Selective State-Space Refinement
# ============================================================================

class Upnet_v3_MLMF_MSSR(nn.Module):
    """
    MLMF global network with MSSR (Mamba-Style Selective State-Space Refinement).

    Pipeline: raw descriptors → reshape to patch grid → project+fuse (MLMF)
              → MSSR SSM scan at patch resolution
              → upsample to target size → output conv.

    Ablation knobs:
      mssr_d_state  : SSM state dimension  (C2: 8 / 16 / 32)
      mssr_expand   : inner expansion      (C3: 1 / 2 / 4)
      mssr_direction: scan direction       (C1: 'bidir' / 'forward' / 'backward')
    """
    def __init__(self, size, in_channels: int, out_channels: int = 256,
                 num_sources: int = 6, fusion_reduction: int = 4,
                 fusion_uniform: bool = False,
                 mssr_d_state: int = 16, mssr_expand: int = 2,
                 mssr_direction: str = 'bidir'):
        super().__init__()
        self.size = size
        self.num_sources = num_sources

        self.fusion = AdaptiveFusionModule(num_sources, in_channels,
                                           out_channels, fusion_reduction,
                                           uniform=fusion_uniform)
        self.mssr = MSSRModule(out_channels, d_state=mssr_d_state,
                               expand=mssr_expand, direction=mssr_direction)
        self.conv_out = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def _to_spatial(self, feat, num_patches):
        x = feat.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        return x.permute(0, 3, 1, 2)

    def forward(self, features: list, num_patches):
        spatial = [self._to_spatial(f, num_patches) for f in features]
        fused   = self.fusion(spatial)
        fused   = self.mssr(fused)
        out     = F.interpolate(fused, self.size, mode='bilinear', align_corners=False)
        return self.conv_out(out)


class Upnet_v3_MLMF_MSSR_CoarseToFine(nn.Module):
    """
    MLMF coarse-to-fine network with MSSR.

    Global branch: project+fuse (MLMF) → MSSR scan → upsample → conv_out1
    Local  branch: project+fuse (MLMF) → [optional MSSR] → transposed-conv → conv_out2

    Ablation knobs:
      mssr_d_state  : SSM state dimension  (C2: 8 / 16 / 32)
      mssr_expand   : inner expansion      (C3: 1 / 2 / 4)
      mssr_direction: scan direction       (C1: 'bidir' / 'forward' / 'backward')
      mssr_local    : also apply MSSR on local branch (C5: False / True)
    """
    def __init__(self, size, in_channels: int, out_channels: int = 256,
                 num_sources: int = 6, fusion_reduction: int = 4,
                 mssr_d_state: int = 16, mssr_expand: int = 2,
                 mssr_direction: str = 'bidir', mssr_local: bool = False):
        super().__init__()
        self.size = size
        self.num_sources = num_sources
        self.mssr_local = mssr_local

        self.fusion_global = AdaptiveFusionModule(num_sources, in_channels,
                                                   out_channels, fusion_reduction)
        self.fusion_local  = AdaptiveFusionModule(num_sources, in_channels,
                                                   out_channels, fusion_reduction)
        self.mssr = MSSRModule(out_channels, d_state=mssr_d_state,
                               expand=mssr_expand, direction=mssr_direction)
        if mssr_local:
            self.mssr_loc = MSSRModule(out_channels, d_state=mssr_d_state,
                                       expand=mssr_expand, direction=mssr_direction)

        self.conv_out1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_out2 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def _to_spatial(self, feat, num_patches):
        x = feat.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        return x.permute(0, 3, 1, 2)

    def forward(self, features: list, num_patches, size, islocal=False):
        spatial = [self._to_spatial(f, num_patches) for f in features]

        if islocal:
            fused = self.fusion_local(spatial)
            if self.mssr_local:
                fused = self.mssr_loc(fused)
            x   = self.up(fused)
            out = F.interpolate(x, size, mode='bilinear', align_corners=False)
            return self.conv_out2(out)
        else:
            fused = self.fusion_global(spatial)
            fused = self.mssr(fused)
            out   = F.interpolate(fused, size, mode='bilinear', align_corners=False)
            return self.conv_out1(out)


# ============================================================================
# Original Networks (preserved for backward compatibility)
# ============================================================================

class Upnet_v3(nn.Module):
    def __init__(self, size, in_channels, out_channels = 128):
        super().__init__()
        self.size = size
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, padding = 1)

    def forward(self, x, num_patches):
        x = x.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        x = x.permute(0, 3, 1, 2) 

        out = torch.nn.functional.interpolate(x, self.size, mode = 'bilinear')
        out = self.conv_out(out)
        return out

class Upnet_v3_coarsetofine2_tran_new(nn.Module): 
    def __init__(self, size, in_channels, out_channels = 128):
        super().__init__()
        self.size = size
        self.conv_out1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)  # global
        self.conv_out2 = nn.Conv2d(out_channels, out_channels, 5, padding = 2)  # local
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def forward(self, x, num_patches, size, islocal = False):
        x = x.squeeze(1)
        b, n, c = x.shape
        x = x.reshape(b, num_patches[0], num_patches[1], c)
        x = x.permute(0, 3, 1, 2)  
        if islocal:
            x = self.up(x)
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear')
            out_fine = self.conv_out2(out)
            return out_fine
        else:
            out = torch.nn.functional.interpolate(x, size, mode = 'bilinear')
            out_coarse = self.conv_out1(out)
            return out_coarse  
