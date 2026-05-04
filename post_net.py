# networks

import math
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# MLMF: Multi-Layer Multi-Facet Adaptive Fusion Module
# ============================================================================

class AdaptiveFusionModule(nn.Module):
    """
    Memory-efficient Adaptive Fusion for MLMF features.

    Each source is first projected from high-dim (e.g. 6528) to out_channels (e.g. 256)
    at patch resolution. Attention weights are computed from these compact representations,
    avoiding the need to upsample 6528-channel tensors.
    """
    def __init__(self, num_sources: int, in_channels: int, out_channels: int,
                 reduction: int = 4):
        super().__init__()
        self.num_sources = num_sources

        # Per-source 1×1 projection: 6528 → out_channels (at patch resolution)
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(num_sources)
        ])

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

        # Global pool each projected source for attention
        pooled = [f.mean(dim=[2, 3]) for f in proj_feats]          # each [B, out_ch]
        concat  = torch.cat(pooled, dim=-1)                         # [B, num_sources*out_ch]
        weights = self.source_attention(concat)                      # [B, num_sources]
        weights = F.softmax(weights * self.source_scale.unsqueeze(0), dim=-1)

        # Weighted sum
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
                 num_sources: int = 6, fusion_reduction: int = 4):
        super().__init__()
        self.size = size
        self.num_sources = num_sources

        self.fusion = AdaptiveFusionModule(num_sources, in_channels,
                                           out_channels, fusion_reduction)
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
