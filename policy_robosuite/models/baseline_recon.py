"""Training-only target-view reconstruction branch (AGENTPLAN.md).

Given the fused global observation feature ``z_g`` and the target camera's
Plücker map ``P_a``, decode a 128x128 RGB image of the target view.  The branch
is deliberately free of ``robosuite`` and ``diffusers`` imports so the CPU tests
can import it without a simulator.

Shapes
------
``z_g``             (B, 512)
``P_a``             (B, 6, 256, 256)   # full-resolution target Plücker map
pyramid             g4 (B,16,128,128), g3 (B,32,64,64),
                    g2 (B,64,32,32),   g1 (B,64,16,16)
decoder output      (B, 3, 128, 128)   # sigmoid, range [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_RECON_MODES = ("z_only", "single_plucker", "multi_plucker")


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    groups = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(inplace=True),
    )


class PluckerPyramidEncoder(nn.Module):
    """Lightweight hierarchical CNN over the target Plücker map.

    The first stride-2 block maps the 256x256 Plücker grid to G4 at 128x128;
    the remaining blocks produce G3/G2/G1 at 64/32/16.
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels=(16, 32, 64, 64),
    ) -> None:
        super().__init__()
        c4, c3, c2, c1 = (int(c) for c in channels)
        self.in_channels = int(in_channels)
        self.stage4 = _conv_block(self.in_channels, c4, stride=2)
        self.stage3 = _conv_block(c4, c3, stride=2)
        self.stage2 = _conv_block(c3, c2, stride=2)
        self.stage1 = _conv_block(c2, c1, stride=2)

    def forward(self, plucker: torch.Tensor) -> dict[str, torch.Tensor]:
        if plucker.ndim != 4 or plucker.shape[1] != self.in_channels:
            raise ValueError(
                f"target_plucker must be (B, {self.in_channels}, H, W), "
                f"got {tuple(plucker.shape)}"
            )
        if tuple(plucker.shape[-2:]) != (256, 256):
            raise ValueError(
                "target_plucker must be a 256x256 map (the PluckerEmbedder "
                f"grid is fixed at 256); got {tuple(plucker.shape[-2:])}"
            )
        g4 = self.stage4(plucker)
        g3 = self.stage3(g4)
        g2 = self.stage2(g3)
        g1 = self.stage1(g2)
        return {"g4": g4, "g3": g3, "g2": g2, "g1": g1}


class TargetViewReconstructor(nn.Module):
    """Decode the target view from ``z_g`` and optional multi-scale Plücker features.

    ``mode`` selects the experiment:
      * ``z_only``          - no Plücker encoder/conditioning (Exp A);
      * ``single_plucker``  - condition only at the coarsest stage (Exp B);
      * ``multi_plucker``   - condition at every decoder stage (Exp C).
    """

    def __init__(
        self,
        fused_dim: int = 512,
        mode: str = "multi_plucker",
        out_size: int = 128,
        base_channels: int = 512,
        base_grid: int = 8,
        decoder_channels=(256, 128, 64, 32),
    ) -> None:
        super().__init__()
        if mode not in _RECON_MODES:
            raise ValueError(
                f"mode must be one of {_RECON_MODES}, got {mode!r}"
            )
        self.mode = mode
        self.fused_dim = int(fused_dim)
        self.out_size = int(out_size)
        self.base_channels = int(base_channels)
        self.base_grid = int(base_grid)
        self.decoder_channels = tuple(int(c) for c in decoder_channels)
        if len(self.decoder_channels) != 4:
            raise ValueError("decoder_channels must have four entries")

        self.fc = nn.Linear(
            self.fused_dim,
            self.base_channels * self.base_grid * self.base_grid,
        )

        self._use_g1 = mode in ("single_plucker", "multi_plucker")
        self._use_g2 = mode == "multi_plucker"
        self._use_g3 = mode == "multi_plucker"
        self._use_g4 = mode == "multi_plucker"
        self.pyramid = (
            PluckerPyramidEncoder() if mode != "z_only" else None
        )

        c1, c2, c3, c4 = self.decoder_channels
        stage1_in = self.base_channels + (64 if self._use_g1 else 0)
        stage2_in = c1 + (64 if self._use_g2 else 0)
        stage3_in = c2 + (32 if self._use_g3 else 0)
        stage4_in = c3 + (16 if self._use_g4 else 0)
        self.stage1 = _conv_block(stage1_in, c1)
        self.stage2 = _conv_block(stage2_in, c2)
        self.stage3 = _conv_block(stage3_in, c3)
        self.stage4 = _conv_block(stage4_in, c4)
        self.out = nn.Conv2d(c4, 3, kernel_size=1)

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        z_global: torch.Tensor,
        target_plucker: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_global.ndim != 2 or z_global.shape[1] != self.fused_dim:
            raise ValueError(
                f"z_global must be (B, {self.fused_dim}), "
                f"got {tuple(z_global.shape)}"
            )
        batch = z_global.shape[0]

        if self.mode != "z_only":
            if target_plucker is None:
                raise ValueError(
                    f"mode={self.mode!r} requires target_plucker"
                )
            if target_plucker.shape[0] != batch:
                raise ValueError(
                    "target_plucker batch does not match z_global: "
                    f"{target_plucker.shape[0]} vs {batch}"
                )
            features = self.pyramid(target_plucker)
        else:
            features = None

        h = self.fc(z_global).reshape(
            batch,
            self.base_channels,
            self.base_grid,
            self.base_grid,
        )

        h = self._upsample(h)
        if self._use_g1:
            h = torch.cat([h, features["g1"].to(h.dtype)], dim=1)
        h = self.stage1(h)

        h = self._upsample(h)
        if self._use_g2:
            h = torch.cat([h, features["g2"].to(h.dtype)], dim=1)
        h = self.stage2(h)

        h = self._upsample(h)
        if self._use_g3:
            h = torch.cat([h, features["g3"].to(h.dtype)], dim=1)
        h = self.stage3(h)

        h = self._upsample(h)
        if self._use_g4:
            h = torch.cat([h, features["g4"].to(h.dtype)], dim=1)
        h = self.stage4(h)

        out = torch.sigmoid(self.out(h))
        if tuple(out.shape[-2:]) != (self.out_size, self.out_size):
            raise RuntimeError(
                f"decoder produced {tuple(out.shape[-2:])}, expected "
                f"{(self.out_size, self.out_size)}"
            )
        return out
