"""B1-style encoder (spatial-softmax pooling by default) + diffusion-policy head.

This is the `baseline` method: B1's view-conditioned encoder (no residual token
branch) with the original DP's input conventions, attached to the DP UNet.
``BaselineEncoder(pool="avgpool")`` is the v1-parity ablation: everything else is
identical, but each view is reduced by global average pooling instead of
spatial-softmax keypoints.

    image (N, K, 9, 256, 256)   # rgb/255 + raw Plucker, no ImageNet norm
      -> shared ResNet18 (no pretrained weights, GroupNorm)
      -> per-stage FiLM from the 4-step camera-frame EEF history
      -> per-view spatial-softmax keypoints (32 x 2) -> Linear -> 512
      -> masked cross-view MHA with a learnable query -> 512
      -> concat 7-D joint-qpos proprio -> 519

The policy uses a single observation frame (DP ``n_obs_steps=1``), so the UNet
is conditioned on 519 dims.  The proprio is zeroed before normalisation with
``prob_drop_proprio`` (default 1.0), matching the original DP; the 44-D EEF
history is always passed to FiLM.  Inference returns the 8 actions for
``n_action_steps`` (the ``start=n_obs_steps-1=0`` slice), so CamPose's evaluator
executes exactly 8 actions and replans.

Design note: keep the per-view latent spatial *before* fusion.  Global average
pooling is translation-invariant and discards the object-localisation signal;
the policy then collapses to a state-only controller on the *Rand tasks.

Keep this file free of robosuite imports; the CPU tests instantiate it without
a simulator.  ``diffusers`` is imported lazily inside ``BaselineDiffusionPolicy``
so the encoder can be tested on its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from baseline_common import (
    EEF_HIST_DIM,
    EEF_HIST_STEPS,
    FUSED_DIM,
    OBS_FEATURE_DIM,
    PROPRIO_DIM,
    SpatialSoftmax,
    replace_bn_with_gn,
)


class BaselineEncoder(nn.Module):
    """B1-style encoder: one latent per view, then a masked cross-view query.

    Per-view latents keep their spatial layout via spatial-softmax keypoints by
    default.  ``pool="avgpool"`` is the v1-parity ablation and deliberately
    discards that layout.
    """

    def __init__(
        self,
        in_channels: int = 9,
        eef_hist_steps: int = EEF_HIST_STEPS,
        fused_dim: int = FUSED_DIM,
        n_heads: int = 8,
        cond_dim: int = 128,
        use_group_norm: bool = True,
        pool: str = "spatial_softmax",
    ) -> None:
        super().__init__()
        if in_channels != 9:
            raise ValueError(
                f"baseline expects 9 channels (RGB 3 + Plucker 6), got {in_channels}"
            )
        if pool not in ("spatial_softmax", "avgpool"):
            raise ValueError(
                "pool must be 'spatial_softmax' or 'avgpool', "
                f"got {pool!r}"
            )
        self.in_channels = in_channels
        self.eef_hist_dim = EEF_HIST_DIM
        self.fused_dim = fused_dim
        self.pool = pool
        if eef_hist_steps * 11 != self.eef_hist_dim:
            raise ValueError(
                f"eef_hist_steps={eef_hist_steps} gives {eef_hist_steps * 11} dims, "
                f"expected {self.eef_hist_dim}"
            )

        resnet = torchvision.models.resnet18(weights=None)

        # B1-style initialisation: widen conv1 to 9 channels, copy RGB weights
        # into channels 0:3, zero the 6 new Plucker channels, copy the bias.
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            new_conv.weight[:, 3:].zero_()
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        resnet.conv1 = new_conv
        if use_group_norm:
            replace_bn_with_gn(resnet)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.pool == "spatial_softmax":
            # Keep layer4's spatial grid: global average pooling destroys the
            # localisation signal needed to act on the scene.
            self.spatial_pool = SpatialSoftmax((fused_dim, 8, 8), num_kp=32)
            self.spatial_proj = nn.Linear(2 * 32, fused_dim)
        else:
            # v1-parity ablation: one global feature vector per view.
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FiLM at stem, layer1, layer2, layer3, layer4.
        self.film_widths = (64, 64, 128, 256, 512)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.eef_hist_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film = nn.ModuleList(
            [nn.Linear(cond_dim, 2 * c) for c in self.film_widths]
        )
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.fusion = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=n_heads, batch_first=True
        )
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, fused_dim))

    @staticmethod
    def _film(h, film_layer, cond):
        gamma, beta = film_layer(cond).chunk(2, dim=-1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def _backbone_map(self, x, cond):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self._film(h, self.film[0], cond)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self._film(h, self.film[1], cond)
        h = self.layer2(h)
        h = self._film(h, self.film[2], cond)
        h = self.layer3(h)
        h = self._film(h, self.film[3], cond)
        h = self.layer4(h)
        h = self._film(h, self.film[4], cond)
        return h

    def _prepare_image(self, image):
        if image.ndim != 5:
            raise ValueError(
                f"expected image (N, K, C, H, W), got {tuple(image.shape)}"
            )
        if image.shape[2] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {image.shape[2]}"
            )
        # RGB is rgb/255 and Plucker is raw: the original DP applies no
        # ImageNet normalisation, so the validated input is used unchanged.
        return image

    def forward_full(self, image, proprio, eef_hist, view_mask):
        """Return ``z_global`` and per-view diagnostics.

        Args:
            image: (N, K, 9, H, W), rgb/255 + raw Plucker
            proprio: (N, 7), normalised joint qpos (constant when dropped)
            eef_hist: (N, K, 44), z-scored camera-frame history
            view_mask: (N, K) bool/float; inactive views are ignored
        """
        n, k = image.shape[0], image.shape[1]
        if eef_hist.shape[:2] != (n, k):
            raise ValueError(
                f"eef_hist shape {tuple(eef_hist.shape)} does not match image "
                f"{(n, k, '...')}"
            )
        if proprio.shape != (n, PROPRIO_DIM):
            raise ValueError(
                f"proprio must be (N, {PROPRIO_DIM})=({n}, {PROPRIO_DIM}), "
                f"got {tuple(proprio.shape)}"
            )

        active = view_mask > 0.5
        if active.shape != (n, k):
            raise ValueError(
                f"view_mask must be (N, K)=({n}, {k}), got {tuple(view_mask.shape)}"
            )
        if not bool(active.any(dim=1).all()):
            raise ValueError("every sample needs at least one active view")

        x = self._prepare_image(image)
        rows, cols = active.nonzero(as_tuple=True)
        x_active = x[rows, cols]
        cond = self.cond_mlp(eef_hist[rows, cols])

        fmap = self._backbone_map(x_active, cond)
        if tuple(fmap.shape[-2:]) != (8, 8):
            raise ValueError(
                f"expected an 8x8 layer4 grid at 256x256, got "
                f"{tuple(fmap.shape[-2:])}"
            )
        if self.pool == "spatial_softmax":
            z_base = self.spatial_proj(self.spatial_pool(fmap).flatten(1))
        else:
            z_base = self.avgpool(fmap).flatten(1)

        tokens = z_base.new_zeros((n, k, self.fused_dim))
        tokens[rows, cols] = z_base
        query = self.fusion_query.expand(n, 1, self.fused_dim)
        z_global = self.fusion(
            query, tokens, tokens, key_padding_mask=~active, need_weights=False
        )[0][:, 0]

        out = torch.cat([z_global, proprio], dim=-1)
        return {
            "z_global": z_global,
            "z_views": tokens,
            "view_active": active,
            "out": out,
        }

    def forward(self, image, proprio, eef_hist, view_mask):
        return self.forward_full(image, proprio, eef_hist, view_mask)["out"]


class BaselineDiffusionPolicy(nn.Module):
    """DP head conditioned on B1's 519-D per-frame observation feature."""

    def __init__(
        self,
        args,
        horizon: int = 16,
        n_obs_steps: int = 1,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        down_dims=(512, 1024, 2048),
        diffusion_step_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        # Lazy imports keep the encoder importable without diffusers.
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        from .dp import ConditionalUnet1D

        self.action_dim = int(args.action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.obs_feature_dim = OBS_FEATURE_DIM
        self.global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        self.lr = float(getattr(args, "lr", 1e-4))
        self.weight_decay = float(getattr(args, "weight_decay", 1e-6))

        self.baseline_pool = getattr(args, "baseline_pool", "spatial_softmax")
        self.encoder = BaselineEncoder(pool=self.baseline_pool)
        self.unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(down_dims),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        # Same schedule as the B1 reference and CamPose DP.
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

    def _build_conds(self, data_dict):
        image = data_dict["image"]
        if image.ndim != 6:
            raise ValueError(
                "baseline expects image (B, To, K, 9, H, W); "
                f"got {tuple(image.shape)}"
            )
        b, to, k = image.shape[:3]
        if to != self.n_obs_steps:
            raise ValueError(
                f"expected {self.n_obs_steps} observation frames, got {to}"
            )
        proprio = data_dict["proprio"]
        eef_hist = data_dict["eef_hist"]
        view_mask = data_dict["view_mask"]
        if proprio.shape[:2] != (b, to):
            raise ValueError(
                f"proprio must be (B, To, 7)=({b}, {to}, 7), "
                f"got {tuple(proprio.shape)}"
            )

        image = image.reshape(b * to, k, *image.shape[3:])
        proprio = proprio.reshape(b * to, PROPRIO_DIM)
        eef_hist = eef_hist.reshape(b * to, k, -1)
        view_mask = view_mask.reshape(b * to, k)

        z = self.encoder(image, proprio, eef_hist, view_mask)
        if z.shape != (b * to, self.obs_feature_dim):
            raise ValueError(
                f"encoder returned {tuple(z.shape)}, expected "
                f"({b * to}, {self.obs_feature_dim})"
            )
        global_cond = z.reshape(b, -1)
        return None, global_cond

    @torch.no_grad()
    def _sample(self, batch_size, global_cond, generator=None):
        device = global_cond.device
        sample = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_vec = torch.full(
                (batch_size,), int(t), dtype=torch.long, device=device
            )
            model_output = self.unet(sample, t_vec, global_cond=global_cond)
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample
        return sample

    def __call__(self, data_dict, generator=None):
        _, global_cond = self._build_conds(data_dict)
        actions = data_dict.get("actions")

        if actions is not None:
            b = actions.shape[0]
            actions = actions[:, : self.horizon]
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (b,),
                device=actions.device,
            ).long()
            noisy = self.noise_scheduler.add_noise(actions, noise, timesteps)
            pred = self.unet(noisy, timesteps, global_cond=global_cond)
            loss = F.mse_loss(pred, noise, reduction="none")
            is_pad = data_dict.get("is_pad")
            if is_pad is not None:
                is_pad = is_pad[:, : self.horizon].to(actions.device).bool()
                valid = (~is_pad).unsqueeze(-1).expand_as(loss).float()
                loss = (loss * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss = loss.mean()
            return {"loss": loss}

        batch_size = global_cond.shape[0]
        sample = self._sample(batch_size, global_cond, generator=generator)
        start = self.n_obs_steps - 1
        return sample[:, start : start + self.n_action_steps]
