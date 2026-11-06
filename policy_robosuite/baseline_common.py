"""Simulator-free helpers for the `baseline` method.

The `baseline` policy is the exact B1 encoder (variant A, no residual token
branch) attached to a diffusion-policy UNet.  This module holds only the
geometry, indexing, cropping and normalisation helpers shared by the dataset,
the evaluator and the CPU tests.  Nothing here imports robosuite or mani_skill.

Conventions
-----------
* Quaternions are handled in **wxyz** order unless a function name says
  otherwise.  Robosuite's HDF5 observations store xyzw, so the dataset converts
  once at load time.
* The 9-D low-dim state is ``[pos(3), quat_wxyz(4), gripper(2)]``, matching the
  B1 reference's sorted low-dim keys.
* The 4-step camera-frame EEF history per step is
  ``[R_c^T (p - cam_pos) (3), rot6d(R_c^T R_eef) (6), gripper (2)]`` = 11 dims,
  step-major and oldest-first -> 44 dims.
"""
from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EEF_STEP_DIM = 11
EEF_HIST_STEPS = 4
EEF_HIST_DIM = EEF_STEP_DIM * EEF_HIST_STEPS  # 44
EEF_STATE_DIM = 9
FUSED_DIM = 512
OBS_FEATURE_DIM = 521
HORIZON = 16
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
IMAGE_SIZE = 256
CROP_MIN_SIDE = int(IMAGE_SIZE * 0.8)  # 204, mirrors CamPose's RandomCrop
CROP_MAX_SIDE = IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------
def quat_xyzw_to_wxyz(q):
    """(..., 4) xyzw -> (..., 4) wxyz."""
    q = np.asarray(q, dtype=np.float32)
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_mat(q):
    """(..., 4) wxyz -> (..., 3, 3) rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate quaternion with norm < 1e-8")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def mat_to_rot6d(R):
    """(..., 3, 3) -> (..., 6): the first two rows, flattened."""
    R = np.asarray(R)
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def rot6d_to_mat(d6):
    """(..., 6) -> (..., 3, 3), inverse of ``mat_to_rot6d`` up to Gram-Schmidt."""
    d6 = np.asarray(d6, dtype=np.float64)
    r0 = d6[..., :3]
    r1 = d6[..., 3:]
    r0 = r0 / np.linalg.norm(r0, axis=-1, keepdims=True)
    r1 = r1 - np.sum(r0 * r1, axis=-1, keepdims=True) * r0
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = np.cross(r0, r1)
    return np.stack([r0, r1, r2], axis=-2)


def mat_to_quat_wxyz(R):
    """(3, 3) rotation matrix -> (4,) wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        R = R.reshape(3, 3)
    return quat_xyzw_to_wxyz(Rotation.from_matrix(R).as_quat())


def cam_c2w_to_pos_quat(T):
    """(4, 4) camera-to-world -> (pos (3,), quat_wxyz (4,))."""
    T = np.asarray(T, dtype=np.float32)
    pos = T[:3, 3].copy()
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw_to_wxyz(quat_xyzw)


def c2w_from_pos_quat(pos, quat_wxyz):
    """(3,), (4,) wxyz -> (4, 4) camera-to-world."""
    R = quat_wxyz_to_mat(quat_wxyz)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float32)
    return T


# ---------------------------------------------------------------------------
# EEF state / history
# ---------------------------------------------------------------------------
def eef_state(eef_pos, quat_wxyz, grip):
    """(..., 3), (..., 4), (..., 2) -> (..., 9) in ``[pos, quat, grip]`` order."""
    return np.concatenate(
        [np.asarray(eef_pos), np.asarray(quat_wxyz), np.asarray(grip)], axis=-1
    )


def eef_hist_to_cam(eef_pos, eef_quat_wxyz, grip, cam_pos, cam_quat_wxyz):
    """One history step in the camera frame -> (..., 11).

    ``[R_c^T (p - cam_pos), rot6d(R_c^T R_eef), gripper]``, matching
    ``diffusion_policy/dataset/multiview_image_dataset.py:eef_hist_to_cam``.
    """
    R_c = quat_wxyz_to_mat(cam_quat_wxyz)
    R_e = quat_wxyz_to_mat(eef_quat_wxyz)
    p = R_c.T @ (np.asarray(eef_pos, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    R_rel = R_c.T @ R_e
    return np.concatenate([p, mat_to_rot6d(R_rel), np.asarray(grip, dtype=np.float64)])


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------
def history_indices(t_obs: int, steps: int = EEF_HIST_STEPS) -> np.ndarray:
    """The ``steps`` history indices ending at ``t_obs``, clipped at episode start."""
    idx = np.arange(t_obs - steps + 1, t_obs + 1, dtype=np.int64)
    return np.clip(idx, 0, t_obs)


def action_window_indices(t: int, horizon: int = HORIZON, length: int = None):
    """Action window used by B1: starts at ``t-1`` and spans ``horizon`` steps.

    Returns ``(safe_indices, valid)``.  ``safe_indices`` clips out-of-range
    entries so callers can index safely; ``valid`` marks the entries that are
    within the episode.  The first entry is invalid only when ``t == 0``.
    """
    if length is None:
        raise ValueError("length is required")
    start = int(t) - 1
    idx = start + np.arange(horizon, dtype=np.int64)
    valid = (idx >= 0) & (idx < int(length))
    safe = np.clip(idx, 0, max(int(length) - 1, 0))
    return safe, valid


# ---------------------------------------------------------------------------
# Shared crop for the two observation frames
# ---------------------------------------------------------------------------
def shared_crop_params(
    min_side: int = CROP_MIN_SIDE,
    max_side: int = CROP_MAX_SIDE,
    output_size: int = IMAGE_SIZE,
) -> Tuple[int, int, int, int]:
    """Sample one crop and reuse it for both frames of a view."""
    crop_size = random.randint(min_side, max_side)
    top = random.randint(0, IMAGE_SIZE - crop_size)
    left = random.randint(0, IMAGE_SIZE - crop_size)
    return crop_size, top, left, output_size


def apply_shared_crop(img_chw: torch.Tensor, params: Tuple[int, int, int, int]) -> torch.Tensor:
    """Apply a ``shared_crop_params`` crop to a (C, H, W) tensor with NEAREST resize."""
    crop_size, top, left, output_size = params
    cropped = img_chw[:, top : top + crop_size, left : left + crop_size]
    return TF.resize(
        cropped,
        [output_size, output_size],
        interpolation=TF.InterpolationMode.NEAREST,
    )


# ---------------------------------------------------------------------------
# BatchNorm -> GroupNorm (copied from models/dp.py to keep this module
# diffusers-free; the reference B1 encoder uses GroupNorm + no pretrained BN)
# ---------------------------------------------------------------------------
def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))
        src_module = (
            parent_module[int(k)]
            if isinstance(parent_module, nn.Sequential)
            else getattr(parent_module, k)
        )
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module


def replace_bn_with_gn(root_module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm, as in the B1 reference."""
    return _replace_submodules(
        root_module,
        predicate=lambda m: isinstance(m, nn.BatchNorm2d),
        func=lambda m: nn.GroupNorm(
            num_groups=max(1, m.num_features // 16), num_channels=m.num_features
        ),
    )


# ---------------------------------------------------------------------------
# Environment duck-typing helpers (no robosuite import)
# ---------------------------------------------------------------------------
def find_eef_site_id(env) -> int:
    """Best-effort EEF site id for the robosuite robot."""
    model = env.sim.model
    for name in ("gripper0_right_grip_site", "gripper0_grip_site", "grip_site"):
        try:
            return int(model.site_name2id(name))
        except Exception:
            pass
    try:
        return int(env.robots[0].eef_site_id)
    except Exception:
        pass
    for i in range(model.nsite):
        if "grip" in model.site_id2name(i).lower():
            return int(i)
    raise RuntimeError("could not locate the EEF site; pass an explicit site id")


def find_eef_body_name(env) -> str:
    """Name of the robot's EEF body.

    Robosuite's ``robot0_eef_quat`` observation (the key stored in the demo
    HDF5s and used for training) is the *body* quaternion from
    ``sim.data.get_body_xquat``, not the gripper-site orientation.  Evaluation
    must use the same body or the EEF state/history is ~90 degrees off.
    """
    robot = env.robots[0]
    names = getattr(robot.robot_model, "eef_name", None)
    if isinstance(names, dict):
        names = next(iter(names.values()), None)
    if names is None:
        raise RuntimeError("could not locate the EEF body name")
    return str(names)


def find_gripper_qpos_indices(env) -> np.ndarray:
    """Indices of the two gripper joints in ``sim.data.qpos``."""
    robot = env.robots[0]
    idx = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if isinstance(idx, dict):
        # This robosuite fork stores per-arm index lists keyed by arm name.
        idx = next(iter(idx.values()), None)
    if idx is not None:
        return np.asarray(idx, dtype=np.int64).reshape(-1)
    # Panda: 7 arm joints followed by 2 gripper joints.
    return np.arange(7, 9, dtype=np.int64)
