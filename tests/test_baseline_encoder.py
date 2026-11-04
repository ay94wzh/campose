"""CPU-only tests for the `baseline` (B1 + DP) implementation.

Run from the repo root (no GPU, simulator, or demos required):

    /home/zihan-wang/anaconda3/envs/know_your_camera/bin/python \
        tests/test_baseline_encoder.py

The encoder/geometry tests import no robosuite and no diffusers.  The policy
shape test is skipped if diffusers is unavailable.
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(REPO_ROOT, "policy_robosuite")
if POLICY_ROOT not in sys.path:
    sys.path.insert(0, POLICY_ROOT)

from baseline_common import (  # noqa: E402
    EEF_HIST_STEPS,
    action_window_indices,
    apply_shared_crop,
    cam_c2w_to_pos_quat,
    eef_hist_to_cam,
    history_indices,
    mat_to_rot6d,
    rot6d_to_mat,
    shared_crop_params,
)
from models.baseline import BaselineDiffusionPolicy, BaselineEncoder  # noqa: E402


def _make_inputs(n_frames, k, seed=0):
    torch.manual_seed(seed)
    image = torch.rand(n_frames, k, 9, 256, 256)
    eef_state = torch.randn(n_frames, 9)
    eef_hist = torch.randn(n_frames, k, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(n_frames, k, dtype=torch.bool)
    return image, eef_state, eef_hist, view_mask


def test_common_geometry():
    pos, quat = cam_c2w_to_pos_quat(np.eye(4, dtype=np.float32))
    assert np.allclose(pos, 0.0)
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])

    step = eef_hist_to_cam(
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    expected = np.array([1, 2, 3, 1, 0, 0, 0, 1, 0, 0.1, 0.2], dtype=np.float64)
    assert np.allclose(step, expected), step

    R = Rotation.random(random_state=0).as_matrix()
    assert np.allclose(rot6d_to_mat(mat_to_rot6d(R)), R, atol=1e-10)
    print("ok common geometry")


def test_index_helpers():
    assert np.array_equal(history_indices(0), [0, 0, 0, 0])
    assert np.array_equal(history_indices(5), [2, 3, 4, 5])

    idx, valid = action_window_indices(0, 16, 100)
    assert not valid[0]
    assert valid[1]
    assert idx[0] == 0

    idx, valid = action_window_indices(5, 16, 100)
    assert valid.all()
    assert idx[0] == 4

    _, valid = action_window_indices(99, 16, 100)
    assert valid.sum() == 2
    print("ok index helpers")


def test_shared_crop():
    x = torch.rand(9, 256, 256)
    params = shared_crop_params()
    y1 = apply_shared_crop(x, params)
    y2 = apply_shared_crop(x, params)
    assert y1.shape == (9, 256, 256)
    assert torch.equal(y1, y2)
    print("ok shared crop")


def test_encoder_shapes():
    for k in (1, 2):
        image, eef_state, eef_hist, view_mask = _make_inputs(2, k)
        model = BaselineEncoder().eval()
        with torch.no_grad():
            out = model(image, eef_state, eef_hist, view_mask)
            full = model.forward_full(image, eef_state, eef_hist, view_mask)
        assert out.shape == (2, 521), out.shape
        assert full["z_global"].shape == (2, 512)
        assert full["z_views"].shape == (2, k, 512)
        assert torch.isfinite(out).all()
        print(f"ok encoder shapes k={k}")


def test_film_identity_at_init():
    image, eef_state, eef_hist, view_mask = _make_inputs(2, 1)
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_random = model(image, eef_state, eef_hist, view_mask)
        out_zero = model(image, eef_state, torch.zeros_like(eef_hist), view_mask)
    assert torch.allclose(out_random, out_zero, atol=1e-5), (
        (out_random - out_zero).abs().max().item()
    )
    print("ok zero-init FiLM identity")


def test_masked_view_equivalence():
    image, eef_state, eef_hist, _ = _make_inputs(2, 2)
    mask_second_off = torch.tensor([[True, False], [True, False]])
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_masked = model(image, eef_state, eef_hist, mask_second_off)
        out_single = model(
            image[:, :1],
            eef_state,
            eef_hist[:, :1],
            torch.ones(2, 1, dtype=torch.bool),
        )
    assert torch.allclose(out_masked, out_single, atol=1e-5), (
        (out_masked - out_single).abs().max().item()
    )
    print("ok masked-view equivalence")


def test_policy_shapes():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip policy shapes: diffusers not installed")
        return

    class Args:
        action_dim = 7
        lr = 1e-4
        weight_decay = 1e-6

    policy = BaselineDiffusionPolicy(
        Args(),
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
    ).eval()

    image = torch.rand(1, 2, 1, 9, 256, 256)
    eef_state = torch.randn(1, 2, 9)
    eef_hist = torch.randn(1, 2, 1, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    actions = torch.randn(1, 16, 7)
    is_pad = torch.zeros(1, 16, dtype=torch.bool)

    out = policy(
        {
            "image": image,
            "eef_state": eef_state,
            "eef_hist": eef_hist,
            "view_mask": view_mask,
            "actions": actions,
            "is_pad": is_pad,
        }
    )
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])

    with torch.no_grad():
        act = policy(
            {
                "image": image,
                "eef_state": eef_state,
                "eef_hist": eef_hist,
                "view_mask": view_mask,
            }
        )
    assert act.shape == (1, 8, 7), act.shape
    print("ok policy train/inference shapes")


if __name__ == "__main__":
    assert "robosuite" not in sys.modules, "baseline tests must not import robosuite"
    test_common_geometry()
    test_index_helpers()
    test_shared_crop()
    test_encoder_shapes()
    test_film_identity_at_init()
    test_masked_view_equivalence()
    test_policy_shapes()
    print("all baseline tests passed")
