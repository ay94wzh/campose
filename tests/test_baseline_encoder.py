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
    OBS_FEATURE_DIM,
    PROPRIO_DIM,
    SpatialSoftmax,
    action_window_indices,
    apply_shared_crop,
    cam_c2w_to_pos_quat,
    eef_hist_array,
    eef_hist_to_cam,
    history_indices,
    mat_to_rot6d,
    normalize_proprio,
    rot6d_to_mat,
    shared_crop_params,
)
from models.baseline import BaselineDiffusionPolicy, BaselineEncoder  # noqa: E402


def _make_inputs(n_frames, k, seed=0):
    torch.manual_seed(seed)
    image = torch.rand(n_frames, k, 9, 256, 256)
    proprio = torch.randn(n_frames, PROPRIO_DIM)
    eef_hist = torch.randn(n_frames, k, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(n_frames, k, dtype=torch.bool)
    return image, proprio, eef_hist, view_mask


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


def test_eef_hist_array_matches_loop():
    rng = np.random.RandomState(0)
    T = 12
    pos = rng.randn(T, 3)
    quat = rng.randn(T, 4)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    grip = rng.randn(T, 2)
    cam_pos = rng.randn(3)
    cam_quat = rng.randn(4)
    cam_quat /= np.linalg.norm(cam_quat)

    hist = eef_hist_array(pos, quat, grip, cam_pos, cam_quat)
    assert hist.shape == (T, EEF_HIST_STEPS * 11), hist.shape
    for t in range(T):
        loop = np.concatenate(
            [
                eef_hist_to_cam(pos[i], quat[i], grip[i], cam_pos, cam_quat)
                for i in history_indices(t, EEF_HIST_STEPS)
            ]
        )
        assert np.allclose(hist[t], loop, atol=1e-10), (
            t,
            np.abs(hist[t] - loop).max(),
        )
    print("ok vectorised eef history")


def test_history_is_past_only():
    rng = np.random.RandomState(1)
    T = 10
    pos = rng.randn(T, 3)
    quat = rng.randn(T, 4)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    grip = rng.randn(T, 2)
    cam_pos = rng.randn(3)
    cam_quat = rng.randn(4)
    cam_quat /= np.linalg.norm(cam_quat)

    hist = eef_hist_array(pos, quat, grip, cam_pos, cam_quat)
    for t in range(T):
        assert np.all(history_indices(t, EEF_HIST_STEPS) <= t)
        pos2, quat2, grip2 = pos.copy(), quat.copy(), grip.copy()
        pos2[t + 1:] += 100.0
        quat2[t + 1:] += 5.0
        grip2[t + 1:] += 100.0
        hist2 = eef_hist_array(pos2, quat2, grip2, cam_pos, cam_quat)
        assert np.array_equal(hist[t], hist2[t]), t
    print("ok past-only history")


def test_index_helpers():
    assert np.array_equal(history_indices(0), [0, 0, 0, 0])
    assert np.array_equal(history_indices(5), [2, 3, 4, 5])

    idx, valid = action_window_indices(0, 16, 100)
    assert valid.all()
    assert idx[0] == 0

    idx, valid = action_window_indices(5, 16, 100)
    assert valid.all()
    assert idx[0] == 5

    _, valid = action_window_indices(99, 16, 100)
    assert valid.sum() == 1
    print("ok index helpers")


def test_shared_crop():
    x = torch.rand(9, 256, 256)
    params = shared_crop_params()
    y1 = apply_shared_crop(x, params)
    y2 = apply_shared_crop(x, params)
    assert y1.shape == (9, 256, 256)
    assert torch.equal(y1, y2)
    print("ok shared crop")


def test_spatial_softmax_operator():
    # num_kp=None keeps each input channel as a keypoint, so the test is
    # deterministic: a softmax peak at a grid cell yields that cell's (x, y).
    op = SpatialSoftmax((1, 8, 8), num_kp=None)
    a = torch.zeros(1, 1, 8, 8)
    b = torch.zeros(1, 1, 8, 8)
    a[0, 0, 1, 1] = 10.0
    b[0, 0, 6, 6] = 10.0
    ka = op(a).flatten()
    kb = op(b).flatten()
    assert ka.shape == (2,)
    assert abs(ka[0].item() + 0.714) < 1e-2
    assert abs(ka[1].item() + 0.714) < 1e-2
    assert abs(kb[0].item() - 0.714) < 1e-2
    assert abs(kb[1].item() - 0.714) < 1e-2

    pooled = SpatialSoftmax((4, 8, 8), num_kp=3)(torch.randn(2, 4, 8, 8))
    assert pooled.shape == (2, 3, 2)
    print("ok spatial softmax")


def test_proprio_drop_before_zscore():
    raw = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    mean = np.full(PROPRIO_DIM, 0.5, dtype=np.float32)
    std = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    kept = normalize_proprio(raw, mean, std, drop=False)
    dropped = normalize_proprio(raw, mean, std, drop=True)
    assert np.allclose(kept, (raw - mean) / std)
    # DP zeroes the raw qpos *before* normalisation, so the constant is
    # -state_mean/state_std, not zero.
    assert np.allclose(dropped, -mean / std)
    assert not np.allclose(dropped, np.zeros_like(raw))
    print("ok proprio drop before z-score")


def test_encoder_shapes():
    for k in (1, 2):
        image, proprio, eef_hist, view_mask = _make_inputs(2, k)
        model = BaselineEncoder().eval()
        with torch.no_grad():
            out = model(image, proprio, eef_hist, view_mask)
            full = model.forward_full(image, proprio, eef_hist, view_mask)
        assert out.shape == (2, OBS_FEATURE_DIM), out.shape
        assert full["z_global"].shape == (2, 512)
        assert full["z_views"].shape == (2, k, 512)
        assert torch.isfinite(out).all()
        print(f"ok encoder shapes k={k}")


def test_no_imagenet_rgb_norm():
    model = BaselineEncoder().eval()
    assert not hasattr(model, "rgb_mean")
    assert not hasattr(model, "rgb_std")
    image = torch.rand(1, 1, 9, 256, 256)
    assert torch.equal(model._prepare_image(image), image)
    print("ok no ImageNet RGB normalisation")


def test_film_identity_at_init():
    image, proprio, eef_hist, view_mask = _make_inputs(2, 1)
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_random = model(image, proprio, eef_hist, view_mask)
        out_zero = model(image, proprio, torch.zeros_like(eef_hist), view_mask)
    assert torch.allclose(out_random, out_zero, atol=1e-5), (
        (out_random - out_zero).abs().max().item()
    )
    print("ok zero-init FiLM identity")


def test_masked_view_equivalence():
    image, proprio, eef_hist, _ = _make_inputs(2, 2)
    mask_second_off = torch.tensor([[True, False], [True, False]])
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_masked = model(image, proprio, eef_hist, mask_second_off)
        out_single = model(
            image[:, :1],
            proprio,
            eef_hist[:, :1],
            torch.ones(2, 1, dtype=torch.bool),
        )
    assert torch.allclose(out_masked, out_single, atol=1e-5), (
        (out_masked - out_single).abs().max().item()
    )
    print("ok masked-view equivalence")


def test_encoder_cpu_autocast():
    image, proprio, eef_hist, view_mask = _make_inputs(1, 1)
    model = BaselineEncoder().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(image, proprio, eef_hist, view_mask)
    assert out.shape == (1, OBS_FEATURE_DIM), out.shape
    assert torch.isfinite(out.float()).all()
    print("ok encoder CPU autocast")


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

    image = torch.rand(1, 1, 1, 9, 256, 256)
    proprio = torch.randn(1, 1, PROPRIO_DIM)
    eef_hist = torch.randn(1, 1, 1, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    actions = torch.randn(1, 16, 7)
    is_pad = torch.zeros(1, 16, dtype=torch.bool)

    out = policy(
        {
            "image": image,
            "proprio": proprio,
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
                "proprio": proprio,
                "eef_hist": eef_hist,
                "view_mask": view_mask,
            }
        )
    assert act.shape == (1, 8, 7), act.shape
    print("ok policy train/inference shapes")


if __name__ == "__main__":
    assert "robosuite" not in sys.modules, "baseline tests must not import robosuite"
    test_common_geometry()
    test_eef_hist_array_matches_loop()
    test_history_is_past_only()
    test_index_helpers()
    test_shared_crop()
    test_spatial_softmax_operator()
    test_proprio_drop_before_zscore()
    test_encoder_shapes()
    test_no_imagenet_rgb_norm()
    test_film_identity_at_init()
    test_masked_view_equivalence()
    test_encoder_cpu_autocast()
    test_policy_shapes()
    print("all baseline tests passed")
